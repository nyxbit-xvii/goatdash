# streamlit_app.py (simplified, no sidebar; bird's-eye maps)
import os
import re
import json
import zipfile
import datetime as dt
from math import pi, log, tan
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
import altair as alt
from xml.etree import ElementTree as ET
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio

# ===================== CONFIG (edit here if needed) =====================
DASHBOARD_TITLE = "HerdTracker at Steep Mountain Farm — Bonus’s Weekly Recap"
DEFAULT_GPX_FILE = "assets/Bonus_Latest.gpx"
VIDEO_PATH = "assets/bonus_week.mp4"
ICON_URL = "https://raw.githubusercontent.com/nyxbit-xvii/goatdash/refs/heads/main/assets/bonus_icon.png"

MAPBOX_STYLE = "mapbox://styles/mapbox/satellite-streets-v11"
STATIC_IMAGE_STYLE = "satellite-streets-v11"
MAP_ZOOM = 18
MAP_PITCH = 0  # <-- bird's-eye for all maps
HEAT_RADIUS_PX = 40  # used only if HEATMAP_MODE == "heatmap"

# Density map choice: "hex" (3D hex bins) or "heatmap" (KDE)
HEATMAP_MODE = "hex"
HEX_RADIUS_M = 60   # hex/grid cell radius (meters)
ELEV_SCALE = 12     # 3D hex elevation scale

# Home/run polygons (from KMZ or GeoJSON)
KMZ_PATH = "assets/herd_home.kmz"          # contains "Ungulate Home" & "Ungulate Run"
GEOJSON_PATH = "assets/herd_home.geojson"  # optional fallback if present
HOME_NAMES = ["Ungulate Home", "Ungulate Run"]

# How we treat the home area on the density map:
HIDE_HOME_ON_DENSITY = True     # True = remove home/run points entirely
HOME_DAMP = 0.1                 # if not hiding, weight to assign inside home/run
HOME_DOWNSAMPLE = 10            # keep 1 in N home points (only used when not hiding)

# Pathline
TRAIL_POINTS = 250

# Static image (for timelapse)
STATIC_W, STATIC_H, STATIC_ZOOM = 800, 600, 18

# ===================== MAPBOX TOKEN =====================
mapbox_token = None
try:
    mapbox_token = st.secrets["MAPBOX_API_KEY"]
    pdk.settings.mapbox_api_key = mapbox_token
except Exception:
    pdk.settings.mapbox_api_key = None  # app still runs; maps blank without key

# ===================== HELPERS =====================
def parse_gpx_text(text: str) -> pd.DataFrame:
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    root = ET.fromstring(text)
    pts = root.findall(".//g:trkpt", ns)
    rows = []
    for p in pts:
        lat = p.attrib.get("lat")
        lon = p.attrib.get("lon")
        t = (p.find("g:time", ns).text if p.find("g:time", ns) is not None else None)
        c = (p.find("g:cmt", ns).text if p.find("g:cmt", ns) is not None else None)
        rows.append({"timestamp": t, "lat": lat, "lon": lon, "cmt": c})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No <trkpt> points found in GPX.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if "cmt" in df.columns:
        sp = df["cmt"].str.extract(r"speed:\s*([\d\.]+)")
        df["speed"] = pd.to_numeric(sp[0], errors="coerce")
    df = df.dropna(subset=["timestamp", "lat", "lon"]).sort_values("timestamp").reset_index(drop=True)
    return df

def parse_gpx(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        return parse_gpx_text(f.read())

@st.cache_data(ttl=3600)
def fetch_open_meteo(lat: float, lon: float, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": float(lat), "longitude": float(lon),
            "hourly": "temperature_2m,wind_speed_10m,precipitation",
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "timezone": "UTC",
        }
        r = requests.get(url, params=params, timeout=45)
        r.raise_for_status()
        data = r.json().get("hourly", {})
        if not data:
            return pd.DataFrame()
        wdf = pd.DataFrame(data)
        wdf["time"] = pd.to_datetime(wdf["time"], utc=True, errors="coerce")
        return wdf
    except requests.exceptions.RequestException:
        return pd.DataFrame()

def haversine_miles(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

def total_distance_miles(df: pd.DataFrame) -> float:
    if df is None or len(df) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(df)):
        dist += haversine_miles(
            df.iloc[i-1].lat,
            df.iloc[i-1].lon,
            df.iloc[i].lat,
            df.iloc[i].lon
        )
    return float(dist)

def build_interpolated_track(df: pd.DataFrame, step_minutes: int = 1):
    s = df.set_index("timestamp")[["lat", "lon"]].sort_index()
    start, end = s.index.min(), s.index.max()
    idx = pd.date_range(start, end, freq=f"{step_minutes}min", tz="UTC")
    interp = s.reindex(idx).interpolate(method="time").ffill().bfill()
    interp.index.name = "timestamp"
    return interp.reset_index()

# ---- Web mercator helpers (for timelapse frames) ----
TILE_SIZE = 256
def _lon_to_x(lon, z): return (lon + 180.0) / 360.0 * (2**z) * TILE_SIZE
def _lat_to_y(lat, z):
    lat = max(min(lat, 85.051129), -85.051129)
    return (1 - log(tan((lat + 90)*pi/360)) / pi) / 2 * (2**z) * TILE_SIZE
def latlon_to_pixel(lat, lon, center_lat, center_lon, zoom, width, height):
    cx, cy = _lon_to_x(center_lon, zoom), _lat_to_y(center_lat, zoom)
    x, y = _lon_to_x(lon, zoom), _lat_to_y(lat, zoom)
    px = (x - cx) + (width/2)
    py = (y - cy) + (height/2)
    return int(px), int(py)

@st.cache_data(ttl=3600)
def fetch_static_map(center_lat, center_lon, w=STATIC_W, h=STATIC_H, zoom=STATIC_ZOOM):
    if not mapbox_token:
        raise RuntimeError("Mapbox token missing; cannot fetch static image.")
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/{STATIC_IMAGE_STYLE}/static/"
        f"{center_lon},{center_lat},{zoom},0/{w}x{h}?access_token={mapbox_token}"
    )
    img = Image.open(BytesIO(requests.get(url, timeout=45).content)).convert("RGBA")
    return img

@st.cache_data(ttl=3600)
def fetch_icon():
    resp = requests.get(ICON_URL, timeout=30)
    im = Image.open(BytesIO(resp.content)).convert("RGBA")
    return im

def _get_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def render_mp4_safe(interp: pd.DataFrame,
                    center_lat: float,
                    center_lon: float,
                    outfile: str = VIDEO_PATH,
                    zoom: int = None,
                    w: int = None,
                    h: int = None,
                    step: int = 12,
                    fps: int = 10) -> str:
    if zoom is None: zoom = STATIC_ZOOM
    if w is None: w = STATIC_W
    if h is None: h = STATIC_H

    bg = fetch_static_map(center_lat, center_lon, w, h, zoom)
    icon = fetch_icon().resize((64, 64),
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
    font = _get_font()

    coords = [latlon_to_pixel(float(r.lat), float(r.lon), center_lat, center_lon, zoom, w, h)
              for r in interp.itertuples(index=False)]

    codecs = [dict(codec="libx264", pix_fmt="yuv420p"), dict(codec="mpeg4", pix_fmt="yuv420p")]
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    last_err = None
    for enc in codecs:
        try:
            with iio.imopen(outfile, "w", plugin="FFMPEG", fps=fps, **enc) as writer:
                for i in range(0, len(interp), step):
                    frame = bg.copy()
                    draw = ImageDraw.Draw(frame, "RGBA")
                    if i > 1:
                        start_idx = max(0, i - 80)
                        draw.line(coords[start_idx:i+1], fill=(0, 200, 255, 200), width=4)
                    x, y = coords[i]
                    frame.alpha_composite(icon, (x - icon.width//2, y - icon.height))
                    ts = interp.iloc[i]["timestamp"]
                    draw.rectangle([(10, h - 40), (380, h - 10)], fill=(0, 0, 0, 140))
                    text = f"Bonus — {ts.strftime('%Y-%m-%d %H:%M UTC')}"
                    if font:
                        draw.text((16, h - 34), text, fill=(255, 255, 255, 230), font=font)
                    else:
                        draw.text((16, h - 34), text, fill=(255, 255, 255, 230))
                    writer.write(np.array(frame))
            return outfile
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"FFmpeg failed (tried libx264 & mpeg4). Last error: {last_err}")

# ---------- KMZ / GeoJSON polygon utilities ----------
try:
    from shapely.geometry import shape, Point
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

def _kmz_to_geojson_features(kmz_path: str) -> List[dict]:
    """Extract Polygon features from the first KML inside a KMZ."""
    if not os.path.exists(kmz_path):
        return []
    with zipfile.ZipFile(kmz_path, "r") as z:
        kml_name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
        if not kml_name:
            return []
        data = z.read(kml_name)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    root = ET.fromstring(data)
    feats = []
    for pm in root.findall(".//kml:Placemark", ns):
        name_el = pm.find("kml:name", ns)
        name = (name_el.text or "").strip() if name_el is not None else "Unnamed"
        for poly in pm.findall(".//kml:Polygon", ns):
            outer = poly.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
            if outer is None or not outer.text:
                continue
            coords = []
            for trip in re.split(r"\s+", outer.text.strip()):
                if not trip:
                    continue
                parts = trip.split(",")
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    coords.append([lon, lat])
            if len(coords) >= 3:
                feats.append({
                    "type": "Feature",
                    "properties": {"name": name},
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                })
    return feats

def _features_by_name(features: List[dict], names: List[str]) -> List[dict]:
    wanted = {n.lower() for n in names}
    return [f for f in features if f.get("properties", {}).get("name", "").lower() in wanted]

def _polys_from_features(features: List[dict]) -> List[List[Tuple[float, float]]]:
    polys = []
    for f in features:
        geom = f.get("geometry", {})
        if geom.get("type") == "Polygon":
            ring = geom.get("coordinates", [[]])[0]
            polys.append([(float(lon), float(lat)) for lon, lat in ring])
        elif geom.get("type") == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                ring = poly[0]
                polys.append([(float(lon), float(lat)) for lon, lat in ring])
    return polys

# ---- Pure-Python point-in-polygon (ray casting) ----
def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    n = len(poly)
    if n < 3:
        return False
    inside = False
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        eps = 1e-12
        if min(x0, x1) - eps <= x <= max(x0, x1) + eps and min(y0, y1) - eps <= y <= max(y0, y1) + eps:
            dx = x1 - x0
            dy = y1 - y0
            if abs(dx) < eps and abs(x - x0) < eps:
                if min(y0, y1) - eps <= y <= max(y0, y1) + eps:
                    return True
            elif abs(dy) < eps and abs(y - y0) < eps:
                if min(x0, x1) - eps <= x <= max(x0, x1) + eps:
                    return True
            else:
                t = ((x - x0) * dy - (y - y0) * dx)
                if abs(t) < eps:
                    return True
        cond = ((y0 > y) != (y1 > y))
        if cond:
            x_intersect = x0 + (x1 - x0) * (y - y0) / (y1 - y0)
            if x_intersect >= x:
                inside = not inside
        x0, y0 = x1, y1
    return inside

def _points_in_any_polygon(lons: np.ndarray, lats: np.ndarray, polygons: List[List[Tuple[float, float]]]) -> np.ndarray:
    if len(polygons) == 0:
        return np.zeros(lons.shape, dtype=bool)
    if _HAS_SHAPELY:
        from shapely.geometry import shape, Point  # local import to avoid lint noise
        shapely_polys = [shape({"type": "Polygon", "coordinates": [poly]}) for poly in polygons]
        mask = np.zeros(lons.shape, dtype=bool)
        for i in range(lons.size):
            p = Point(float(lons[i]), float(lats[i]))
            if any(poly.covers(p) for poly in shapely_polys):
                mask[i] = True
        return mask
    mask = np.zeros(lons.shape, dtype=bool)
    for i in range(lons.size):
        x, y = float(lons[i]), float(lats[i])
        for poly in polygons:
            if _point_in_polygon(x, y, poly):
                mask[i] = True
                break
    return mask

# ===================== UI (no sidebar) =====================
st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
st.title(DASHBOARD_TITLE)

# Load GPX (repo file by default; optional upload in an expander)
try:
    df = parse_gpx(DEFAULT_GPX_FILE)
except Exception:
    with st.expander("Upload a GPX (optional)"):
        up = st.file_uploader("Upload a .gpx file", type=["gpx"])
        if up:
            df = parse_gpx_text(up.read().decode("utf-8"))
        else:
            st.stop()

if df.empty:
    st.error("Parsed GPX is empty after cleaning.")
    st.stop()

mid_lat, mid_lon = float(df["lat"].mean()), float(df["lon"].mean())

# KPIs (minimal)
dist_miles = total_distance_miles(df)
span = df["timestamp"].max() - df["timestamp"].min()
hours_span = int((span.total_seconds() // 3600) if pd.notna(span) else 0)
c1, c2, c3 = st.columns(3)
c1.metric("Distance", f"{dist_miles:.2f} mi")
c2.metric("Span", f"{hours_span} hrs")
c3.metric("Points", f"{len(df):,}")

# Weather + interpolation once (kept minimal)
interp = build_interpolated_track(df, step_minutes=1)
wdf = fetch_open_meteo(mid_lat, mid_lon, df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime())

# ===================== Outside Activity Density (home suppressed) =====================
st.subheader("Outside Activity (home suppressed)")

# Load home/run polygons
features = _kmz_to_geojson_features(KMZ_PATH) if os.path.exists(KMZ_PATH) else []
if not features and os.path.exists(GEOJSON_PATH):
    try:
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            gj = json.load(f)
        features = gj.get("features", [])
    except Exception as e:
        features = []
home_feats = _features_by_name(features, HOME_NAMES)
home_polys = _polys_from_features(home_feats)

lats = df["lat"].to_numpy(float)
lons = df["lon"].to_numpy(float)
home_mask = _points_in_any_polygon(lons, lats, home_polys) if home_polys else np.zeros(lons.shape, dtype=bool)

# Build focused dataset
if HIDE_HOME_ON_DENSITY:
    focus_df = df.loc[~home_mask].copy()
else:
    # down-weight + thin home points
    focus_df = df.copy()
    if HOME_DOWNSAMPLE > 1 and home_polys:
        idx_home = np.flatnonzero(home_mask)
        keep = np.zeros(home_mask.shape, dtype=bool)
        keep[idx_home[::int(HOME_DOWNSAMPLE)]] = True
        keep |= ~home_mask
        focus_df = df.loc[keep].copy()
    weights = np.where(home_mask, HOME_DAMP, 1.0).astype(float)
    focus_df["weight_heat"] = weights[keep] if (not HIDE_HOME_ON_DENSITY and HOME_DOWNSAMPLE > 1 and home_polys) else weights[~home_mask | home_mask]

# Build layer
view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=MAP_ZOOM, pitch=MAP_PITCH)

if HEATMAP_MODE == "hex":
    density_data = (
        focus_df[["lon", "lat"]]
        .rename(columns={"lon": "longitude", "lat": "latitude"})
        .dropna()
        .astype({"latitude": "float64", "longitude": "float64"})
    )
    # use uniform weight 1.0 (outside only) or custom if you kept home points
    if "weight_heat" in focus_df.columns:
        density_data["weight"] = focus_df["weight_heat"].astype("float64").to_numpy()
        get_weight = "weight"
    else:
        density_data["weight"] = 1.0
        get_weight = "weight"

    layer = pdk.Layer(
        "HexagonLayer",
        data=density_data,
        get_position='[longitude, latitude]',
        get_weight=get_weight,
        radius=HEX_RADIUS_M,
        elevation_scale=int(ELEV_SCALE),
        extruded=True,
        pickable=False,
        coverage=1.0,
    )
else:
    heat_data = (
        focus_df[["lon", "lat"]]
        .rename(columns={"lon": "longitude", "lat": "latitude"})
        .dropna()
        .astype({"latitude": "float64", "longitude": "float64"})
    )
    heat_data["weight"] = 1.0 if HIDE_HOME_ON_DENSITY else focus_df["weight_heat"].astype("float64").to_numpy()
    layer = pdk.Layer(
        "HeatmapLayer",
        data=heat_data,
        get_position='[longitude, latitude]',
        get_weight='weight',
        radius_pixels=int(HEAT_RADIUS_PX),
        aggregation="MEAN",
        color_range=[[0,255,0,160],[255,255,0,180],[255,128,0,200],[255,0,0,220]],
    )

st.pydeck_chart(pdk.Deck(map_style=MAPBOX_STYLE, layers=[layer], initial_view_state=view))
st.caption("Shows where Bonus goes outside the Ungulate Home/Run. (Bird’s-eye view)")

st.markdown("---")

# ===================== Inspect a Time (icon at nearest minute) =====================
st.subheader("Inspect a Time (UTC)")
# choose the earliest hour by default; no slider to keep things simple
sel_time = df["timestamp"].min().floor("h")

i_idx = (interp["timestamp"] - sel_time).abs().idxmin()
cur_lat, cur_lon = float(interp.loc[i_idx, "lat"]), float(interp.loc[i_idx, "lon"])

icon_df = pd.DataFrame([{
    "longitude": cur_lon, "latitude": cur_lat,
    "icon_data": {"url": ICON_URL, "width": 256, "height": 256, "anchorY": 256}
}])
icon_layer = pdk.Layer(
    "IconLayer",
    data=icon_df,
    get_icon="icon_data",
    get_size=12,
    size_scale=14,
    get_position="[longitude, latitude]",
)
st.pydeck_chart(pdk.Deck(map_style=MAPBOX_STYLE, layers=[icon_layer],
                         initial_view_state=pdk.ViewState(latitude=cur_lat, longitude=cur_lon, zoom=MAP_ZOOM, pitch=MAP_PITCH)))
if not wdf.empty:
    base = alt.Chart(wdf).encode(x=alt.X("time:T", title="Time (UTC)"))
    line_temp = base.mark_line(color="red").encode(y=alt.Y("temperature_2m:Q", title="Temp °C"))
    line_wind = base.mark_line(color="blue").encode(y=alt.Y("wind_speed_10m:Q", title="Wind m/s"))
    line_precip = base.mark_line(color="green").encode(y=alt.Y("precipitation:Q", title="Precip mm"))
    rule = alt.Chart(pd.DataFrame({"time": [sel_time]})).mark_rule(color="white", strokeDash=[4, 4]).encode(x="time:T")
    chart = alt.layer(line_temp, line_wind, line_precip, rule).resolve_scale(y="independent").properties(height=220)
    st.altair_chart(chart, use_container_width=True)
st.markdown("---")

# ===================== Breadcrumb Pathline =====================
st.subheader("Breadcrumb Pathline")
trail_df = interp.copy()
trail_df["delta"] = (trail_df["timestamp"] - sel_time).abs()
trail_df = trail_df.sort_values("delta").head(TRAIL_POINTS).sort_values("timestamp")
path_data = [{"path": trail_df[["lon","lat"]].to_numpy().tolist()}]

path_layer = pdk.Layer(
    "PathLayer",
    data=path_data,
    get_path="path",
    get_width=4,
    width_units="pixels",
    get_color=[0, 200, 255, 200],
)
icon_layer2 = pdk.Layer(
    "IconLayer",
    data=pd.DataFrame([{
        "longitude": cur_lon, "latitude": cur_lat,
        "icon_data": {"url": ICON_URL, "width": 256, "height": 256, "anchorY": 256}
    }]),
    get_icon="icon_data",
    get_size=12,
    size_scale=14,
    get_position="[longitude, latitude]",
)
st.pydeck_chart(pdk.Deck(map_style=MAPBOX_STYLE, layers=[path_layer, icon_layer2],
                         initial_view_state=pdk.ViewState(latitude=cur_lat, longitude=cur_lon, zoom=MAP_ZOOM, pitch=MAP_PITCH)))
st.markdown("---")

# ===================== Timelapse (optional) =====================
st.subheader("Weekly Timelapse — Bonus’s Recap")
if os.path.exists(VIDEO_PATH):
    st.video(VIDEO_PATH)
else:
    st.info("No video found at assets/bonus_week.mp4.")
with st.expander("Generate video (optional)"):
    st.caption("Renders a short MP4 over a static satellite image (needs Mapbox token + FFmpeg).")
    if st.button("Render MP4 now"):
        try:
            out = render_mp4_safe(interp, mid_lat, mid_lon, outfile=VIDEO_PATH, step=12, fps=10)
            st.success(f"Rendered: {out}")
            st.video(out)
        except Exception as e:
            st.error(f"Render failed: {e}")

# ===================== Footer =====================
st.markdown("---")
st.caption("Built with ❤️ at Steep Mountain Farm • Bonus the Explorer • Goatdash")

















