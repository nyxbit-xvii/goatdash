# streamlit_app.py
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

# ===================== CONFIG =====================
DASHBOARD_TITLE = "HerdTracker at Steep Mountain Farm — Bonus’s Weekly Recap"
DEFAULT_GPX_FILE = "assets/Bonus_Latest.gpx"
ICON_URL = "https://raw.githubusercontent.com/nyxbit-xvii/goatdash/refs/heads/main/assets/bonus_icon.png"

# Map styles / static map settings
MAPBOX_STYLE = "mapbox://styles/mapbox/satellite-streets-v11"
STATIC_IMAGE_STYLE = "satellite-streets-v11"
STATIC_W, STATIC_H, STATIC_ZOOM = 800, 600, 18

# Home/run polygons (KMZ or GeoJSON)
KMZ_PATH = "assets/herd_home.kmz"       # contains "Ungulate Home" and "Ungulate Run"
GEOJSON_PATH = "assets/herd_home.geojson"  # optional fallback if present

HOME_NAMES = ["Ungulate Home", "Ungulate Run"]

# ===================== MAPBOX TOKEN =====================
mapbox_token = None
try:
    mapbox_token = st.secrets["MAPBOX_API_KEY"]
    pdk.settings.mapbox_api_key = mapbox_token
except Exception:
    pdk.settings.mapbox_api_key = None  # allow app to run; maps blank without key

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
                    outfile: str = "assets/bonus_week.mp4",
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

    codecs = [
        dict(codec="libx264", pix_fmt="yuv420p"),
        dict(codec="mpeg4",  pix_fmt="yuv420p"),
    ]
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
    from matplotlib.path import Path

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

def _points_in_any_polygon(lons: np.ndarray, lats: np.ndarray, polygons: List[List[Tuple[float, float]]]) -> np.ndarray:
    if len(polygons) == 0:
        return np.zeros(lons.shape, dtype=bool)
    if _HAS_SHAPELY:
        shapely_polys = [shape({"type": "Polygon", "coordinates": [poly]}) for poly in polygons]
        mask = np.zeros(lons.shape, dtype=bool)
        for i in range(lons.size):
            p = Point(float(lons[i]), float(lats[i]))
            if any(poly.covers(p) for poly in shapely_polys):
                mask[i] = True
        return mask
    else:
        mask = np.zeros(lons.shape, dtype=bool)
        pts = np.c_[lons, lats]
        for poly in polygons:
            path = Path(poly, closed=True)
            mask |= path.contains_points(pts, radius=0.0)
        return mask

# ===================== UI =====================
st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
st.title(DASHBOARD_TITLE)

# Sidebar
st.sidebar.header("Controls")
if not mapbox_token:
    st.sidebar.warning("⚠️ Add MAPBOX_API_KEY to .streamlit/secrets.toml for maps.")

heat_radius = st.sidebar.slider("Heatmap radius (px)", 10, 120, 40, 5)
map_pitch = st.sidebar.slider("Map pitch (°)", 0, 60, 45, 5)
map_zoom = st.sidebar.slider("Map zoom", 12, 20, 18, 1)
trail_len_points = st.sidebar.slider("Breadcrumb trail length (points)", 20, 1000, 250, 10)

st.sidebar.markdown("---")
suppress_home = st.sidebar.toggle("Suppress home area on heatmap", value=True, help="Down-weight points inside Ungulate Home & Ungulate Run")
home_damp = st.sidebar.slider("Home dampening factor", 0.0, 1.0, 0.25, 0.05)

# GPX source
st.sidebar.markdown("---")
gpx_source = st.sidebar.radio("GPX source", ["Repo file", "Upload"], horizontal=True)
gpx_bytes = None
if gpx_source == "Upload":
    up = st.sidebar.file_uploader("Upload a .gpx file", type=["gpx"])
    if up is not None:
        gpx_bytes = up.read()

# Data ingest
try:
    if gpx_bytes:
        df = parse_gpx_text(gpx_bytes.decode("utf-8"))
    else:
        if not os.path.exists(DEFAULT_GPX_FILE):
            st.error("No GPX found. Upload a file in the sidebar or place one at assets/Bonus_Latest.gpx")
            st.stop()
        df = parse_gpx(DEFAULT_GPX_FILE)
except Exception as e:
    st.error(f"Failed to parse GPX: {e}")
    st.stop()

if df.empty:
    st.error("Parsed GPX is empty after cleaning.")
    st.stop()

mid_lat, mid_lon = float(df["lat"].mean()), float(df["lon"].mean())

# KPIs
dist_miles = total_distance_miles(df)
span = df["timestamp"].max() - df["timestamp"].min()
hours_span = int((span.total_seconds() // 3600) if pd.notna(span) else 0)
c1, c2, c3 = st.columns(3)
c1.metric("Distance traveled", f"{dist_miles:.2f} miles")
c2.metric("Time span", f"{hours_span} hours")
c3.metric("Points", f"{len(df):,}")

# Weather + interpolation
wdf = fetch_open_meteo(mid_lat, mid_lon, df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime())
interp = build_interpolated_track(df, step_minutes=1)

# ===================== 0) Daily Summary Panel =====================
st.subheader("0) Daily Summary Panel")

def _bin_zone(lat, lon, precision=3):
    return (round(lat, precision), round(lon, precision))

df["date"] = df["timestamp"].dt.date

def _daily_distance(g):
    if len(g) < 2:
        return 0.0
    return total_distance_miles(g[["timestamp","lat","lon"]].reset_index(drop=True))

daily = (
    df.groupby("date", as_index=False)
      .apply(lambda g: pd.Series({
          "distance_miles": _daily_distance(g),
          "points": len(g),
          "start": g["timestamp"].min(),
          "end": g["timestamp"].max()
      }))
      .reset_index(drop=True)
)
daily["active_hours"] = (daily["end"] - daily["start"]).dt.total_seconds() / 3600

df["zone"] = df.apply(lambda r: _bin_zone(r["lat"], r["lon"], precision=3), axis=1)
mv = (
    df.groupby(["date","zone"])
      .size()
      .reset_index(name="count")
      .sort_values(["date","count"], ascending=[True, False])
      .drop_duplicates(subset=["date"])
      .rename(columns={"zone":"most_visited_zone", "count":"zone_hits"})
)
daily = daily.merge(mv, on="date", how="left")

with st.expander("Daily details", expanded=True):
    dist_chart = alt.Chart(daily).mark_bar().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("distance_miles:Q", title="Distance (mi)"),
        tooltip=["date:T","distance_miles:Q","active_hours:Q","points:Q","most_visited_zone:N"]
    ).properties(height=140)
    st.altair_chart(dist_chart, use_container_width=True)

    for _, r in daily.iterrows():
        z = r["most_visited_zone"]
        ztxt = f"{z[0]:.3f}, {z[1]:.3f}" if isinstance(z, tuple) else "—"
        st.markdown(
            f"- **{r['date']}** — {r['distance_miles']:.2f} mi, "
            f"{r['active_hours']:.1f} hrs, pts {int(r['points'])}, "
            f"most-visited zone: `{ztxt}`"
        )

st.markdown("---")

# ===================== HEATMAP WEIGHTING =====================
# Build heatmap-specific weights (optionally suppress home/run)
lats = df["lat"].to_numpy(dtype=float)
lons = df["lon"].to_numpy(dtype=float)

weights_heat = np.ones(lons.shape, dtype=float)

if suppress_home:
    # Load polygons: try KMZ first, fall back to GeoJSON if present
    features = _kmz_to_geojson_features(KMZ_PATH) if os.path.exists(KMZ_PATH) else []
    if not features and os.path.exists(GEOJSON_PATH):
        try:
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                gj = json.load(f)
            features = gj.get("features", [])
        except Exception as e:
            st.warning(f"GeoJSON load failed: {e}")

    home_feats = _features_by_name(features, HOME_NAMES)
    home_polys = _polys_from_features(home_feats)

    if not home_polys:
        st.info("Home/run polygons not found in KMZ/GeoJSON; heatmap uses uniform weights.")
    else:
        home_mask = _points_in_any_polygon(lons, lats, home_polys)
        weights_heat[:] = 1.0
        # Down-weight inside home/run
        weights_heat[home_mask] = float(home_damp)

# Keep heatmap data minimal and serializable
df["weight_heat"] = weights_heat

# ===================== 1) Weekly Heatmap =====================
st.subheader("1) Weekly Heatmap")

heat_data = (
    df[["lat", "lon", "weight_heat"]]
      .rename(columns={"lat": "latitude", "lon": "longitude", "weight_heat": "weight"})
      .dropna()
      .astype({"latitude": "float64", "longitude": "float64", "weight": "float64"})
    # .to_dict(orient="records")  # uncomment if pydeck has serialization issues
)

heat_layer = pdk.Layer(
    "HeatmapLayer",
    data=heat_data,
    get_position='[longitude, latitude]',
    get_weight='weight',
    radius_pixels=int(heat_radius),
    aggregation="MEAN",
    color_range=[
        [0, 255, 0, 160],
        [255, 255, 0, 180],
        [255, 128, 0, 200],
        [255, 0, 0, 220],
    ],
)

heat_view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=map_zoom, pitch=map_pitch)
st.pydeck_chart(pdk.Deck(map_style=MAPBOX_STYLE, layers=[heat_layer], initial_view_state=heat_view))
st.caption("Toggle and tune suppression in the sidebar. Only affects the heatmap.")
st.markdown("---")

# ===================== 2) Inspect a Time + Weather =====================
st.subheader("2) Inspect a Time — Bonus moves to the selected hour")
start_ts = df["timestamp"].min().floor("h")
end_ts   = df["timestamp"].max().ceil("h")
all_hours = pd.date_range(start_ts, end_ts, freq="h", tz="UTC")
if len(all_hours) == 0:
    all_hours = pd.DatetimeIndex([start_ts])

sel_time = st.slider(
    "Select time (UTC)",
    min_value=all_hours[0].to_pydatetime(),
    max_value=all_hours[-1].to_pydatetime(),
    value=all_hours[0].to_pydatetime(),
    format="YYYY-MM-DD HH:mm",
)
sel_time = pd.Timestamp(sel_time)
sel_time = sel_time.tz_localize("UTC") if sel_time.tzinfo is None else sel_time.tz_convert("UTC")

# nearest in interpolated track
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
icon_view = pdk.ViewState(latitude=cur_lat, longitude=cur_lon, zoom=map_zoom, pitch=map_pitch)
st.pydeck_chart(pdk.Deck(map_style=MAPBOX_STYLE, layers=[icon_layer], initial_view_state=icon_view))

st.markdown("**Weather at the selected time**")
if not wdf.empty:
    base = alt.Chart(wdf).encode(x=alt.X("time:T", title="Time (UTC)"))
    line_temp = base.mark_line(color="red").encode(y=alt.Y("temperature_2m:Q", title="Temp °C"))
    line_wind = base.mark_line(color="blue").encode(y=alt.Y("wind_speed_10m:Q", title="Wind m/s"))
    line_precip = base.mark_line(color="green").encode(y=alt.Y("precipitation:Q", title="Precip mm"))
    rule = alt.Chart(pd.DataFrame({"time": [sel_time]})).mark_rule(color="white", strokeDash=[4, 4]).encode(x="time:T")
    chart = alt.layer(line_temp, line_wind, line_precip, rule).resolve_scale(y="independent").properties(height=260)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No weather data available for this window.")
st.markdown("---")

# ===================== 3) Breadcrumb Pathline =====================
st.subheader("3) Breadcrumb Pathline (sequential movement)")

trail_df = interp.copy()
trail_df["delta"] = (trail_df["timestamp"] - sel_time).abs()
trail_df = trail_df.sort_values("delta").head(trail_len_points).sort_values("timestamp")
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

view = pdk.ViewState(latitude=cur_lat, longitude=cur_lon, zoom=map_zoom, pitch=map_pitch)
st.pydeck_chart(pdk.Deck(map_style=MAPBOX_STYLE, layers=[path_layer, icon_layer2], initial_view_state=view))
st.caption("Trail ends at the selected time above. Adjust length in the sidebar.")
st.markdown("---")

# ===================== 4) Weekly Timelapse =====================
st.subheader("4) Weekly Timelapse — Bonus’s Weekly Recap")
VIDEO_PATH = "assets/bonus_week.mp4"
colv1, colv2 = st.columns([3,2], vertical_alignment="center")

with colv1:
    if os.path.exists(VIDEO_PATH):
        st.video(VIDEO_PATH)
    else:
        st.warning("No video found at assets/bonus_week.mp4. Generate below or add to repo.")

with colv2:
    if os.path.exists(VIDEO_PATH):
        with open(VIDEO_PATH, "rb") as f:
            st.download_button("⬇️ Download bonus_week.mp4", f, file_name="bonus_week.mp4", mime="video/mp4")
    with st.expander("Generate/refresh timelapse (experimental)"):
        st.caption("Renders an MP4 with a moving icon over a static satellite image. Needs Mapbox token + FFmpeg libs.")
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















