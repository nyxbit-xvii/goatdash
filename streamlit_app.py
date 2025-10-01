# streamlit_app.py

import os
import datetime as dt
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
from math import pi, log, tan

# ----------------- CONFIG -----------------
DEFAULT_GPX_FILE = "assets/Bonus_Latest.gpx"
DASHBOARD_TITLE = "HerdTracker at Steep Mountain Farm — Bonus’s Weekly Recap"
ICON_URL = "https://raw.githubusercontent.com/nyxbit-xvii/goatdash/refs/heads/main/assets/bonus_icon.png"

# Mapbox / styles
MAPBOX_STYLE = "mapbox://styles/mapbox/satellite-streets-v11"
STATIC_IMAGE_STYLE = "satellite-streets-v11"
STATIC_W, STATIC_H, STATIC_ZOOM = 800, 600, 18  # for static frames

# ----------------- MAPBOX TOKEN -----------------
mapbox_token = None
try:
    mapbox_token = st.secrets["MAPBOX_API_KEY"]
    pdk.settings.mapbox_api_key = mapbox_token
except Exception:
    # Allow app to start with a warning; pydeck will show a blank map without a token
    st.sidebar.warning("⚠️ MAPBOX_API_KEY not found in secrets. Add it to .streamlit/secrets.toml for maps.")
    pdk.settings.mapbox_api_key = None

# ----------------- HELPERS -----------------
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
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

def total_distance_miles(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    shifted = df.shift(1)
    dist = haversine_miles(df["lat"], df["lon"], shifted["lat"], shifted["lon"])
    if isinstance(dist, pd.Series):
        return float(dist.iloc[1:].sum())
    return float(dist)

def build_interpolated_track(df: pd.DataFrame, step_minutes: int = 1):
    s = df.set_index("timestamp")[["lat", "lon"]].sort_index()
    start, end = s.index.min(), s.index.max()
    idx = pd.date_range(start, end, freq=f"{step_minutes}min", tz="UTC")
    interp = s.reindex(idx).interpolate(method="time").ffill().bfill()
    interp.index.name = "timestamp"
    return interp.reset_index()

# --- Web mercator helpers for static map to pixels ---
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
    """Stream frames to FFmpeg. No huge lists in RAM."""
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
    last_err = None
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    for enc in codecs:
        try:
            with iio.imopen(outfile, "w", plugin="FFMPEG", fps=fps, **enc) as writer:
                for i in range(0, len(interp), step):
                    frame = bg.copy()
                    draw = ImageDraw.Draw(frame, "RGBA")
                    if i > 1:
                        start_idx = max(0, i - 80)  # short trail
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

# ----------------- UI -----------------
st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
st.title(DASHBOARD_TITLE)

# -------- Sidebar Controls --------
st.sidebar.header("Controls")
heat_radius = st.sidebar.slider("Heatmap radius (px)", 10, 120, 40, 5)
map_pitch = st.sidebar.slider("Map pitch (°)", 0, 60, 45, 5)
map_zoom = st.sidebar.slider("Map zoom", 12, 20, 18, 1)
trail_len_points = st.sidebar.slider("Breadcrumb trail length (points)", 20, 1000, 250, 10)

# GPX load choice
gpx_source = st.sidebar.radio("GPX source", ["Repo file", "Upload"], horizontal=True)
gpx_bytes = None
if gpx_source == "Upload":
    up = st.sidebar.file_uploader("Upload a .gpx file", type=["gpx"])
    if up is not None:
        gpx_bytes = up.read()

# -------- Data Ingest --------
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

# ----------------- 0) Daily Summary Panel -----------------
st.subheader("0) Daily Summary Panel")

def _bin_zone(lat, lon, precision=3):
    # ~precision=3 => ~100–150 m bins (roughly; depends on latitude)
    return (round(lat, precision), round(lon, precision))

if "date" not in df.columns:
    df["date"] = df["timestamp"].dt.date

# Distance per day
def _daily_distance(d):
    if len(d) < 2:
        return 0.0
    return total_distance_miles(d[["timestamp","lat","lon"]].reset_index(drop=True))

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

# Active hours ~ hours spanned with data
daily["active_hours"] = (daily["end"] - daily["start"]).dt.total_seconds() / 3600

# Most-visited zone per day
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

# Show table + compact cards
with st.expander("Daily details", expanded=True):
    # Chart distance per day
    dist_chart = alt.Chart(daily).mark_bar().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("distance_miles:Q", title="Distance (mi)"),
        tooltip=["date:T","distance_miles:Q","active_hours:Q","points:Q","most_visited_zone:N"]
    ).properties(height=140)
    st.altair_chart(dist_chart, use_container_width=True)

    # Small list
    for _, r in daily.iterrows():
        z = r["most_visited_zone"]
        ztxt = f"{z[0]:.3f}, {z[1]:.3f}" if isinstance(z, tuple) else "—"
        st.markdown(
            f"- **{r['date']}** — {r['distance_miles']:.2f} mi, "
            f"{r['active_hours']:.1f} hrs, pts {int(r['points'])}, "
            f"most-visited zone: `{ztxt}`"
        )

st.markdown("---")

# ----------------- 1) Weekly Heatmap -----------------
st.subheader("1) Weekly Heatmap")
heat_data = df.rename(columns={"lat": "latitude", "lon": "longitude"})
heat_layer = pdk.Layer(
    "HeatmapLayer",
    data=heat_data,
    get_position='[longitude, latitude]',
    get_weight=1,
    radius_pixels=int(heat_radius),
    aggregation="MEAN",
    color_range=[
        [0, 255, 0, 160],
        [255, 255, 0, 180],
        [255, 128, 0, 200],
        [255, 0, 0, 220]
    ],
)
heat_view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=map_zoom, pitch=map_pitch)
st.pydeck_chart(pdk.Deck(map_style=MAPBOX_STYLE, layers=[heat_layer], initial_view_state=heat_view))
st.markdown("---")

# ----------------- 2) Inspect a Time (icon jumps) + Weather -----------------
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

# tz-safe nearest
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
    st.info("No weather data available for this time window.")
st.markdown("---")

# ----------------- 3) Breadcrumb Pathline Map -----------------
st.subheader("3) Breadcrumb Pathline (sequential movement)")

# Build a short trail ending at sel_time
trail_df = interp.copy()
trail_df["delta"] = (trail_df["timestamp"] - sel_time).abs()
trail_df = trail_df.sort_values("delta").head(trail_len_points).sort_values("timestamp")
path_data = [{
    "path": trail_df[["lon","lat"]].to_numpy().tolist()
}]

path_layer = pdk.Layer(
    "PathLayer",
    data=path_data,
    get_path="path",
    get_width=4,
    width_units="pixels",
    get_color=[0, 200, 255, 200],
)

# Current icon on the same map
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
st.caption("Trail length is controlled in the sidebar. This draws a time-ordered breadcrumb ending at the selected time above.")
st.markdown("---")

# ----------------- 4) Weekly Timelapse — pre-generated MP4 -----------------
st.subheader("4) Weekly Timelapse — Bonus’s Weekly Recap")

VIDEO_PATH = "assets/bonus_week.mp4"
colv1, colv2 = st.columns([3,2], vertical_alignment="center")

with colv1:
    if os.path.exists(VIDEO_PATH):
        st.video(VIDEO_PATH)
    else:
        st.warning("No video found at assets/bonus_week.mp4. Generate below or pre-commit it to the repo.")

with colv2:
    if os.path.exists(VIDEO_PATH):
        with open(VIDEO_PATH, "rb") as f:
            st.download_button("⬇️ Download bonus_week.mp4", f, file_name="bonus_week.mp4", mime="video/mp4")
    with st.expander("Generate/refresh timelapse (experimental)"):
        st.caption("Renders a short MP4 with a moving icon over a static satellite image. Requires Mapbox token and FFmpeg libs.")
        if st.button("Render MP4 now"):
            try:
                out = render_mp4_safe(interp, mid_lat, mid_lon, outfile=VIDEO_PATH, step=12, fps=10)
                st.success(f"Rendered: {out}")
                st.video(out)
            except Exception as e:
                st.error(f"Render failed: {e}")

# ----------------- Footer -----------------
st.markdown("---")
st.caption("Built with ❤️ at Steep Mountain Farm • Bonus the Explorer • Goatdash")














