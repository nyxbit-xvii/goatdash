import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
import altair as alt
from xml.etree import ElementTree as ET
from io import BytesIO
from PIL import Image, ImageDraw
import imageio.v3 as iio

# ----------------- CONFIG -----------------
GPX_FILE = "assets/Bonus_Latest.gpx"
DASHBOARD_TITLE = "HerdTracker at Steep Mountain Farm — Bonus’s Weekly Recap"
ICON_URL = "https://raw.githubusercontent.com/nyxbit-xvii/goatdash/refs/heads/main/assets/bonus_icon.png"
MAPBOX_STYLE = "mapbox://styles/mapbox/satellite-streets-v11"
STATIC_IMAGE_STYLE = "satellite-streets-v11"
STATIC_W, STATIC_H, STATIC_ZOOM = 800, 600, 18  # smaller for MP4 export

# Mapbox token
pdk.settings.mapbox_api_key = st.secrets["MAPBOX_API_KEY"]
MAPBOX_TOKEN = st.secrets["MAPBOX_API_KEY"]

# ----------------- HELPERS -----------------
def parse_gpx(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
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
    if df.empty: raise ValueError("No <trkpt> points found in GPX.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if "cmt" in df.columns:
        sp = df["cmt"].str.extract(r"speed:\s*([\d\.]+)")
        df["speed"] = pd.to_numeric(sp[0], errors="coerce")
    df = df.dropna(subset=["timestamp", "lat", "lon"]).sort_values("timestamp").reset_index(drop=True)
    return df

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
        if not data: return pd.DataFrame()
        wdf = pd.DataFrame(data)
        wdf["time"] = pd.to_datetime(wdf["time"], utc=True, errors="coerce")
        return wdf
    except requests.exceptions.RequestException:
        return pd.DataFrame()

def haversine_miles(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2-lat1; dlon = lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

def total_distance_miles(df: pd.DataFrame) -> float:
    dist = 0.0
    for i in range(1, len(df)):
        dist += haversine_miles(df.iloc[i-1].lat, df.iloc[i-1].lon, df.iloc[i].lat, df.iloc[i].lon)
    return float(dist)

def build_interpolated_track(df: pd.DataFrame, step_minutes: int = 1):
    s = df.set_index("timestamp")[["lat", "lon"]].sort_index()
    start, end = s.index.min(), s.index.max()
    idx = pd.date_range(start, end, freq=f"{step_minutes}min", tz="UTC")
    interp = s.reindex(idx).interpolate(method="time").ffill().bfill()
    interp.index.name = "timestamp"
    return interp.reset_index()

# --- For MP4 generation ---
from math import pi, log, tan
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
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/{STATIC_IMAGE_STYLE}/static/"
        f"{center_lon},{center_lat},{zoom},0/{w}x{h}?access_token={MAPBOX_TOKEN}"
    )
    img = Image.open(BytesIO(requests.get(url, timeout=45).content)).convert("RGBA")
    return img

@st.cache_data(ttl=3600)
def fetch_icon():
    resp = requests.get(ICON_URL, timeout=30)
    im = Image.open(BytesIO(resp.content)).convert("RGBA")
    return im

def render_mp4(interp: pd.DataFrame, center_lat, center_lon,
               outfile="bonus_week.mp4", zoom=STATIC_ZOOM,
               w=STATIC_W, h=STATIC_H, step=10, fps=10):
    """Render an MP4 timelapse instead of a GIF."""
    bg = fetch_static_map(center_lat, center_lon, w, h, zoom)
    icon = fetch_icon().resize((64,64), Image.LANCZOS)
    frames = []
    coords = [latlon_to_pixel(float(r.lat), float(r.lon), center_lat, center_lon, zoom, w, h)
              for r in interp.itertuples(index=False)]

    for i in range(0, len(interp), step):
        base = bg.copy()
        draw = ImageDraw.Draw(base, "RGBA")
        if i > 1:
            draw.line(coords[max(0, i-100):i+1], fill=(0,200,255,200), width=4)
        x, y = coords[i]
        base.alpha_composite(icon, (x - icon.width//2, y - icon.height))
        ts = interp.iloc[i]["timestamp"]
        draw.rectangle([(10, h-40), (320, h-10)], fill=(0,0,0,140))
        draw.text((16, h-34), f"Bonus — {ts.strftime('%Y-%m-%d %H:%M UTC')}", fill=(255,255,255,230))
        frames.append(np.array(base))

    iio.imwrite(outfile, frames, fps=fps)
    return outfile

# ----------------- UI -----------------
st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
st.title(DASHBOARD_TITLE)

df = parse_gpx(GPX_FILE)
mid_lat, mid_lon = float(df["lat"].mean()), float(df["lon"].mean())

# KPIs
dist_miles = total_distance_miles(df)
span = df["timestamp"].max() - df["timestamp"].min()
hours_span = int((span.total_seconds() // 3600) if pd.notna(span) else 0)
c1, c2 = st.columns(2)
c1.metric("Distance traveled", f"{dist_miles:.2f} miles")
c2.metric("Time span", f"{hours_span} hours")

wdf = fetch_open_meteo(mid_lat, mid_lon, df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime())
interp = build_interpolated_track(df, step_minutes=1)

# 1) Heatmap
st.subheader("1) Weekly Heatmap")
heat_data = df.rename(columns={"lat":"latitude","lon":"longitude"})
heat_layer = pdk.Layer(
    "HeatmapLayer", data=heat_data,
    get_position='[longitude, latitude]', get_weight=1,
    radius_pixels=40, aggregation="MEAN",
    color_range=[[0,255,0,160],[255,255,0,180],[255,128,0,200],[255,0,0,220]],
)
heat_view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=18, pitch=45)
st.pydeck_chart(pdk.Deck(map_style=MAPBOX_STYLE, layers=[heat_layer], initial_view_state=heat_view))
st.markdown("---")

# 2) Inspect a time
st.subheader("2) Inspect a Time — Bonus moves to the selected hour")
start_ts = df["timestamp"].min().floor("H")
end_ts = df["timestamp"].max().ceil("H")
all_hours = pd.date_range(start_ts, end_ts, freq="H", tz="UTC")
if len(all_hours) == 0:
    all_hours = pd.DatetimeIndex([start_ts])

sel_time = st.slider("Select time (UTC)",
    min_value=all_hours[0].to_pydatetime(),
    max_value=all_hours[-1].to_pydatetime(),
    value=all_hours[0].to_pydatetime(),
    format="YYYY-MM-DD HH:mm"
)

sel_time = pd.Timestamp(sel_time)
if sel_time.tzinfo is None:
    sel_time = sel_time.tz_localize("UTC")
else:
    sel_time = sel_time.tz_convert("UTC")

i_idx = int(np.argmin(np.abs(interp["timestamp"].values - np.array(sel_time, dtype="datetime64[ns]"))))
cur_lat, cur_lon = float(interp.iloc[i_idx]["lat"]), float(interp.iloc[i_idx]["lon"])

icon_df = pd.DataFrame([{
    "longitude": cur_lon, "latitude": cur_lat,
    "icon_data": {"url": ICON_URL, "width": 256, "height": 256, "anchorY": 256}
}])
icon_layer = pdk.Layer(
    "IconLayer", data=icon_df,
    get_icon="icon_data", get_size=12, size_scale=14,
    get_position="[longitude, latitude]",
)
icon_view = pdk.ViewState(latitude=cur_lat, longitude=cur_lon, zoom=18, pitch=45)
st.pydeck_chart(pdk.Deck(map_style=MAPBOX_STYLE, layers=[icon_layer], initial_view_state=icon_view))

st.markdown("**Weather at the selected time**")
if not wdf.empty:
    base = alt.Chart(wdf).encode(x=alt.X("time:T", title="Time (UTC)"))
    line_temp = base.mark_line(color="red").encode(y=alt.Y("temperature_2m:Q", title="Temp °C"))
    line_wind = base.mark_line(color="blue").encode(y=alt.Y("wind_speed_10m:Q", title="Wind m/s"))
    line_precip = base.mark_line(color="green").encode(y=alt.Y("precipitation:Q", title="Precip mm"))
    rule = alt.Chart(pd.DataFrame({"time":[sel_time]})).mark_rule(color="white", strokeDash=[4,4]).encode(x="time:T")
    chart = alt.layer(line_temp, line_wind, line_precip, rule).resolve_scale(y="independent").properties(height=260)
    st.altair_chart(chart, use_container_width=True)
st.markdown("---")

# 3) MP4
st.subheader("3) Weekly Timelapse (downloadable MP4)")
make_mp4 = st.button("🎬 Generate MP4")
if make_mp4:
    with st.spinner("Rendering timelapse video…"):
        path = render_mp4(interp, mid_lat, mid_lon, step=10, w=800, h=600, fps=10)
    with open(path, "rb") as f:
        st.download_button("⬇️ Download bonus_week.mp4", data=f, file_name="bonus_week.mp4", mime="video/mp4")

with st.expander("Raw data (first 500 rows)"):
    st.dataframe(df.head(500), use_container_width=True)








