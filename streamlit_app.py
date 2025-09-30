import datetime as dt
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
from xml.etree import ElementTree as ET
import re

# ----------------- CONFIG -----------------
GPX_FILE = "assets/Bonus_Latest.gpx"
DASHBOARD_TITLE = "HerdTracker at Steep Mountain Farm — Bonus’s Weekly Recap"
ICON_URL = "https://raw.githubusercontent.com/nyxbit-xvii/goatdash/refs/heads/main/assets/bonus_icon.png"

# Playback speed: 0.5 seconds per hour of real time
HOURS_PER_SECOND = 2

# ----------------- HELPERS -----------------
def parse_gpx(path: str) -> pd.DataFrame:
    """Parse Tractive GPX export into DataFrame with timestamp, lat, lon, optional speed."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    root = ET.fromstring(text)
    ns = {"g": "http://www.topografix.com/GPX/1/1"}

    pts = root.findall(".//g:trkpt", ns)
    rows = []
    for p in pts:
        lat = p.attrib.get("lat")
        lon = p.attrib.get("lon")
        time_el = p.find("g:time", ns)
        cmt_el = p.find("g:cmt", ns)
        rows.append({
            "timestamp": time_el.text if time_el is not None else None,
            "lat": lat,
            "lon": lon,
            "cmt": cmt_el.text if cmt_el is not None else None
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No <trkpt> points found in GPX.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    if "cmt" in df.columns:
        sp = df["cmt"].str.extract(r"speed:\s*([\d\.]+)")
        df["speed"] = pd.to_numeric(sp[0], errors="coerce")

    return df.dropna(subset=["timestamp", "lat", "lon"]).sort_values("timestamp").reset_index(drop=True)

def fetch_open_meteo(lat: float, lon: float, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,precipitation",
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("hourly", {})
    if not data: 
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df

def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    from math import radians, sin, cos, asin, sqrt
    R = 3958.8  # miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

def total_distance_miles(df: pd.DataFrame) -> float:
    dist = 0.0
    for i in range(1, len(df)):
        dist += haversine_miles(df.iloc[i-1].lat, df.iloc[i-1].lon, df.iloc[i].lat, df.iloc[i].lon)
    return dist

# ----------------- STREAMLIT -----------------
st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
st.title(DASHBOARD_TITLE)

# Load GPX
df = parse_gpx(GPX_FILE)
if df.empty:
    st.error("No data in GPX file.")
    st.stop()

# KPIs
dist_miles = total_distance_miles(df)
duration = df["timestamp"].max() - df["timestamp"].min()
hours = duration.total_seconds()/3600 if pd.notna(duration) else 0.0

c1, c2 = st.columns(2)
c1.metric("Distance traveled", f"{dist_miles:.2f} miles")
c2.metric("Time span", f"{hours:.1f} hours")

# Weather
mid_lat, mid_lon = df["lat"].mean(), df["lon"].mean()
wdf = fetch_open_meteo(mid_lat, mid_lon, df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime())

# TripsLayer data
trip_data = df[["lat", "lon", "timestamp"]].copy()
trip_data["timestamp"] = (trip_data["timestamp"] - df["timestamp"].min()).dt.total_seconds()
trip_data.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)

# Heatmap data
heat_data = df.rename(columns={"lat": "latitude", "lon": "longitude"})

# Layers
layers = []

# Heatmap
layers.append(
    pdk.Layer(
        "HeatmapLayer",
        data=heat_data,
        get_position='[longitude, latitude]',
        aggregation="'MEAN'",
        get_weight=1,
        radiusPixels=40,
        colorRange=[
            [0, 255, 0, 180],   # green
            [255, 255, 0, 200], # yellow
            [255, 0, 0, 200],   # red
        ],
    )
)

# TripsLayer (animated path)
layers.append(
    pdk.Layer(
        "TripsLayer",
        data=trip_data,
        get_path="[['longitude','latitude']]",
        get_timestamps="timestamp",
        get_color=[0, 200, 255],
        width_min_pixels=4,
        trail_length=600,  # seconds shown behind Bonus
        current_time=trip_data["timestamp"].max(),
    )
)

# IconLayer for Bonus
bonus_df = pd.DataFrame([{
    "longitude": float(df.iloc[-1]["lon"]),
    "latitude": float(df.iloc[-1]["lat"]),
    "icon_data": {
        "url": ICON_URL,
        "width": 256,
        "height": 256,
        "anchorY": 256
    }
}])
layers.append(
    pdk.Layer(
        "IconLayer",
        data=bonus_df,
        get_icon="icon_data",
        get_size=8,
        size_scale=12,
        get_position="[longitude, latitude]",
    )
)

view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=15, pitch=45)

# Render deck
r = pdk.Deck(
    map_style="mapbox://styles/mapbox/satellite-streets-v11",
    mapbox_key=st.secrets.get("MAPBOX_API_KEY", ""),
    layers=layers,
    initial_view_state=view_state,
    tooltip={"text": "Bonus's Weekly Path"}
)
st.pydeck_chart(r)

# Weather chart
if not wdf.empty:
    st.subheader("Weather vs Movement")
    highlight_time = st.slider("Playback hour", 0, int(hours), 0, step=1)
    current_time = df["timestamp"].min() + dt.timedelta(hours=highlight_time)
    st.caption(f"Showing weather at: {current_time.strftime('%Y-%m-%d %H:%M UTC')}")
    st.line_chart(
        wdf.set_index("time")[["temperature_2m","wind_speed_10m","precipitation"]],
        height=300
    )

with st.expander("Raw data"):
    st.dataframe(df, use_container_width=True)

