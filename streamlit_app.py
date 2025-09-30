import datetime as dt
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
from xml.etree import ElementTree as ET

# ----------------- CONFIG -----------------
GPX_FILE = "assets/Bonus_Latest.gpx"
DASHBOARD_TITLE = "HerdTracker at Steep Mountain Farm — Bonus’s Weekly Recap"
ICON_URL = "https://raw.githubusercontent.com/nyxbit-xvii/goatdash/refs/heads/main/assets/bonus_icon.png"

# Mapbox token from Streamlit secrets
pdk.settings.mapbox_api_key = st.secrets["MAPBOX_API_KEY"]

# ----------------- HELPERS -----------------
def parse_gpx(path: str) -> pd.DataFrame:
    """Parse Tractive GPX export into DataFrame with timestamp, lat, lon, optional speed."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    root = ET.fromstring(text)
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

    df = df.dropna(subset=["timestamp", "lat", "lon"]).sort_values("timestamp").reset_index(drop=True)
    return df

@st.cache_data(ttl=3600)
def fetch_open_meteo(lat: float, lon: float, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo API, cached to avoid refetching during autoplay."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "hourly": "temperature_2m,wind_speed_10m,precipitation",
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "timezone": "UTC",
        }
        r = requests.get(url, params=params, timeout=45)  # longer timeout
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
    return float(dist)

def nearest_point_at(df: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """Return the row closest in time to t (UTC)."""
    i = np.argmin(np.abs(df["timestamp"].values - np.array(t, dtype="datetime64[ns]")))
    return df.iloc[i]

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
st.title(DASHBOARD_TITLE)

# Load weekly GPX
df = parse_gpx(GPX_FILE)
if df.empty:
    st.error("No data in GPX file.")
    st.stop()

mid_lat, mid_lon = float(df["lat"].mean()), float(df["lon"].mean())

# KPIs
dist_miles = total_distance_miles(df)
duration = df["timestamp"].max() - df["timestamp"].min()
hours_span = int((duration.total_seconds() // 3600) if pd.notna(duration) else 0)

c1, c2 = st.columns(2)
c1.metric("Distance traveled", f"{dist_miles:.2f} miles")
c2.metric("Time span", f"{hours_span} hours")

# Weather (cached)
wdf = fetch_open_meteo(mid_lat, mid_lon, df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime())

# ---- Time controls ----
st.subheader("Weekly Recap")

start_ts = df["timestamp"].min()
end_ts = df["timestamp"].max()
total_hours = max(1, int((end_ts - start_ts).total_seconds() // 3600))

# Playback speed selector
speed = st.selectbox("Playback speed", ["Slow", "Normal", "Fast"], index=1)
speed_map = {"Slow": 0.5, "Normal": 0.2, "Fast": 0.05}

col1, col2 = st.columns([4, 1])
with col1:
    sel_hour = st.slider("Playback hour", 0, total_hours, 0, step=1)
with col2:
    play = st.button("▶️ Play")

if play:
    for h in range(sel_hour, total_hours + 1):
        current_time = start_ts + pd.Timedelta(hours=h)
        st.caption(f"Time: **{current_time.strftime('%Y-%m-%d %H:%M UTC')}**")

        # Current position of Bonus
        cur = nearest_point_at(df, current_time)

        # Heatmap + icon
        heat_data = df.rename(columns={"lat": "latitude", "lon": "longitude"})
        bonus_df = pd.DataFrame([{
            "longitude": float(cur["lon"]),
            "latitude": float(cur["lat"]),
            "icon_data": {
                "url": ICON_URL,
                "width": 256,
                "height": 256,
                "anchorY": 256
            }
        }])

        layers = [
            pdk.Layer(
                "HeatmapLayer",
                data=heat_data,
                get_position='[longitude, latitude]',
                get_weight=1,
                radius_pixels=40,
                aggregation="MEAN",
                color_range=[
                    [0, 255, 0, 160],
                    [255, 255, 0, 180],
                    [255, 128, 0, 200],
                    [255, 0, 0, 220],
                ],
            ),
            pdk.Layer(
                "IconLayer",
                data=bonus_df,
                get_icon="icon_data",
                get_size=10,
                size_scale=14,
                get_position="[longitude, latitude]",
            ),
        ]

        view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=15, pitch=45)
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-streets-v11",
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "Bonus’s weekly movement"}
        )
        st.pydeck_chart(deck)

        # Weather overlay
        if not wdf.empty:
            idx = np.argmin(np.abs(wdf["time"].values - np.array(current_time, dtype="datetime64[ns]")))
            cur_wx = wdf.iloc[idx]
            st.write(
                f"**{cur_wx['time'].strftime('%Y-%m-%d %H:%M UTC')}**  "
                f"• Temp: **{cur_wx['temperature_2m']:.1f}°C**  "
                f"• Wind: **{cur_wx['wind_speed_10m']:.1f} m/s**  "
                f"• Precip: **{cur_wx['precipitation']:.2f} mm**"
            )

        time.sleep(speed_map[speed])

# If not playing, still show the selected hour
if not play:
    current_time = start_ts + pd.Timedelta(hours=sel_hour)
    st.caption(f"Time: **{current_time.strftime('%Y-%m-%d %H:%M UTC')}**")

    cur = nearest_point_at(df, current_time)

    heat_data = df.rename(columns={"lat": "latitude", "lon": "longitude"})
    bonus_df = pd.DataFrame([{
        "longitude": float(cur["lon"]),
        "latitude": float(cur["lat"]),
        "icon_data": {
            "url": ICON_URL,
            "width": 256,
            "height": 256,
            "anchorY": 256
        }
    }])

    layers = [
        pdk.Layer(
            "HeatmapLayer",
            data=heat_data,
            get_position='[longitude, latitude]',
            get_weight=1,
            radius_pixels=40,
            aggregation="MEAN",
            color_range=[
                [0, 255, 0, 160],
                [255, 255, 0, 180],
                [255, 128, 0, 200],
                [255, 0, 0, 220],
            ],
        ),
        pdk.Layer(
            "IconLayer",
            data=bonus_df,
            get_icon="icon_data",
            get_size=10,
            size_scale=14,
            get_position="[longitude, latitude]",
        ),
    ]

    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=15, pitch=45)
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v11",
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "Bonus’s weekly movement"}
    )
    st.pydeck_chart(deck)

# Weather aligned panel
st.subheader("Weather aligned to playback hour")
if not wdf.empty:
    idx = np.argmin(np.abs(wdf["time"].values - np.array(current_time, dtype="datetime64[ns]")))
    cur_wx = wdf.iloc[idx]
    st.write(
        f"**{cur_wx['time'].strftime('%Y-%m-%d %H:%M UTC')}**  "
        f"• Temp: **{cur_wx['temperature_2m']:.1f}°C**  "
        f"• Wind: **{cur_wx['wind_speed_10m']:.1f} m/s**  "
        f"• Precip: **{cur_wx['precipitation']:.2f} mm**"
    )
    st.line_chart(
        wdf.set_index("time")[["temperature_2m", "wind_speed_10m", "precipitation"]],
        height=260
    )
else:
    st.info("Weather service temporarily unavailable — map and playback still work.")

with st.expander("Raw data (first 500 rows)"):
    st.dataframe(df.head(500), use_container_width=True)




