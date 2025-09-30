import datetime as dt
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
from xml.etree import ElementTree as ET
import numpy as np

# ----------------- CONFIG -----------------
GPX_FILE = "assets/Bonus_Latest.gpx"
DASHBOARD_TITLE = "HerdTracker at Steep Mountain Farm — Bonus’s Weekly Recap"
ICON_URL = "https://raw.githubusercontent.com/nyxbit-xvii/goatdash/refs/heads/main/assets/bonus_icon.png"

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
    wdf = pd.DataFrame(data)
    wdf["time"] = pd.to_datetime(wdf["time"], utc=True, errors="coerce")
    return wdf

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

# ----------------- STREAMLIT -----------------
st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
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

# Weather (hourly)
wdf = fetch_open_meteo(mid_lat, mid_lon, df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime())

# ---- Time scrubber (fast recap) ----
st.subheader("Weekly Recap")
colL, colR = st.columns([3, 2])

with colL:
    # Slider in hours from start
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    total_hours = max(1, int((end_ts - start_ts).total_seconds() // 3600))
    sel_hour = st.slider("Playback hour (fast scrub)", 0, total_hours, 0, step=1)
    current_time = start_ts + pd.Timedelta(hours=sel_hour)
    st.caption(f"Time: **{current_time.strftime('%Y-%m-%d %H:%M UTC')}**")

    # Current position of Bonus at this hour
    cur = nearest_point_at(df, current_time)

    # Heatmap data (all points)
    heat_data = df.rename(columns={"lat": "latitude", "lon": "longitude"})

    # Icon for Bonus at current time
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

    layers = []

    # Heatmap (green -> yellow -> red)
    layers.append(
        pdk.Layer(
            "HeatmapLayer",
            data=heat_data,
            get_position='[longitude, latitude]',
            get_weight=1,
            radius_pixels=40,
            aggregation="MEAN",
            color_range=[
                [0, 255, 0, 160],    # green (least)
                [255, 255, 0, 180],  # yellow
                [255, 128, 0, 200],  # orange
                [255, 0, 0, 220],    # red (most)
            ],
        )
    )

    # Bonus icon (bigger)
    layers.append(
        pdk.Layer(
            "IconLayer",
            data=bonus_df,
            get_icon="icon_data",
            get_size=10,       # bigger icon
            size_scale=14,     # scale factor
            get_position="[longitude, latitude]",
        )
    )

    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=15, pitch=45)

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v11",
        mapbox_key=st.secrets.get("MAPBOX_API_KEY", ""),
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "Bonus’s weekly movement"}
    )
    st.pydeck_chart(deck)

with colR:
    st.markdown("**Weather aligned to playback hour**")
    if not wdf.empty:
        # Snap to the same hour in weather
        # Find closest weather row to current_time (by hour)
        wdf_sorted = wdf.sort_values("time")
        idx = np.argmin(np.abs(wdf_sorted["time"].values - np.array(current_time, dtype="datetime64[ns]")))
        cur_wx = wdf_sorted.iloc[idx]

        st.write(
            f"**{cur_wx['time'].strftime('%Y-%m-%d %H:%M UTC')}**  "
            f"• Temp: **{cur_wx['temperature_2m']:.1f}°C**  "
            f"• Wind: **{cur_wx['wind_speed_10m']:.1f} m/s**  "
            f"• Precip: **{cur_wx['precipitation']:.2f} mm**"
        )

        # Show whole-week weather chart (line chart)
        st.line_chart(
            wdf.set_index("time")[["temperature_2m", "wind_speed_10m", "precipitation"]],
            height=260
        )
        st.caption("Tip: drag the slider to scrub fast through the week and watch weather + position update.")
    else:
        st.info("No hourly weather returned for this time range.")

with st.expander("Raw data (first 500 rows)"):
    st.dataframe(df.head(500), use_container_width=True)


