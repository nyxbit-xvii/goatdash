import os
import io
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
import plotly.express as px
from utils import haversine_km

load_dotenv()

st.set_page_config(page_title="GoatDash — Tractive + Weather", layout="wide")

st.title("🐐 GoatDash — Tractive + Weather (MVP)")
st.caption("Live tracker + weather context for herd movement.")

with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Choose input:", ["Demo CSV (synthetic)", "Upload CSV export", "Tractive Login"], index=0)
    start = st.date_input("Start date", value=dt.date.today() - dt.timedelta(days=1))
    end = st.date_input("End date", value=dt.date.today())
    if start > end:
        st.error("Start date must be before end date.")

    device_id = st.text_input("Device ID (optional)", os.getenv("TRACTIVE_DEVICE_ID", ""))

    st.divider()
    st.header("Map & Chart Options")
    show_heat = st.checkbox("Show heatmap", value=True)
    show_path = st.checkbox("Show path line", value=True)
    cluster_visits = st.checkbox("Compute 'most visited zone' (DBSCAN)", value=False)
    st.caption("Tip: DBSCAN is slower; disable for large datasets.")

@st.cache_data(show_spinner=False)
def load_demo():
    return pd.read_csv("sample_data.csv", parse_dates=["timestamp"])

def compute_kpis(df: pd.DataFrame):
    if df.empty:
        return 0.0, 0.0, 0.0, None
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Distance traveled (km)
    dists = []
    for i in range(1, len(df)):
        d = haversine_km(df.loc[i-1, "lat"], df.loc[i-1, "lon"], df.loc[i, "lat"], df.loc[i, "lon"])
        dists.append(d)
    total_km = float(np.nansum(dists))

    # Time moving (assume speed > 0.3 m/s counts as moving)
    moving = df["speed"].fillna(0) > 0.3
    if "timestamp" in df.columns:
        # Approximate per-row interval
        if len(df) >= 2:
            dt_s = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
        else:
            dt_s = 0
        time_moving_hours = float(moving.sum() * (dt_s / 3600))
    else:
        time_moving_hours = 0.0

    avg_speed = float(df["speed"].mean()) if "speed" in df.columns else 0.0

    # Centroid for initial map view
    cent = (df["lat"].mean(), df["lon"].mean()) if not df[["lat","lon"]].isna().all().all() else None
    return total_km, time_moving_hours, avg_speed, cent

def compute_most_visited_zone(df: pd.DataFrame):
    try:
        from sklearn.cluster import DBSCAN
        pts = df[["lat","lon"]].dropna().to_numpy()
        if len(pts) < 5:
            return None, None
        # eps in degrees (~50m ≈ 0.00045 deg), adjust for your site
        db = DBSCAN(eps=0.00045, min_samples=10).fit(pts)
        dfc = df.copy()
        dfc["cluster"] = db.labels_
        valid = dfc[dfc["cluster"] != -1]
        if valid.empty:
            return None, None
        top = valid["cluster"].value_counts().idxmax()
        zone = valid[valid["cluster"] == top]
        latc = zone["lat"].mean()
        lonc = zone["lon"].mean()
        return (latc, lonc), zone.shape[0]
    except Exception:
        return None, None

def merge_weather(df: pd.DataFrame):
    """Call Open-Meteo for the bounding box center/time window and join hourly weather."""
    if df.empty:
        return df, None
    import weather_client as wc
    mid_lat, mid_lon = df["lat"].mean(), df["lon"].mean()
    start_ts = df["timestamp"].min().to_pydatetime()
    end_ts = df["timestamp"].max().to_pydatetime()
    wdf = wc.fetch_open_meteo(mid_lat, mid_lon, start_ts, end_ts)
    if wdf is None or wdf.empty:
        return df, None
    # Align on hour
    df["ts_hour"] = df["timestamp"].dt.floor("H")
    wdf.rename(columns={"time":"ts_hour"}, inplace=True)
    merged = df.merge(wdf, on="ts_hour", how="left")
    return merged, wdf

# Load data based on mode
df = pd.DataFrame()
error = None
if mode == "Demo CSV (synthetic)":
    df = load_demo()
elif mode == "Upload CSV export":
    upl = st.file_uploader("Upload a CSV with columns: timestamp, lat, lon, speed, battery", type=["csv"])
    if upl:
        df = pd.read_csv(upl, parse_dates=["timestamp"])
elif mode == "Tractive Login":
    from tractive_client import TractiveClient
    user = st.text_input("Email", os.getenv("TRACTIVE_USER",""))
    pw = st.text_input("Password", os.getenv("TRACTIVE_PASS",""), type="password")
    if st.button("Log in"):
        tc = TractiveClient(user, pw)
        ok, msg = tc.login()
        st.info(msg)
        if ok and device_id and start <= end:
            try:
                df = tc.get_positions(device_id, dt.datetime.combine(start, dt.time.min), dt.datetime.combine(end, dt.time.max))
            except Exception as e:
                error = str(e)
        elif ok and not device_id:
            st.warning("Enter a Device ID to pull positions.")

if error:
    st.error(error)

if not df.empty:
    # Filter to requested window (for demo/upload modes)
    mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
    df = df.loc[mask].copy()
    df.sort_values("timestamp", inplace=True)

k1, k2, k3, center = compute_kpis(df)

col1, col2, col3 = st.columns(3)
col1.metric("Distance traveled", f"{k1:.2f} km")
col2.metric("Time moving (est.)", f"{k2:.2f} h")
col3.metric("Avg speed", f"{k3:.2f} m/s")

if df.empty:
    st.info("No data loaded yet. Use the sidebar to load demo data, upload a CSV, or log in to Tractive.")
    st.stop()

# Weather merge
with st.spinner("Fetching weather from Open-Meteo…"):
    merged, wdf = merge_weather(df)

# Map
st.subheader("Map")
if center is None:
    center = (float(df['lat'].iloc[0]), float(df['lon'].iloc[0]))
m = folium.Map(location=center, zoom_start=16, control_scale=True)
# Heat
if show_heat:
    HeatMap(df[["lat","lon"]].dropna().values.tolist(), radius=12, blur=18).add_to(m)
# Path
if show_path:
    coords = df[["lat","lon"]].dropna().values.tolist()
    folium.PolyLine(coords, weight=3).add_to(m)
# Most visited
if cluster_visits:
    zone_center, count = compute_most_visited_zone(df)
    if zone_center:
        folium.CircleMarker(location=zone_center, radius=10, fill=True, tooltip=f"Most visited zone (n={count})").add_to(m)

st_folium(m, width=None, height=500)

# Time-series charts
st.subheader("Time Series")
c1, c2 = st.columns(2)
fig1 = px.line(df, x="timestamp", y="speed", title="Speed (m/s)")
c1.plotly_chart(fig1, use_container_width=True)

if wdf is not None and not wdf.empty:
    fig2 = px.line(wdf, x="time", y=["temperature_2m","wind_speed_10m","precipitation"], title="Weather (hourly)")
    c2.plotly_chart(fig2, use_container_width=True)
else:
    c2.info("No weather data available for the time window.")

# Data table
with st.expander("Raw merged data"):
    st.dataframe(merged if isinstance(merged, pd.DataFrame) else df, use_container_width=True)
