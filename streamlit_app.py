import io
import datetime as dt
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

# ---------- Helpers ----------
def parse_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols = {c.lower(): c for c in df.columns}
    ts_col = next((cols[k] for k in cols if k in ("timestamp","time","datetime","date_time","recorded_at")), None)
    lat_col = next((cols[k] for k in cols if k in ("lat","latitude")), None)
    lon_col = next((cols[k] for k in cols if k in ("lon","lng","long","longitude")), None)
    spd_col = next((cols[k] for k in cols if k in ("speed","spd")), None)
    bat_col = next((cols[k] for k in cols if k in ("battery","battery_level","batt")), None)

    if not (ts_col and lat_col and lon_col):
        raise ValueError("CSV must have timestamp, lat, lon columns (speed/battery optional).")

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
    })
    if spd_col: out["speed"] = pd.to_numeric(df[spd_col], errors="coerce")
    if bat_col: out["battery"] = pd.to_numeric(df[bat_col], errors="coerce")
    return out.dropna(subset=["timestamp","lat","lon"]).sort_values("timestamp")

def parse_gpx(file) -> pd.DataFrame:
    """
    Parse Tractive GPX export into DataFrame with timestamp, lat, lon, (optional) speed.
    """
    from xml.etree import ElementTree as ET

    # Read file contents
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="ignore")
    else:
        text = str(data)

    root = ET.fromstring(text)

    # GPX namespace
    ns = {"g": "http://www.topografix.com/GPX/1/1"}

    # Find all track points
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

    # Convert datatypes
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Extract speed from comment if available
    if "cmt" in df.columns:
        sp = df["cmt"].str.extract(r"speed:\s*([\d\.]+)")
        df["speed"] = pd.to_numeric(sp[0], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["timestamp","lat","lon"]).sort_values("timestamp").reset_index(drop=True)

    return df

def fetch_open_meteo(lat: float, lon: float, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["time","temperature_2m","wind_speed_10m","precipitation"])
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    return R*c

def total_distance_km(df: pd.DataFrame) -> float:
    d = 0.0
    for i in range(1, len(df)):
        d += haversine_km(df.iloc[i-1].lat, df.iloc[i-1].lon, df.iloc[i].lat, df.iloc[i].lon)
    return float(d)

# ---------- UI ----------
st.set_page_config(page_title="GoatDash — Bonus the Sheep", layout="wide")
st.title("🐑 GoatDash — Bonus the Sheep")
st.caption("Upload Bonus’s location history (CSV or GPX). The app maps the path, shows KPIs, and overlays hourly weather.")

with st.sidebar:
    st.header("Upload data")
    file = st.file_uploader("Choose CSV or GPX exported from Tractive", type=["csv","gpx"])
    date_filter = st.checkbox("Filter by date range", value=False)
    start_date = st.date_input("Start", value=dt.date.today() - dt.timedelta(days=3), disabled=not date_filter)
    end_date = st.date_input("End", value=dt.date.today(), disabled=not date_filter)

if not file:
    st.info("Upload a **CSV** (timestamp, lat, lon, [speed, battery]) or a **GPX** file to begin.")
    st.stop()

# Parse
try:
    if file.name.lower().endswith(".csv"):
        df = parse_csv(file)
    else:
        df = parse_gpx(file)
except Exception as e:
    st.error(f"Couldn’t read your file: {e}")
    st.stop()

if df.empty:
    st.warning("No valid points found in the file.")
    st.stop()

# Filter by date
if date_filter:
    mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    df = df.loc[mask]
    if df.empty:
        st.warning("No points in the selected date range.")
        st.stop()

# KPIs
dist_km = total_distance_km(df)
avg_speed = float(df["speed"].mean()) if "speed" in df.columns else 0.0
duration = (df["timestamp"].max() - df["timestamp"].min())
hours = duration.total_seconds()/3600 if pd.notna(duration) else 0.0

k1, k2, k3 = st.columns(3)
k1.metric("Distance traveled", f"{dist_km:.2f} km")
k2.metric("Time span", f"{hours:.1f} h")
k3.metric("Avg speed", f"{avg_speed:.2f} m/s" if avg_speed else "—")

# Center for map
center = [float(df["lat"].median()), float(df["lon"].median())]

# Path data for map
path_data = df[["lat","lon"]].rename(columns={"lat":"latitude","lon":"longitude"})
coords = path_data[["longitude","latitude"]].values.tolist()
path = [{"path": coords}] if len(coords) >= 2 else []

layers = []

if path:
    layers.append(
        pdk.Layer(
            "PathLayer",
            data=path,
            get_path="path",
            width_scale=2,
            width_min_pixels=3,
            get_color=[30, 144, 255, 200],  # blue-ish
        )
    )

# Points layer
layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        data=path_data,
        get_position='[longitude, latitude]',
        get_radius=4,
        radius_min_pixels=2,
        radius_max_pixels=6,
        get_fill_color=[34, 139, 34, 160],  # green-ish
    )
)

# 🐑 Bonus marker with custom icon
icon_url = "https://raw.githubusercontent.com/nyxbit-xvii/goatdash/refs/heads/main/assets/bonus_icon.png"
last = df.iloc[-1]
bonus_df = pd.DataFrame([{
    "lon": float(last["lon"]),
    "lat": float(last["lat"]),
    "icon_data": {
        "url": icon_url,
        "width": 128,
        "height": 128,
        "anchorY": 128
    }
}])
layers.append(
    pdk.Layer(
        "IconLayer",
        data=bonus_df,
        get_icon="icon_data",
        get_size=4,
        size_scale=8,
        get_position="[lon, lat]",
    )
)

view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=15, pitch=0)
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None))

# Weather
with st.expander("Weather overlay (hourly)"):
    try:
        wdf = fetch_open_meteo(center[0], center[1], df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime())
        if not wdf.empty:
            st.line_chart(
                wdf.set_index("time")[["temperature_2m","wind_speed_10m","precipitation"]],
                height=240
            )
        else:
            st.info("No hourly weather returned for this time span.")
    except Exception as e:
        st.warning(f"Weather fetch issue: {e}")

with st.expander("Raw data"):
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
