import requests
import pandas as pd
import datetime as dt

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_open_meteo(lat: float, lon: float, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Fetch hourly weather (temp, wind, precip) from Open-Meteo for a time window.
    Returns DataFrame with columns: time (UTC), temperature_2m, wind_speed_10m, precipitation.
    """
    # Open-Meteo accepts start/end in ISO8601, and can return hourly series.
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,precipitation",
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    hourly = data.get("hourly", {})
    if not hourly:
        return pd.DataFrame(columns=["time","temperature_2m","wind_speed_10m","precipitation"])

    df = pd.DataFrame(hourly)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["temperature_2m","wind_speed_10m","precipitation"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
