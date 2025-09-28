import os
import pandas as pd
import datetime as dt

class TractiveClient:
    """Thin wrapper for Tractive via FAXES/tractive library (if installed).
    Falls back to CSV upload if not available.
    """
    def __init__(self, username: str | None = None, password: str | None = None):
        self.username = username or os.getenv("TRACTIVE_USER")
        self.password = password or os.getenv("TRACTIVE_PASS")
        self._ok = False
        self._client = None
        try:
            # Attempt to import lazily
            import tractive  # type: ignore
            self.tractive_mod = tractive
        except Exception:
            self.tractive_mod = None

    def login(self):
        if self.tractive_mod is None:
            return False, "Tractive library not installed. Install with: pip install git+https://github.com/FAXES/tractive.git"
        try:
            # The FAXES/tractive library exposes a Client class.
            # Exact API may vary; this is a best-effort example.
            self._client = self.tractive_mod.Client(self.username, self.password)
            self._ok = True
            return True, "Logged in to Tractive."
        except Exception as e:
            return False, f"Failed to log in: {e}"

    def list_devices(self):
        if not self._ok or self._client is None:
            return []
        try:
            return self._client.devices
        except Exception:
            return []

    def get_positions(self, device_id: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """Return a DataFrame with columns: timestamp (UTC), lat, lon, speed, battery."""
        if not self._ok or self._client is None:
            raise RuntimeError("Not logged in to Tractive or library missing.")
        try:
            # NOTE: The API details can differ; update these calls if needed.
            points = self._client.get_positions(device_id, start, end)
            df = pd.DataFrame(points)
            # Normalize expected columns
            if "time" in df.columns:
                df.rename(columns={"time": "timestamp"}, inplace=True)
            if "latitude" in df.columns:
                df.rename(columns={"latitude": "lat"}, inplace=True)
            if "longitude" in df.columns:
                df.rename(columns={"longitude": "lon"}, inplace=True)
            # Enforce types
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            for col in ["lat", "lon", "speed", "battery"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df[["timestamp", "lat", "lon", "speed", "battery"]].dropna(subset=["lat", "lon"])
        except Exception as e:
            raise RuntimeError(f"Error fetching positions: {e}")
