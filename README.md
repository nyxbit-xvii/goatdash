# 🐐 GoatDash — Tractive + Weather (MVP)

A Streamlit app to visualize herd movement from **Tractive** and overlay **weather** (Open‑Meteo). Includes:
- Path/heatmap map
- Distance/time moving/average speed KPIs
- Hourly weather charts (temperature, wind, precipitation)
- Optional clustering to estimate “most visited zone”

## Quick Start

```bash
# 1) Create and activate a virtualenv (recommended)
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Set your Tractive credentials
cp .env.example .env
# then edit .env and fill in TRACTIVE_USER / TRACTIVE_PASS

# 4) Run
streamlit run streamlit_app.py
```

The app ships with a **synthetic demo CSV** so you can click around immediately.

## Using Tractive

This project is set up to work with the community `FAXES/tractive` Python client. If you want live pulls:

```bash
pip install git+https://github.com/FAXES/tractive.git
```

Then choose **“Tractive Login”** in the sidebar, enter your credentials and device ID, and select a time window.

> If the library API differs, open `tractive_client.py` and adjust the `get_positions` method.

## Weather: Open‑Meteo (free, no key)

We call the [Open‑Meteo](https://open-meteo.com/) forecast API and join hourly weather to your tracks.
If you prefer NWS (US only) or Tomorrow.io, you can swap out `weather_client.py` accordingly.

## CSV Format

If you export CSVs, we expect columns:

- `timestamp` (ISO8601)
- `lat`, `lon` (decimal degrees)
- `speed` (m/s)
- `battery` (%)

See `sample_data.csv` for an example.

## Roadmap Ideas

- Live WebSocket updates from Tractive to get near‑real‑time tracks
- Barn/pasture geofences with in/out counters
- Multi‑goat layering and comparison
- Pasture usage heat map by day/week
- Downloadable daily summaries
- Mobile PWA packaging (Streamlit Cloud + Android/iOS wrapper), or migrate UI to React/Expo and keep a Python FastAPI backend

## License

MIT — use freely and have fun building **Antflix 🐜** and **GoatDash 🐐**!
