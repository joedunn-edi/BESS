"""
sources_weather.py — weather fetchers for the ERCOT weather-as-a-feature
exploration (see DECISIONS.md).

Responsible for:
    * fetch_weather_open_meteo(): hourly historical weather (temperature,
      wind speed, cloud cover, solar radiation) from Open-Meteo's free,
      no-API-key archive endpoint, for one location and date range.

Deliberately NOT responsible for:
    * modelling real-world reporting latency — this fetches archival data
      confirmed live against a real call (2026-09-16), suitable for a
      PERFECT-FORESIGHT ceiling test only (does weather carry any signal
      at all, if known with certainty?). It is NOT yet safe to use as a
      live trading feature: a real deployment would need to model the
      actual latency between "conditions at time t" and "when that
      observation is actually available," the same "only use what's
      truly known" discipline features.py's lag guard already applies to
      price data — deferred until the ceiling test says it's worth it.
    * choosing which location(s) are representative of ERCOT's real solar/
      wind fleet — MIDLAND_TX below is a reasonable single-point proxy for
      West Texas (where most of ERCOT's wind and solar capacity sits),
      not a properly weighted one. EIA Form 860 publishes the exact
      lat/long of every US utility-scale generator, which would let this
      be done properly if the ceiling test shows the effort is worth it.

Output contract (the "canonical weather frame" other providers should
also produce, so features.py's merge logic doesn't need to know which
provider a given column came from): one row per hourly timestamp, with
`timestamp_utc` (tz-aware UTC) plus one float column per weather
variable, sorted chronologically, no gaps assumed or validated (unlike
schema.py's price contract, gaps here are just left as NaN — a much
smaller, less formal contract for what is currently exploratory work).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# a single proxy point for West Texas, where most of ERCOT's wind and
# solar capacity sits — not weighted by real generator locations, see
# module docstring
MIDLAND_TX = (31.9973, -102.0779)

HOURLY_VARIABLES = ["temperature_2m", "wind_speed_10m", "cloud_cover", "shortwave_radiation"]

_TIMEOUT_S = 30


def fetch_weather_open_meteo(
    start_date: date,
    end_date: date,
    latitude: float = MIDLAND_TX[0],
    longitude: float = MIDLAND_TX[1],
    session: requests.Session = requests,
) -> pd.DataFrame:
    """
    Fetch hourly historical weather for [start_date, end_date] (inclusive)
    at (latitude, longitude) from Open-Meteo's archive API — free, no key
    needed for non-commercial use. Confirmed live (2026-09-16): values are
    plausible (a real August West Texas day: 26-40°C, solar radiation
    tracking daylight hours correctly) and `timezone=UTC` returns
    timestamps already in UTC, avoiding any local-DST ambiguity.

    Returns the canonical weather frame: timestamp_utc (tz-aware UTC),
    temperature_c, wind_speed_kmh, cloud_cover_pct, shortwave_radiation_wm2.
    """
    response = session.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC",
        },
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]

    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(hourly["time"], utc=True),
            "temperature_c": pd.array(hourly["temperature_2m"], dtype="float64"),
            "wind_speed_kmh": pd.array(hourly["wind_speed_10m"], dtype="float64"),
            "cloud_cover_pct": pd.array(hourly["cloud_cover"], dtype="float64"),
            "shortwave_radiation_wm2": pd.array(hourly["shortwave_radiation"], dtype="float64"),
        }
    )
