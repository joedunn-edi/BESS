"""
test_sources_weather.py — tests for bess/sources_weather.py.

Fixture shaped to match a real Open-Meteo archive response, confirmed
live on 2026-09-16 (see sources_weather.py's module docstring).
"""

from datetime import date

import pandas as pd
import pytest

from bess.sources_weather import fetch_weather_open_meteo


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.captured_params = None

    def get(self, url, params=None, timeout=None):
        self.captured_params = params
        return _FakeResponse(self._payload)


def _real_shaped_payload(n_hours: int = 4) -> dict:
    return {
        "hourly": {
            "time": [f"2026-08-24T{h:02d}:00" for h in range(n_hours)],
            "temperature_2m": [38.2, 36.5, 34.6, 32.1][:n_hours],
            "wind_speed_10m": [11.4, 14.0, 17.9, 19.0][:n_hours],
            "cloud_cover": [8, 12, 16, 20][:n_hours],
            "shortwave_radiation": [287.0, 0.0, 0.0, 0.0][:n_hours],
        }
    }


def test_fetch_weather_open_meteo_parses_a_real_shaped_response():
    session = _FakeSession(_real_shaped_payload())
    df = fetch_weather_open_meteo(date(2026, 8, 24), date(2026, 8, 24), session=session)

    assert list(df.columns) == ["timestamp_utc", "temperature_c", "wind_speed_kmh", "cloud_cover_pct", "shortwave_radiation_wm2"]
    assert len(df) == 4
    assert df["timestamp_utc"].dt.tz is not None  # tz-aware, not naive
    assert df["temperature_c"].iloc[0] == pytest.approx(38.2)
    assert df["shortwave_radiation_wm2"].iloc[1] == pytest.approx(0.0)


def test_fetch_weather_open_meteo_sends_utc_timezone_and_the_requested_location():
    session = _FakeSession(_real_shaped_payload())
    fetch_weather_open_meteo(date(2026, 8, 24), date(2026, 8, 25), latitude=31.9973, longitude=-102.0779, session=session)

    assert session.captured_params["timezone"] == "UTC"
    assert session.captured_params["latitude"] == 31.9973
    assert session.captured_params["longitude"] == -102.0779
    assert session.captured_params["start_date"] == "2026-08-24"
    assert session.captured_params["end_date"] == "2026-08-25"


def test_fetch_weather_open_meteo_defaults_to_midland_tx():
    session = _FakeSession(_real_shaped_payload())
    fetch_weather_open_meteo(date(2026, 8, 24), date(2026, 8, 24), session=session)

    assert session.captured_params["latitude"] == pytest.approx(31.9973)
    assert session.captured_params["longitude"] == pytest.approx(-102.0779)


def test_fetch_weather_open_meteo_timestamps_are_chronological_and_hourly():
    session = _FakeSession(_real_shaped_payload(n_hours=4))
    df = fetch_weather_open_meteo(date(2026, 8, 24), date(2026, 8, 24), session=session)

    diffs = df["timestamp_utc"].diff().dropna()
    assert (diffs.dt.total_seconds() == 3600).all()
