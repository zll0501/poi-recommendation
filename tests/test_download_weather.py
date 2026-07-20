from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from src.download_weather import build_request_url, validate_hourly_payload


def _config() -> dict:
    return {
        "source": {
            "provider": "open_meteo",
            "dataset": "era5",
            "api_url": "https://example.test/archive",
        },
        "location": {"latitude": 40.7128, "longitude": -74.0060},
        "time": {
            "start_date": "2012-04-12",
            "end_date": "2012-04-12",
            "timezone": "UTC",
        },
        "hourly_features": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
        ],
    }


def _payload() -> dict:
    times = pd.date_range("2012-04-12", periods=24, freq="h").strftime(
        "%Y-%m-%dT%H:%M"
    ).tolist()
    return {
        "hourly": {
            "time": times,
            "temperature_2m": list(range(24)),
            "relative_humidity_2m": [60] * 24,
            "precipitation": [0.0] * 24,
            "weather_code": [0] * 24,
            "wind_speed_10m": [10.0] * 24,
        },
        "hourly_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "weather_code": "wmo code",
            "wind_speed_10m": "km/h",
        },
    }


def test_build_request_url_contains_reproducible_parameters() -> None:
    url = build_request_url(_config())
    assert "start_date=2012-04-12" in url
    assert "end_date=2012-04-12" in url
    assert "timezone=UTC" in url
    assert "models=era5" in url
    assert "temperature_2m%2Crelative_humidity_2m" in url


def test_validate_hourly_payload_accepts_complete_day() -> None:
    frame, report = validate_hourly_payload(_payload(), _config())
    assert len(frame) == 24
    assert isinstance(frame["weather_time_utc"].dtype, pd.DatetimeTZDtype)
    assert str(frame["weather_time_utc"].dt.tz) == "UTC"
    assert report["duplicate_timestamps"] == 0
    assert sum(report["missing_values"].values()) == 0


def test_validate_hourly_payload_rejects_missing_hour() -> None:
    payload = deepcopy(_payload())
    for values in payload["hourly"].values():
        values.pop(5)
    with pytest.raises(ValueError, match="expected 24 hourly rows"):
        validate_hourly_payload(payload, _config())


def test_validate_hourly_payload_rejects_duplicate_time() -> None:
    payload = deepcopy(_payload())
    payload["hourly"]["time"][1] = payload["hourly"]["time"][0]
    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_hourly_payload(payload, _config())
