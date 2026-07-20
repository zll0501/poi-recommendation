from __future__ import annotations

import pandas as pd
import pytest

from src.match_weather import match_events_to_weather, weather_group


def _weather() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "weather_time_utc": ["2012-04-03T18:00:00Z", "2012-04-03T19:00:00Z"],
            "temperature_2m": [10.0, 9.5],
            "relative_humidity_2m": [50, 55],
            "precipitation": [0.0, 0.4],
            "weather_code": [0, 61],
            "wind_speed_10m": [8.0, 10.0],
        }
    )


def test_match_events_floors_to_utc_hour() -> None:
    events = pd.DataFrame(
        {
            "event_id": [1, 2],
            "utc_time": ["2012-04-03T18:00:09Z", "2012-04-03T19:59:59Z"],
        }
    )
    result = match_events_to_weather(events, _weather(), partition="train")
    assert result["temperature_2m"].tolist() == [10.0, 9.5]
    assert result["weather_group"].tolist() == ["clear", "rain"]
    assert result["has_precipitation"].tolist() == [False, True]
    assert not result["weather_missing"].any()


def test_missing_hour_is_explicitly_flagged() -> None:
    events = pd.DataFrame(
        {"event_id": [3], "utc_time": ["2012-04-03T20:10:00Z"]}
    )
    result = match_events_to_weather(events, _weather(), partition="test")
    assert result.loc[0, "weather_missing"]
    assert result.loc[0, "weather_group"] == "unknown"


def test_duplicate_weather_hour_is_rejected() -> None:
    weather = pd.concat([_weather(), _weather().iloc[[0]]], ignore_index=True)
    events = pd.DataFrame(
        {"event_id": [1], "utc_time": ["2012-04-03T18:01:00Z"]}
    )
    with pytest.raises(ValueError, match="duplicate hourly timestamps"):
        match_events_to_weather(events, weather, partition="train")


@pytest.mark.parametrize(
    ("code", "expected"),
    [(0, "clear"), (3, "cloudy"), (45, "fog"), (53, "drizzle"),
     (82, "rain"), (75, "snow"), (95, "thunderstorm"), (None, "unknown")],
)
def test_weather_group(code: int | None, expected: str) -> None:
    assert weather_group(code) == expected

