import pandas as pd
import pytest

from src.weather_data import attach_weather


def test_attach_weather_preserves_event_order() -> None:
    events = pd.DataFrame({"event_id": [2, 1], "poi_idx": [20, 10]})
    sidecar = pd.DataFrame({
        "event_id": [1, 2],
        "weather_group": ["rain", "clear"],
    })
    result = attach_weather(events, sidecar)
    assert result["event_id"].tolist() == [2, 1]
    assert result["weather_group"].tolist() == ["clear", "rain"]


def test_attach_weather_rejects_missing_event() -> None:
    events = pd.DataFrame({"event_id": [1, 2]})
    sidecar = pd.DataFrame({"event_id": [1], "weather_group": ["rain"]})
    with pytest.raises(ValueError, match="no weather sidecar"):
        attach_weather(events, sidecar)
