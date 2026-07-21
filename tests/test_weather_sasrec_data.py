import pandas as pd
import pytest
import torch

from src.weather_features import WeatherFeatureStore
from src.weather_sasrec_data import WeatherSASRecCollator, build_poi_category_index


def _store() -> WeatherFeatureStore:
    frame = pd.DataFrame(
        {
            "event_id": [20, 10],
            "weather_group_idx": [2, 1],
            "temperature_z": [0.2, -0.1],
            "has_precipitation": [1, 0],
        }
    )
    statistics = {
        "model_numeric_columns": ["temperature_z", "has_precipitation"],
        "weather_group_vocab_size": 3,
    }
    return WeatherFeatureStore(frame, statistics)


def test_weather_collator_preserves_event_order() -> None:
    def base_collator(_):
        return {
            "inputs": {"poi_sequence": torch.tensor([[2], [3]])},
            "target": torch.tensor([3, 4]),
            "event_id": torch.tensor([10, 20]),
        }

    batch = WeatherSASRecCollator(base_collator, _store())([object(), object()])
    assert batch["inputs"]["weather_group_idx"].tolist() == [1, 2]
    torch.testing.assert_close(
        batch["inputs"]["weather_numeric"],
        torch.tensor([[-0.1, 0.0], [0.2, 1.0]]),
    )


def test_poi_category_index_covers_special_tokens_and_metadata() -> None:
    metadata = pd.DataFrame(
        {"poi_idx": [2, 4, 3], "category_idx": [5, 6, 5]}
    )
    lookup = build_poi_category_index(metadata, num_pois=6)
    assert lookup.tolist() == [0, 1, 5, 5, 6, 1]


def test_poi_category_index_rejects_duplicate_pois() -> None:
    metadata = pd.DataFrame(
        {"poi_idx": [2, 2], "category_idx": [5, 6]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_poi_category_index(metadata, num_pois=5)
