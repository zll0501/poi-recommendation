from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.weather_features import (
    WeatherFeatureStore,
    fit_weather_statistics,
    transform_weather_features,
)


def _preprocessing() -> dict:
    return {
        "categorical_column": "weather_group",
        "unknown_weather_group_id": 0,
        "numeric_features": [
            {"input": "temperature_2m", "output": "temperature_z", "transform": "identity"},
            {"input": "precipitation", "output": "precipitation_log_z", "transform": "log1p"},
        ],
        "binary_features": ["has_precipitation"],
    }


def _train() -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": [1, 2],
        "weather_group": ["clear", "cloudy"],
        "temperature_2m": [0.0, 2.0],
        "precipitation": [0.0, 0.0],
        "has_precipitation": [False, False],
    })


def test_statistics_are_fitted_only_on_train() -> None:
    statistics = fit_weather_statistics(_train(), _preprocessing())
    validation = pd.DataFrame({
        "event_id": [3],
        "weather_group": ["rain"],
        "temperature_2m": [100.0],
        "precipitation": [1.0],
        "has_precipitation": [True],
    })
    transformed = transform_weather_features(validation, statistics, partition="validation")
    assert statistics["numeric_statistics"]["temperature_2m"]["mean"] == 1.0
    assert statistics["numeric_statistics"]["temperature_2m"]["std"] == 1.0
    assert transformed.loc[0, "temperature_z"] == 99.0
    assert transformed.loc[0, "weather_group_idx"] == 0
    assert transformed.loc[0, "has_precipitation"] == 1


def test_feature_store_preserves_requested_event_order() -> None:
    statistics = fit_weather_statistics(_train(), _preprocessing())
    features = transform_weather_features(_train(), statistics, partition="train")
    store = WeatherFeatureStore(features, statistics)
    batch = store.get_by_event_ids([2, 1])
    assert batch.group_idx.dtype == torch.long
    assert batch.group_idx.tolist() == [2, 1]
    assert batch.numeric.shape == (2, 3)


def test_feature_store_rejects_unknown_event() -> None:
    statistics = fit_weather_statistics(_train(), _preprocessing())
    features = transform_weather_features(_train(), statistics, partition="train")
    with pytest.raises(KeyError, match="missing for 1"):
        WeatherFeatureStore(features, statistics).get_by_event_ids([999])
