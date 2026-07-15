"""Tests for train-only vocabularies and encoded model data."""

import pandas as pd
import pytest

from src.encode_data import prepare_encoded_data


ENCODING_CONFIG = {
    "fit_on": "train",
    "pad_id": 0,
    "unknown_id": 1,
    "normal_id_start": 2,
    "vocabularies": {
        "user_id": "user_idx",
        "poi_id": "poi_idx",
        "category_id": "category_idx",
    },
}


def make_partition(user: str, pois: list[str], categories: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": range(len(pois)),
            "user_id": [user] * len(pois),
            "poi_id": pois,
            "category_id": categories,
            "category_name": categories,
            "latitude": [40.7] * len(pois),
            "longitude": [-74.0] * len(pois),
            "utc_time": pd.date_range("2024-01-01", periods=len(pois), tz="UTC"),
            "hour": range(len(pois)),
            "weekday": [0] * len(pois),
            "time_slot": ["morning"] * len(pois),
        }
    )


def test_mappings_are_fitted_only_on_training_tokens() -> None:
    train = make_partition("u1", ["p2", "p1"], ["c2", "c1"])
    validation = make_partition("u1", ["future_poi"], ["future_category"])
    test = make_partition("future_user", ["p1"], ["c1"])

    partitions, mappings, metadata, report = prepare_encoded_data(
        train, validation, test, ENCODING_CONFIG
    )

    poi_mapping = mappings["vocabularies"]["poi_id"]["token_to_id"]
    assert poi_mapping == {"p1": 2, "p2": 3}
    assert "future_poi" not in poi_mapping
    assert partitions["validation"]["poi_idx"].tolist() == [1]
    assert partitions["validation"]["category_idx"].tolist() == [1]
    assert partitions["test"]["user_idx"].tolist() == [1]
    assert partitions["train"][["user_idx", "poi_idx", "category_idx"]].min().min() >= 2
    assert metadata["poi_idx"].tolist() == [2, 3]
    assert report["quality_checks"]["training_contains_unknown_ids"] is False


def test_poi_metadata_is_canonical_and_fitted_on_train_only() -> None:
    train = make_partition("u1", ["p1", "p1", "p1"], ["c1", "c1", "c2"])
    train["latitude"] = [40.0, 40.2, 99.0]
    validation = make_partition("u1", ["p1"], ["future_category"])
    validation[["latitude", "longitude"]] = [0.0, 0.0]
    test = make_partition("u1", ["p1"], ["c1"])

    partitions, _, metadata, _ = prepare_encoded_data(
        train, validation, test, ENCODING_CONFIG
    )

    assert metadata.loc[0, "category_confidence"] == pytest.approx(2 / 3)
    assert metadata.loc[0, "latitude"] == 40.2
    assert partitions["validation"].loc[0, "category_idx"] == metadata.loc[0, "category_idx"]
    assert partitions["validation"].loc[0, "latitude"] == 40.2
