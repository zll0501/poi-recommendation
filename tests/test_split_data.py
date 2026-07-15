"""Tests for the shared global-time evaluation protocol."""

import pandas as pd
import pytest

from src.split_data import split_checkins


SPLIT_CONFIG = {
    "method": "global_chronological",
    "train_ratio": 0.6,
    "validation_ratio": 0.2,
    "test_ratio": 0.2,
    "timestamp_column": "utc_time",
    "query_time_known": True,
    "candidate_scope": "train_pois",
    "random_shuffle": False,
}


def make_checkins() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": ["u1", "u2"] * 5,
        "poi_id": [f"p{i}" for i in range(10)],
        "utc_time": pd.date_range("2024-01-01", periods=10, tz="UTC"),
    }).sample(frac=1, random_state=7).reset_index(drop=True)


def test_split_is_globally_chronological_complete_and_disjoint() -> None:
    train, validation, test, report = split_checkins(make_checkins(), SPLIT_CONFIG)
    assert [len(train), len(validation), len(test)] == [6, 2, 2]
    assert train.utc_time.max() < validation.utc_time.min() < test.utc_time.min()
    assert report["quality_checks"] == {
        "event_count_preserved": True,
        "overlapping_event_ids": 0,
        "strict_global_time_order": True,
    }


def test_same_timestamp_never_crosses_a_boundary() -> None:
    data = make_checkins()
    data.loc[data["poi_id"].eq("p6"), "utc_time"] = data.loc[data["poi_id"].eq("p5"), "utc_time"].iloc[0]
    train, validation, _, _ = split_checkins(data, SPLIT_CONFIG)
    assert train.utc_time.max() < validation.utc_time.min()


def test_split_reports_closed_world_coverage() -> None:
    _, _, test, report = split_checkins(make_checkins(), SPLIT_CONFIG)
    assert report["cold_start"]["test_unseen_poi_rows"] == len(test)
    assert report["cold_start"]["test_evaluable_rows"] == 0


def test_split_rejects_wrong_method_and_random_shuffle() -> None:
    with pytest.raises(ValueError, match="global_chronological"):
        split_checkins(make_checkins(), {**SPLIT_CONFIG, "method": "per_user_chronological"})
    with pytest.raises(ValueError, match="random_shuffle"):
        split_checkins(make_checkins(), {**SPLIT_CONFIG, "random_shuffle": True})
