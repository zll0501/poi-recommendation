"""Tests for stage-one data preprocessing."""

from copy import deepcopy

import pandas as pd
import pytest

from src.preprocess import (
    assign_time_slots,
    clean_checkins,
    iterative_frequency_filter,
    standardize_checkins,
)


BASE_CONFIG = {
    "time": {
        "utc_format": "%a %b %d %H:%M:%S %z %Y",
        "slots": [
            {"name": "night", "start_hour": 0, "end_hour": 6},
            {"name": "morning", "start_hour": 6, "end_hour": 12},
            {"name": "afternoon", "start_hour": 12, "end_hour": 18},
            {"name": "evening", "start_hour": 18, "end_hour": 24},
        ],
    },
    "cleaning": {
        "remove_exact_duplicates": True,
        "simultaneous_conflict_strategy": "drop_group",
        "merge_consecutive_same_poi": True,
        "consecutive_same_poi_minutes": 10,
    },
    "filtering": {
        "min_user_checkins": 1,
        "min_poi_visits": 1,
        "iterative": True,
    },
}


def make_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["1", "2"],
            "poi_id": ["p1", "p2"],
            "category_id": ["c1", "c2"],
            "category_name": ["Cafe", "Park"],
            "latitude": [40.7, 40.8],
            "longitude": [-74.0, -73.9],
            "timezone_offset_minutes": [-240, -300],
            "utc_time_raw": [
                "Tue Apr 03 18:00:09 +0000 2012",
                "Wed Jan 02 04:30:00 +0000 2013",
            ],
        }
    )


def test_standardize_checkins_converts_local_time() -> None:
    result = standardize_checkins(make_raw_frame(), BASE_CONFIG)

    assert str(result.loc[0, "utc_time"]) == "2012-04-03 18:00:09+00:00"
    assert str(result.loc[0, "local_time"]) == "2012-04-03 14:00:09"
    assert result.loc[0, "hour"] == 14
    assert result.loc[0, "time_slot"] == "afternoon"
    assert result.loc[1, "hour"] == 23
    assert result.loc[1, "time_slot"] == "evening"


def test_invalid_timestamp_becomes_missing() -> None:
    frame = make_raw_frame()
    frame.loc[0, "utc_time_raw"] = "invalid"

    result = standardize_checkins(frame, BASE_CONFIG)

    assert pd.isna(result.loc[0, "utc_time"])
    assert pd.isna(result.loc[0, "local_time"])
    assert pd.isna(result.loc[0, "hour"])


def test_time_slots_must_cover_all_hours() -> None:
    invalid_config = deepcopy(BASE_CONFIG)
    invalid_config["time"]["slots"] = [
        {"name": "morning", "start_hour": 6, "end_hour": 12}
    ]

    with pytest.raises(ValueError, match="cover hours"):
        assign_time_slots(pd.Series([8]), invalid_config["time"]["slots"])


def test_cleaning_removes_duplicates_conflicts_and_short_repeats() -> None:
    rows = [
        ("1", "p1", "Tue Apr 03 18:00:00 +0000 2012"),
        ("1", "p1", "Tue Apr 03 18:00:00 +0000 2012"),
        ("1", "p2", "Tue Apr 03 18:20:00 +0000 2012"),
        ("1", "p3", "Tue Apr 03 18:20:00 +0000 2012"),
        ("1", "p4", "Tue Apr 03 18:30:00 +0000 2012"),
        ("1", "p4", "Tue Apr 03 18:38:00 +0000 2012"),
        ("1", "p4", "Tue Apr 03 19:00:00 +0000 2012"),
    ]
    raw = pd.DataFrame(
        {
            "user_id": [row[0] for row in rows],
            "poi_id": [row[1] for row in rows],
            "category_id": ["c"] * len(rows),
            "category_name": ["Category"] * len(rows),
            "latitude": [40.7] * len(rows),
            "longitude": [-74.0] * len(rows),
            "timezone_offset_minutes": [-240] * len(rows),
            "utc_time_raw": [row[2] for row in rows],
        }
    )
    standardized = standardize_checkins(raw, BASE_CONFIG)

    cleaned, report = clean_checkins(standardized, BASE_CONFIG)

    assert cleaned["poi_id"].tolist() == ["p1", "p4", "p4"]
    assert report["removed"]["exact_duplicate_rows"] == 1
    assert report["removed"]["simultaneous_conflict_groups"] == 1
    assert report["removed"]["simultaneous_conflict_rows"] == 2
    assert report["removed"]["short_consecutive_same_poi_rows"] == 1


def test_iterative_filter_reaches_fixed_point() -> None:
    data = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "poi_id": ["p1", "p1", "rare", "p1", "p2", "p2"],
        }
    )

    filtered, history = iterative_frequency_filter(
        data, min_user_checkins=3, min_poi_visits=2
    )

    assert filtered.empty
    assert len(history) >= 2
    assert history[-1]["removed_checkins"] == 0
