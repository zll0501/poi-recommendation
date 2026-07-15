"""Tests for domain-aware POI dataset profiling."""

import pandas as pd

from src.analysis.data_profile import build_data_profile, haversine_km


PROFILE_CONFIG = {
    "coordinate_round_decimals": 5,
    "coordinate_tolerance_m": 100,
    "expected_timezone_offsets_minutes": [-300, -240],
    "suspicious_speed_kmh": 300,
    "popularity_top_fractions": [0.5],
}


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "poi_id": ["p1", "p2", "p1", "p1"],
            "category_id": ["c1", "c2", "changed", "c1"],
            "category_name": ["Cafe", "Park", "Changed", "Cafe"],
            "latitude": [40.0, 41.0, 40.1, 40.0],
            "longitude": [-74.0, -74.0, -74.0, -74.0],
            "timezone_offset_minutes": [-240] * 4,
            "utc_time": pd.to_datetime(
                [
                    "2024-01-01 10:00:00+00:00",
                    "2024-01-01 10:01:00+00:00",
                    "2024-01-02 10:00:00+00:00",
                    "2024-01-01 12:00:00+00:00",
                ],
                utc=True,
            ),
            "local_time": pd.to_datetime(
                ["2024-01-01 06:00", "2024-01-01 06:01", "2024-01-02 06:00", "2024-01-01 08:00"]
            ),
            "hour": [6, 6, 6, 8],
            "weekday": [0, 0, 1, 0],
            "is_weekend": [False] * 4,
            "time_slot": ["morning"] * 4,
        }
    )


def test_haversine_distance_is_geographically_meaningful() -> None:
    distance = haversine_km(
        pd.Series([0.0]), pd.Series([0.0]), pd.Series([1.0]), pd.Series([0.0])
    ).iloc[0]
    assert 111.0 < distance < 112.0


def test_profile_detects_long_tail_metadata_and_speed_without_deleting() -> None:
    data = make_data()
    partitions = {
        "train": data.iloc[:2],
        "validation": data.iloc[2:3],
        "test": data.iloc[3:],
    }
    report, users, pois, categories, suspicious = build_data_profile(
        data, partitions, PROFILE_CONFIG
    )

    assert report["scale"]["checkins"] == 4
    assert report["quality"]["inconsistent_metadata_pois"] == 1
    assert report["quality"]["inconsistent_category_pois"] == 1
    assert report["quality"]["inconsistent_coordinate_pois"] == 1
    assert report["quality"]["unexpected_timezone_offset_rows"] == 0
    assert report["sparsity_and_long_tail"]["unique_user_poi_pairs"] == 3
    assert report["behaviour"]["suspicious_speed_transitions"] == 1
    assert len(users) == 2
    assert len(pois) == 2
    assert len(categories) == 3
    assert len(suspicious) == 1
    assert report["scale"]["checkins"] == len(data)
