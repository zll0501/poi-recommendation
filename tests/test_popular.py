"""Tests for the deterministic Global Popular baseline."""

from pathlib import Path

import pandas as pd
import pytest

from src.models.popular import GlobalPopular, TimePopular, TimeWeatherPopular


def make_train() -> pd.DataFrame:
    return pd.DataFrame({"poi_idx": [4, 2, 3, 2, 4, 2, 3]})


def test_global_popular_ranks_by_count_then_poi_id() -> None:
    model = GlobalPopular().fit(make_train())

    assert model.ranking_ == (2, 3, 4)
    assert model.ranking_frame().to_dict(orient="records") == [
        {"rank": 1, "poi_idx": 2, "visit_count": 3},
        {"rank": 2, "poi_idx": 3, "visit_count": 2},
        {"rank": 3, "poi_idx": 4, "visit_count": 2},
    ]


def test_global_popular_returns_shared_prediction_format() -> None:
    predictions = GlobalPopular().fit(make_train()).recommend(
        pd.DataFrame({"event_id": [10, 20]}), top_k=2
    )

    assert predictions.to_dict(orient="records") == [
        {"event_id": 10, "rank": 1, "poi_idx": 2},
        {"event_id": 10, "rank": 2, "poi_idx": 3},
        {"event_id": 20, "rank": 1, "poi_idx": 2},
        {"event_id": 20, "rank": 2, "poi_idx": 3},
    ]


def test_global_popular_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "global_popular.json"
    GlobalPopular().fit(make_train()).save(path)
    restored = GlobalPopular().load(path)

    assert restored.ranking_ == (2, 3, 4)
    assert restored.visit_counts_ == {2: 3, 3: 2, 4: 2}


def test_global_popular_rejects_invalid_usage() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        GlobalPopular().recommend(pd.DataFrame({"event_id": [1]}))
    with pytest.raises(ValueError, match="top_k"):
        GlobalPopular().fit(make_train()).recommend(
            pd.DataFrame({"event_id": [1]}), top_k=4
        )


def make_time_train() -> pd.DataFrame:
    return pd.DataFrame({
        "poi_idx": [2, 2, 3, 3, 3, 4],
        "time_slot": ["morning", "morning", "morning", "evening", "evening", "evening"],
    })


def test_time_popular_builds_slot_rankings_with_global_fallback() -> None:
    model = TimePopular().fit(make_time_train())

    assert model.rankings_["morning"] == (2, 3, 4)
    assert model.rankings_["evening"] == (3, 4, 2)


def test_time_popular_uses_query_slot_and_unknown_slot_fallback() -> None:
    model = TimePopular().fit(make_time_train())
    test = pd.DataFrame({
        "event_id": [10, 20, 30],
        "time_slot": ["morning", "evening", "unknown"],
    })

    predictions = model.recommend(test, top_k=2)
    grouped = predictions.groupby("event_id")["poi_idx"].apply(list).to_dict()
    assert grouped == {10: [2, 3], 20: [3, 4], 30: [3, 2]}


def test_time_popular_ranking_frame_marks_fallback_rows() -> None:
    ranking = TimePopular().fit(make_time_train()).ranking_frame()
    morning_poi4 = ranking.loc[
        ranking["time_slot"].eq("morning") & ranking["poi_idx"].eq(4)
    ].iloc[0]

    assert morning_poi4["visit_count"] == 0
    assert bool(morning_poi4["is_global_fallback"])


def test_time_popular_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "time_popular.json"
    TimePopular().fit(make_time_train()).save(path)
    restored = TimePopular().load(path)

    assert restored.rankings_["morning"] == (2, 3, 4)
    assert restored.visit_counts_["evening"] == {3: 2, 4: 1}


def test_time_popular_rejects_missing_time_slot() -> None:
    with pytest.raises(ValueError, match="time_slot"):
        TimePopular().fit(make_train())


def make_time_weather_train() -> pd.DataFrame:
    return pd.DataFrame({
        "poi_idx": [2, 2, 3, 3, 3, 4, 4],
        "time_slot": ["morning"] * 7,
        "weather_group": ["clear", "clear", "clear", "rain", "rain", "rain", "rain"],
    })


def test_time_weather_popular_uses_exact_bucket_and_time_fallback() -> None:
    model = TimeWeatherPopular(min_bucket_events=1).fit(make_time_weather_train())
    test = pd.DataFrame({
        "event_id": [10, 20, 30],
        "time_slot": ["morning", "morning", "unknown"],
        "weather_group": ["clear", "unknown", "rain"],
    })
    predictions = model.recommend(test, top_k=2)
    grouped = predictions.groupby("event_id")["poi_idx"].apply(list).to_dict()
    assert grouped[10] == [2, 3]
    assert grouped[20] == [3, 2]
    assert grouped[30] == [3, 2]


def test_time_weather_popular_skips_small_bucket() -> None:
    model = TimeWeatherPopular(min_bucket_events=4).fit(make_time_weather_train())
    assert ("morning", "clear") not in model.rankings_
    assert ("morning", "rain") in model.rankings_


def test_time_weather_popular_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "time_weather.json"
    original = TimeWeatherPopular(min_bucket_events=1).fit(make_time_weather_train())
    original.save(path)
    restored = TimeWeatherPopular().load(path)
    assert restored.rankings_ == original.rankings_
    assert restored.min_bucket_events == 1
