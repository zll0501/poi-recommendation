"""Tests for leakage-safe spatial transition analysis."""

import numpy as np
import pandas as pd
import pytest

from src.spatial_reranker import (
    build_query_contexts,
    build_training_transitions,
    haversine_km,
    rerank_by_distance,
    summarize_training_transitions,
)


def make_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [3, 1, 4, 2, 5],
            "user_idx": [2, 2, 3, 2, 3],
            "poi_idx": [12, 11, 21, 10, 22],
            "timestamp": [7200, 3600, 1800, 0, 9000],
            "latitude": [0.0, 0.0, 40.0, 0.0, 40.01],
            "longitude": [1.0, 0.0, -74.0, 0.0, -74.0],
        }
    )


def test_haversine_is_vectorized_and_geographically_meaningful() -> None:
    distance = haversine_km([0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0])
    assert distance.shape == (2,)
    assert 111.0 < distance[0] < 112.0
    assert distance[1] == pytest.approx(0.0)


def test_transitions_are_sorted_within_user_and_never_cross_users() -> None:
    transitions = build_training_transitions(make_train())
    assert len(transitions) == 3
    assert transitions["user_idx"].tolist() == [2, 2, 3]
    assert transitions["previous_poi_idx"].astype(int).tolist() == [10, 11, 21]
    assert transitions["poi_idx"].astype(int).tolist() == [11, 12, 22]
    assert transitions["time_gap_hours"].gt(0).all()


def test_summary_reports_locality_and_time_gap_groups() -> None:
    transitions = pd.DataFrame(
        {
            "user_idx": [2, 2, 3, 3],
            "distance_km": [0.0, 2.0, 6.0, 12.0],
            "time_gap_hours": [0.5, 2.0, 12.0, 200.0],
            "same_poi": [True, False, False, False],
        }
    )
    report, by_time = summarize_training_transitions(transitions, [1, 5, 10])
    assert report["fit_partition"] == "train_only"
    assert report["transition_count"] == 4
    assert report["distance_locality"]["within_5km_ratio"] == pytest.approx(0.5)
    assert report["same_poi_ratio"] == pytest.approx(0.25)
    assert set(by_time["time_gap_bucket"]) == {"0-1h", "1-6h", "6-24h", ">7d"}


def test_invalid_coordinates_are_rejected() -> None:
    train = make_train()
    train.loc[0, "latitude"] = np.nan
    with pytest.raises(ValueError, match="invalid spatial rows"):
        build_training_transitions(train)


def test_query_context_uses_previous_event_not_target_location() -> None:
    prior = pd.DataFrame(
        {
            "event_id": [1],
            "user_id": ["u1"],
            "timestamp": [0],
            "latitude": [40.0],
            "longitude": [-74.0],
        }
    )
    targets = pd.DataFrame(
        {
            "event_id": [2, 3],
            "user_id": ["u1", "u1"],
            "timestamp": [3600, 10800],
            "latitude": [41.0, 42.0],
            "longitude": [-73.0, -72.0],
        }
    )
    contexts = build_query_contexts(prior, targets).set_index("event_id")
    assert contexts.loc[2, "last_latitude"] == pytest.approx(40.0)
    assert contexts.loc[3, "last_latitude"] == pytest.approx(41.0)
    assert contexts.loc[3, "time_gap_hours"] == pytest.approx(2.0)


def test_soft_penalty_can_promote_a_nearby_candidate() -> None:
    poi_ids = [10, 11, 12]
    scores = [1.0, 0.9, 0.8]
    distances = [20.0, 1.0, 2.0]
    assert rerank_by_distance(poi_ids, scores, distances, 0.0, top_k=2) == [10, 11]
    assert rerank_by_distance(poi_ids, scores, distances, 0.2, top_k=2) == [11, 12]
