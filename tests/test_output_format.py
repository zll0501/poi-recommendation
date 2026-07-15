"""Tests for the shared prediction table and closed-world evaluator."""

import math

import pandas as pd
import pytest

from src.evaluator import evaluate_next_poi, recommendations_to_frame


def make_targets() -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": [10, 11, 12, 13],
        "user_idx": [2, 3, 1, 2],
        "poi_idx": [2, 5, 2, 1],
    })


def make_predictions() -> dict[int, list[int]]:
    return {10: [2, 3, 4], 11: [2, 5, 4]}


def test_recommendation_mapping_becomes_long_table() -> None:
    frame = recommendations_to_frame({10: [4, 2]})
    assert frame.to_dict(orient="records") == [
        {"event_id": 10, "rank": 1, "poi_idx": 4},
        {"event_id": 10, "rank": 2, "poi_idx": 2},
    ]


def test_evaluator_filters_cold_start_and_computes_shared_metrics() -> None:
    metrics = evaluate_next_poi(
        make_targets(), make_predictions(), [2, 3, 4, 5, 6], ks=(1, 3), mrr_k=3
    )

    assert metrics["Coverage"] == 0.5
    assert metrics["HitRate@1"] == 0.5
    assert metrics["HitRate@3"] == 1.0
    assert metrics["NDCG@1"] == 0.5
    assert metrics["NDCG@3"] == pytest.approx((1 + 1 / math.log2(3)) / 2)
    assert metrics["MRR@3"] == 0.75
    assert set(metrics) == {
        "Coverage",
        "HitRate@1",
        "HitRate@3",
        "NDCG@1",
        "NDCG@3",
        "MRR@3",
    }


@pytest.mark.parametrize(
    "predictions, error",
    [
        ({10: [2, 3, 4]}, "exactly match"),
        ({10: [2, 3, 4], 11: [2, 99, 4]}, "outside"),
        ({10: [2, 2, 4], 11: [2, 5, 4]}, "twice"),
        ({10: [2, 3], 11: [2, 5, 4]}, "exactly 3"),
    ],
)
def test_evaluator_rejects_unfair_or_malformed_predictions(predictions, error) -> None:
    with pytest.raises(ValueError, match=error):
        evaluate_next_poi(
            make_targets(), predictions, [2, 3, 4, 5, 6], ks=(1, 3), mrr_k=3
        )
