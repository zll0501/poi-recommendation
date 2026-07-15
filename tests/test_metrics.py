"""Unit tests for shared one-target ranking metrics."""

import math

import pytest

from src.metrics import evaluation_coverage, hit_rate_at_k, mrr_at_k, ndcg_at_k


def test_ranking_metrics_use_the_true_poi_rank() -> None:
    ranks = [1, 2, None, 5]

    assert hit_rate_at_k(ranks, 1) == 0.25
    assert hit_rate_at_k(ranks, 5) == 0.75
    assert ndcg_at_k(ranks, 5) == pytest.approx(
        (1.0 + 1 / math.log2(3) + 1 / math.log2(6)) / 4
    )
    assert mrr_at_k(ranks, 5) == pytest.approx((1.0 + 0.5 + 0.2) / 4)


def test_evaluation_coverage_counts_evaluable_targets() -> None:
    assert evaluation_coverage(8, 10) == 0.8


@pytest.mark.parametrize("function", [hit_rate_at_k, ndcg_at_k, mrr_at_k])
def test_ranking_metrics_reject_empty_or_invalid_input(function) -> None:
    with pytest.raises(ValueError):
        function([], 10)
    with pytest.raises(ValueError):
        function([0], 10)
    with pytest.raises(ValueError):
        function([1], 0)
