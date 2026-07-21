import pytest

from experiments.run_ndcg_selected_sasrec import best_ndcg_epoch


def test_best_ndcg_epoch_uses_ranking_metric_not_loss() -> None:
    rows = [
        {"valid_loss": 4.8, "NDCG@10": 0.30},
        {"valid_loss": 4.9, "NDCG@10": 0.35},
        {"valid_loss": 5.0, "NDCG@10": 0.34},
    ]
    assert best_ndcg_epoch(rows) == 2


def test_best_ndcg_epoch_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="empty"):
        best_ndcg_epoch([])
