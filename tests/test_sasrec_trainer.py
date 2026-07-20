"""验证独立多输入 SASRec Trainer 的训练、记录与闭集预测。"""

import csv

import torch
from torch import nn

from experiments.run_sasrec import (
    HISTORY_COLUMNS,
    _best_epoch,
    _metrics_paths,
    _save_history,
)
from src.sasrec_trainer import SASRecTrainer


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4, padding_idx=0)
        self.output = nn.Linear(4, 8)

    def forward(self, poi_sequence, attention_mask, time_sequence=None):
        del time_sequence
        lengths = attention_mask.long().sum(dim=1).sub(1)
        starts = poi_sequence.size(1) - attention_mask.long().sum(dim=1)
        positions = starts + lengths
        hidden = self.embedding(poi_sequence)
        return self.output(hidden[torch.arange(len(hidden)), positions])


def _batch():
    return {
        "inputs": {
            "poi_sequence": torch.tensor([[0, 2, 3], [0, 4, 5]]),
            "time_sequence": torch.tensor([[0, 1, 2], [0, 3, 4]]),
            "attention_mask": torch.tensor(
                [[False, True, True], [False, True, True]]
            ),
        },
        "target": torch.tensor([4, 6]),
        "event_id": torch.tensor([10, 11]),
    }


def test_trainer_supports_nested_multi_input_batches_and_prediction():
    model = _TinyModel()
    trainer = SASRecTrainer(
        model,
        torch.optim.Adam(model.parameters(), lr=0.01),
        nn.CrossEntropyLoss(),
        "cpu",
    )

    def epoch_evaluator(epoch):
        return {
            "HitRate@5": 0.1 * epoch,
            "HitRate@10": 0.2 * epoch,
            "NDCG@5": 0.05 * epoch,
            "NDCG@10": 0.08 * epoch,
            "MRR@10": 0.04 * epoch,
        }

    history = trainer.fit(
        [_batch()],
        [_batch()],
        epochs=2,
        patience=1,
        epoch_evaluator=epoch_evaluator,
    )
    predictions = trainer.predict([_batch()], [2, 3, 4, 5, 6, 7], top_k=3)

    assert len(history["train_loss"]) >= 1
    assert len(history["valid_loss"]) >= 1
    assert len(history["epochs"]) == len(history["train_loss"])
    assert set(history["epochs"][0]) == set(HISTORY_COLUMNS)
    assert history["epochs"][0]["epoch"] == 1
    assert history["epochs"][0]["HitRate@5"] == 0.1
    assert set(predictions) == {10, 11}
    assert all(len(values) == 3 for values in predictions.values())
    assert all(set(values) <= {2, 3, 4, 5, 6, 7} for values in predictions.values())


def test_history_is_saved_with_the_required_csv_columns(tmp_path):
    rows = [
        {
            "epoch": 1,
            "train_loss": 2.0,
            "valid_loss": 1.5,
            "HitRate@5": 0.1,
            "HitRate@10": 0.2,
            "NDCG@5": 0.08,
            "NDCG@10": 0.12,
            "MRR@10": 0.07,
        }
    ]
    csv_path = tmp_path / "history.csv"

    _save_history(rows, csv_path)

    with csv_path.open(encoding="utf-8", newline="") as file:
        saved_rows = list(csv.DictReader(file))
    assert list(saved_rows[0]) == HISTORY_COLUMNS
    assert saved_rows[0]["epoch"] == "1"


def test_metric_paths_and_best_epoch_are_deterministic():
    for name in (
        "sasrec",
        "sasrec_category",
        "sasrec_time",
        "sasrec_time_category",
        "sasrec_query_time",
        "sasrec_query_time_category",
    ):
        metrics_path, history_path = _metrics_paths(name)
        assert metrics_path.as_posix().endswith(
            f"results/metrics/{name}/final_metrics.json"
        )
        assert history_path.as_posix().endswith(
            f"results/metrics/{name}/history.csv"
        )
    rows = [
        {"valid_loss": 3.0},
        {"valid_loss": 2.0},
        {"valid_loss": 2.5},
    ]
    assert _best_epoch(rows) == 2


def test_trainer_selects_checkpoint_by_minimum_validation_loss():
    model = _TinyModel()
    trainer = SASRecTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        nn.CrossEntropyLoss(),
        "cpu",
    )
    history = trainer.fit(
        [_batch()],
        [_batch()],
        epochs=3,
        patience=2,
        selection_metric="valid_loss",
        maximize_selection_metric=False,
    )

    assert len(history["valid_loss"]) >= 1
    assert all("valid_loss" in row for row in history["epochs"])
