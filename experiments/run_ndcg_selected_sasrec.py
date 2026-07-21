"""Run an existing SASRec experiment with validation NDCG@10 selection.

This adapter deliberately leaves the original SASRec and weather experiment
implementations unchanged. It replaces only the trainer class used by the
selected runner and writes results to the distinct paths in the supplied
configuration.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from src.sasrec_trainer import SASRecTrainer
from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_MODULES = {
    "baseline": "experiments.run_sasrec",
    "weather": "experiments.run_weather_category_sasrec",
}


class NDCGSelectionTrainer(SASRecTrainer):
    """Force checkpoint selection and early stopping by validation NDCG@10."""

    def fit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["selection_metric"] = "NDCG@10"
        kwargs["maximize_selection_metric"] = True
        return super().fit(*args, **kwargs)


def best_ndcg_epoch(epoch_rows: list[dict[str, Any]]) -> int:
    if not epoch_rows:
        raise ValueError("epoch history cannot be empty")
    return max(
        range(len(epoch_rows)),
        key=lambda index: float(epoch_rows[index]["NDCG@10"]),
    ) + 1


def _metrics_path(config: dict[str, Any], runner: str) -> Path:
    if runner == "weather":
        directory = Path(config["output"]["metrics_directory"])
        directory = directory if directory.is_absolute() else PROJECT_ROOT / directory
        return directory / "final_metrics.json"
    experiment_name = str(config["model"]["name"])
    return PROJECT_ROOT / "results" / "metrics" / experiment_name / "final_metrics.json"


def run(config_path: str | Path, runner: str) -> dict[str, Any]:
    if runner not in RUNNER_MODULES:
        raise ValueError(f"unknown runner: {runner}")
    config = load_yaml(config_path)
    module = importlib.import_module(RUNNER_MODULES[runner])
    module.SASRecTrainer = NDCGSelectionTrainer
    module._best_epoch = best_ndcg_epoch
    metrics = module.run(config_path)

    history_path = (
        PROJECT_ROOT
        / "results"
        / "metrics"
        / str(config["model"]["name"])
        / "history.csv"
    )
    if runner == "weather":
        output_directory = Path(config["output"]["metrics_directory"])
        output_directory = (
            output_directory
            if output_directory.is_absolute()
            else PROJECT_ROOT / output_directory
        )
        history_path = output_directory / "history.csv"

    import pandas as pd

    history = pd.read_csv(history_path)
    selected_row = history.loc[history["NDCG@10"].idxmax()]
    metrics["selection_metric"] = "NDCG@10"
    metrics["best_validation_ndcg10"] = float(selected_row["NDCG@10"])
    metrics["selected_epoch_validation_loss"] = float(selected_row["valid_loss"])
    metrics.pop("best_validation_loss", None)
    _metrics_path(config, runner).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=sorted(RUNNER_MODULES), required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.runner), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
