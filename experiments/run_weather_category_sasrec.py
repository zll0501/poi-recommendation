"""Train Category-SASRec with an independent query-weather residual."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import load_data_bundle
from src.evaluator import evaluate_next_poi, recommendations_to_frame
from src.layers.weather import WeatherCategoryScorer, WeatherEncoder
from src.models.sasrec import SASRec
from src.models.weather_category_sasrec import WeatherCategorySASRec
from src.sasrec_data import SASRecCollator, SASRecDataset
from src.sasrec_trainer import SASRecTrainer
from src.utils.config import load_yaml
from src.weather_features import load_weather_feature_store
from src.weather_sasrec_data import WeatherSASRecCollator, build_poi_category_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "valid_loss",
    "HitRate@5",
    "HitRate@10",
    "NDCG@5",
    "NDCG@10",
    "MRR@10",
]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _save_history(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: row[column] for column in HISTORY_COLUMNS} for row in rows
        )


def _best_epoch(rows: list[dict[str, Any]]) -> int:
    if not rows:
        raise ValueError("epoch history cannot be empty")
    return min(range(len(rows)), key=lambda index: float(rows[index]["valid_loss"])) + 1


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    if not bool(model_config.get("use_category", False)):
        raise ValueError("weather category fusion requires use_category: true")
    if not bool(model_config.get("use_query_time", False)):
        raise ValueError("weather experiment requires the query-time control baseline")

    seed = int(training_config["seed"])
    _set_seed(seed)
    data = load_data_bundle(_project_path(config["data_config"]))
    weather_model_config_path = _project_path(config["weather_model_config"])
    weather_config = load_yaml(weather_model_config_path)
    weather_store = load_weather_feature_store(weather_model_config_path)

    max_seq_len = int(model_config["max_seq_len"])
    use_history_time = bool(model_config.get("use_history_time", False))
    base_collator = SASRecCollator(
        max_seq_len,
        pad_id=data.pad_id,
        use_time=use_history_time,
        use_category=True,
        use_query_time=True,
    )
    collator = WeatherSASRecCollator(base_collator, weather_store)
    datasets = {
        name: SASRecDataset(data, name, max_seq_len)
        for name in ("train", "validation", "test")
    }
    batch_size = int(training_config["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_size=batch_size, collate_fn=collator
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=batch_size, collate_fn=collator
    )

    hidden_size = int(model_config["hidden_size"])
    base_model = SASRec(
        num_pois=data.vocabulary_size("poi_id"),
        num_categories=data.vocabulary_size("category_id"),
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        pad_id=data.pad_id,
        num_time_tokens=int(model_config.get("num_time_tokens", 25)),
        num_query_weekday_tokens=int(
            model_config.get("num_query_weekday_tokens", 8)
        ),
        num_query_time_slot_tokens=int(
            model_config.get("num_query_time_slot_tokens", 5)
        ),
        use_history_time=use_history_time,
        use_query_time=True,
        use_category=True,
    )
    encoder_config = weather_config["encoder"]
    output_dim = int(encoder_config["output_dim"])
    if output_dim != hidden_size:
        raise ValueError("weather encoder output_dim must equal SASRec hidden_size")
    weather_encoder = WeatherEncoder(
        num_weather_groups=weather_store.weather_group_vocab_size,
        numeric_dim=weather_store.numeric_dim,
        output_dim=output_dim,
        weather_embedding_dim=int(encoder_config["weather_embedding_dim"]),
        numeric_hidden_dim=int(encoder_config["numeric_hidden_dim"]),
        dropout=float(encoder_config["dropout"]),
    )
    poi_category_idx = build_poi_category_index(
        data.poi_metadata,
        num_pois=data.vocabulary_size("poi_id"),
        pad_id=data.pad_id,
        unknown_id=data.unknown_id,
    )
    model = WeatherCategorySASRec(
        base_model,
        weather_encoder,
        WeatherCategoryScorer(
            initial_weight=float(encoder_config["initial_weather_weight"])
        ),
        poi_category_idx,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    trainer = SASRecTrainer(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        _device(str(training_config.get("device", "auto"))),
    )

    def evaluate_validation_epoch(_: int) -> dict[str, float]:
        recommendations = trainer.predict(
            validation_loader,
            data.candidate_poi_ids,
            top_k=int(evaluation_config["output_k"]),
        )
        metrics = evaluate_next_poi(
            data.validation,
            recommendations,
            data.candidate_poi_ids,
            unknown_id=data.unknown_id,
            ks=tuple(int(value) for value in evaluation_config["top_k"]),
            mrr_k=int(evaluation_config["mrr_k"]),
        )
        return {
            name: float(metrics[name])
            for name in ("HitRate@5", "HitRate@10", "NDCG@5", "NDCG@10", "MRR@10")
        }

    history = trainer.fit(
        train_loader,
        validation_loader,
        epochs=int(training_config["epochs"]),
        patience=int(training_config["patience"]),
        epoch_evaluator=evaluate_validation_epoch,
        selection_metric="valid_loss",
        maximize_selection_metric=False,
    )
    recommendations = trainer.predict(
        test_loader,
        data.candidate_poi_ids,
        top_k=int(evaluation_config["output_k"]),
    )
    test_metrics = evaluate_next_poi(
        data.test,
        recommendations,
        data.candidate_poi_ids,
        unknown_id=data.unknown_id,
        ks=tuple(int(value) for value in evaluation_config["top_k"]),
        mrr_k=int(evaluation_config["mrr_k"]),
    )
    best_epoch = _best_epoch(history["epochs"])
    final_metrics = {
        name: float(test_metrics[name])
        for name in (
            "Coverage",
            "HitRate@5",
            "HitRate@10",
            "NDCG@5",
            "NDCG@10",
            "MRR@10",
        )
    }
    final_metrics.update(
        {
            "best_epoch": best_epoch,
            "epochs_trained": len(history["train_loss"]),
            "best_validation_loss": float(history["valid_loss"][best_epoch - 1]),
            "initial_weather_weight": float(
                encoder_config["initial_weather_weight"]
            ),
            "learned_weather_weight": float(model.weather_weight.detach().cpu()),
        }
    )

    output = config["output"]
    checkpoint_path = _project_path(output["checkpoint"])
    predictions_path = _project_path(output["predictions"])
    metrics_directory = _project_path(output["metrics_directory"])
    metrics_path = metrics_directory / "final_metrics.json"
    history_path = metrics_directory / "history.csv"
    for path in (checkpoint_path, predictions_path, metrics_path, history_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config}, checkpoint_path
    )
    recommendations_to_frame(recommendations).to_csv(
        predictions_path, index=False, encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_history(history["epochs"], history_path)
    return final_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/weather_query_time_category_sasrec.yaml"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
