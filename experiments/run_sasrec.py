"""按照项目统一协议训练并评价 SASRec 或 Category-SASRec。"""

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
from src.models.sasrec import SASRec
from src.sasrec_data import SASRecCollator, SASRecDataset
from src.sasrec_trainer import SASRecTrainer
from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _set_seed(seed: int) -> None:
    """固定常用随机源，使两种模型的类别消融尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


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


def _save_history(rows: list[dict[str, Any]], csv_path: Path) -> None:
    """保存每个 epoch 的 loss 与验证集排名指标。"""
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: row[column] for column in HISTORY_COLUMNS} for row in rows
        )


def _metrics_paths(experiment_name: str) -> tuple[Path, Path]:
    """根据配置中的模型名生成互不覆盖的规范结果路径。"""
    if not experiment_name or Path(experiment_name).name != experiment_name:
        raise ValueError("model.name must be a non-empty directory-safe name")
    directory = PROJECT_ROOT / "results" / "metrics" / experiment_name
    return directory / "final_metrics.json", directory / "history.csv"


def _best_epoch(epoch_rows: list[dict[str, Any]]) -> int:
    """按照 validation loss 最小值选择最佳 epoch。"""
    if not epoch_rows:
        raise ValueError("epoch history cannot be empty")
    return min(
        range(len(epoch_rows)),
        key=lambda index: float(epoch_rows[index]["valid_loss"]),
    ) + 1


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    seed = int(training_config["seed"])
    _set_seed(seed)

    # 必须走公共数据入口，不能在模型脚本中重新切分或重新编码数据。
    data = load_data_bundle(_project_path(config["data_config"]))
    max_seq_len = int(model_config["max_seq_len"])
    # 兼容已有 use_time 配置，同时让新实验显式区分历史时间与 Query Time。
    use_history_time = bool(
        model_config.get("use_history_time", model_config.get("use_time", False))
    )
    use_query_time = bool(model_config.get("use_query_time", False))
    use_category = bool(model_config.get("use_category", False))
    collator = SASRecCollator(
        max_seq_len,
        pad_id=data.pad_id,
        use_time=use_history_time,
        use_category=use_category,
        use_query_time=use_query_time,
    )
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

    model = SASRec(
        num_pois=data.vocabulary_size("poi_id"),
        num_categories=data.vocabulary_size("category_id"),
        max_seq_len=max_seq_len,
        hidden_size=int(model_config["hidden_size"]),
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
        use_query_time=use_query_time,
        use_category=use_category,
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
        """计算当前 epoch 的验证集排名指标，不接触测试集。"""
        validation_recommendations = trainer.predict(
            validation_loader,
            data.candidate_poi_ids,
            top_k=int(evaluation_config["output_k"]),
        )
        validation_metrics = evaluate_next_poi(
            data.validation,
            validation_recommendations,
            data.candidate_poi_ids,
            unknown_id=data.unknown_id,
            ks=tuple(int(value) for value in evaluation_config["top_k"]),
            mrr_k=int(evaluation_config["mrr_k"]),
        )
        return {
            "HitRate@5": float(validation_metrics["HitRate@5"]),
            "HitRate@10": float(validation_metrics["HitRate@10"]),
            "NDCG@5": float(validation_metrics["NDCG@5"]),
            "NDCG@10": float(validation_metrics["NDCG@10"]),
            "MRR@10": float(validation_metrics["MRR@10"]),
        }

    loss_history = trainer.fit(
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
    # 复用公共 evaluator，保证与 GRU、ItemCF 等模型的指标口径一致。
    test_metrics = evaluate_next_poi(
        data.test,
        recommendations,
        data.candidate_poi_ids,
        unknown_id=data.unknown_id,
        ks=tuple(int(value) for value in evaluation_config["top_k"]),
        mrr_k=int(evaluation_config["mrr_k"]),
    )
    valid_losses = loss_history["valid_loss"]
    best_epoch = _best_epoch(loss_history["epochs"])
    final_metrics = {
        "Coverage": float(test_metrics["Coverage"]),
        "HitRate@5": float(test_metrics["HitRate@5"]),
        "HitRate@10": float(test_metrics["HitRate@10"]),
        "NDCG@5": float(test_metrics["NDCG@5"]),
        "NDCG@10": float(test_metrics["NDCG@10"]),
        "MRR@10": float(test_metrics["MRR@10"]),
        "best_epoch": best_epoch,
        "epochs_trained": len(loss_history["train_loss"]),
        "best_validation_loss": float(valid_losses[best_epoch - 1]),
    }

    output = config["output"]
    checkpoint_path = _project_path(output["checkpoint"])
    predictions_path = _project_path(output["predictions"])
    experiment_name = str(model_config["name"])
    metrics_path, history_csv_path = _metrics_paths(experiment_name)
    for path in (
        checkpoint_path,
        predictions_path,
        metrics_path,
        history_csv_path,
    ):
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
    _save_history(loss_history["epochs"], history_csv_path)
    return final_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate SASRec")
    parser.add_argument("--config", default="configs/sasrec.yaml")
    args = parser.parse_args()
    metrics = run(args.config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
