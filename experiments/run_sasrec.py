"""按照项目统一协议训练并评价 SASRec 或 Category-SASRec。"""

from __future__ import annotations

import argparse
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
    use_time = bool(model_config.get("use_time", False))
    use_category = bool(model_config.get("use_category", False))
    collator = SASRecCollator(
        max_seq_len,
        pad_id=data.pad_id,
        use_time=use_time,
        use_category=use_category,
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
        use_time=use_time,
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
    loss_history = trainer.fit(
        train_loader,
        validation_loader,
        epochs=int(training_config["epochs"]),
        patience=int(training_config["patience"]),
    )
    recommendations = trainer.predict(
        test_loader,
        data.candidate_poi_ids,
        top_k=int(evaluation_config["output_k"]),
    )
    # 复用公共 evaluator，保证与 GRU、ItemCF 等模型的指标口径一致。
    metrics = evaluate_next_poi(
        data.test,
        recommendations,
        data.candidate_poi_ids,
        unknown_id=data.unknown_id,
        ks=tuple(int(value) for value in evaluation_config["top_k"]),
        mrr_k=int(evaluation_config["mrr_k"]),
    )
    metrics["model"] = str(model_config["name"])
    metrics["epochs_trained"] = len(loss_history["train_loss"])
    metrics["best_validation_loss"] = min(loss_history["valid_loss"])

    output = config["output"]
    checkpoint_path = _project_path(output["checkpoint"])
    predictions_path = _project_path(output["predictions"])
    metrics_path = _project_path(output["metrics"])
    for path in (checkpoint_path, predictions_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config}, checkpoint_path
    )
    recommendations_to_frame(recommendations).to_csv(
        predictions_path, index=False, encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate SASRec")
    parser.add_argument("--config", default="configs/sasrec.yaml")
    args = parser.parse_args()
    metrics = run(args.config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
