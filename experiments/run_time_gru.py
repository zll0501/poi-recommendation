"""独立运行一种请求时间增强 GRU 配置。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import load_data_bundle  # noqa: E402
from src.evaluator import evaluate_next_poi, recommendations_to_frame  # noqa: E402
from src.models.time_gru import TimeGRURecommender  # noqa: E402
from src.torch_datasets import TimeAwareNextPOIDataset  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


def _project_path(path: str | Path) -> Path:
    """将相对路径解析为项目根目录下的路径。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _mapping_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """读取配置段并确保它是映射对象。"""
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise TypeError(f"配置字段 '{name}' 必须是映射对象")
    return section


def _resolve_device(device_name: str) -> torch.device:
    """解析训练设备，并检查显式请求的 CUDA 是否可用。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已请求 CUDA 设备，但当前环境无法使用 GPU")
    return device


def _set_seed(seed: int) -> None:
    """设置训练过程的随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练一种请求时间增强 GRU")
    parser.add_argument(
        "--config",
        default="configs/time_gru.yaml",
        help="Time-GRU 配置路径",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="公共数据配置路径",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="训练设备，例如 auto、cpu、cuda 或 cuda:0",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="数据加载进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/time_gru.pt",
        help="模型权重保存路径",
    )
    parser.add_argument(
        "--history-output",
        default="results/metrics/time_gru_training_history.json",
        help="训练损失历史保存路径",
    )
    parser.add_argument(
        "--result-output",
        default="results/metrics/time_gru.csv",
        help="统一实验结果 CSV 保存路径",
    )
    parser.add_argument(
        "--prediction-output",
        default=None,
        help="Top-10 推荐明细路径，默认按时间特征和 seed 保存",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """训练、评价一种 Time-GRU 配置并返回实验结果。"""
    if args.num_workers < 0:
        raise ValueError("num_workers 不能为负数")

    config = load_yaml(_project_path(args.config))
    model_config = _mapping_section(config, "model")
    if model_config.get("name") != "time_gru":
        raise ValueError("run_time_gru.py 只接受 model.name 为 'time_gru' 的配置")

    data_config = _mapping_section(config, "data")
    training_config = _mapping_section(config, "training")
    max_history = int(data_config.get("max_history", 20))
    batch_size = int(training_config.get("batch_size", 128))
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")

    device = _resolve_device(args.device)
    _set_seed(args.seed)
    print(f"使用设备：{device}")
    print("正在加载公共数据……")
    data_bundle = load_data_bundle(_project_path(args.data_config))

    print("正在构建 Time-GRU 训练集、验证集和测试集……")
    dataset_options = {
        "data_bundle": data_bundle,
        "max_history": max_history,
    }
    train_dataset = TimeAwareNextPOIDataset(partition="train", **dataset_options)
    valid_dataset = TimeAwareNextPOIDataset(
        partition="validation",
        **dataset_options,
    )
    test_dataset = TimeAwareNextPOIDataset(partition="test", **dataset_options)
    if len(train_dataset) == 0:
        raise ValueError("训练集没有可用的下一 POI 样本")
    if len(valid_dataset) == 0:
        raise ValueError("验证集没有可用的下一 POI 样本")
    if len(test_dataset) == 0:
        raise ValueError("测试集没有可评价的下一 POI 样本")

    loader_options = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **loader_options,
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    recommender = TimeGRURecommender(
        config=config,
        num_pois=train_dataset.num_pois,
        padding_idx=train_dataset.padding_idx,
        device=device,
        candidate_poi_ids=data_bundle.candidate_poi_ids,
    )
    print(
        f"模型：{recommender.model_name}，训练样本：{len(train_dataset)}，"
        f"验证样本：{len(valid_dataset)}，测试样本：{len(test_dataset)}，"
        f"POI 词表：{train_dataset.num_pois}"
    )

    training_started_at = time.perf_counter()
    history = recommender.fit(train_data=train_loader, valid_data=valid_loader)
    train_time = time.perf_counter() - training_started_at
    if recommender.training_summary is None:
        raise RuntimeError("训练结束后未生成训练摘要")
    print(
        "训练摘要："
        + json.dumps(
            recommender.training_summary,
            ensure_ascii=False,
        )
    )

    print("正在执行测试集 Top-10 推理和统一评价……")
    recommendations = recommender.recommend(test_data=test_loader, top_k=10)
    metrics = evaluate_next_poi(
        targets=data_bundle.test,
        predictions=recommendations,
        candidate_poi_ids=data_bundle.candidate_poi_ids,
        unknown_id=data_bundle.unknown_id,
        ks=(5, 10),
        mrr_k=10,
    )
    result: dict[str, Any] = {
        "model": recommender.model_name,
        "seed": int(args.seed),
    }
    result.update(metrics)
    result["train_time"] = round(train_time, 3)

    checkpoint_path = _project_path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {
        name: parameter.detach().cpu()
        for name, parameter in recommender.network.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": cpu_state_dict,
            "config": config,
            "num_pois": recommender.num_pois,
            "padding_idx": recommender.padding_idx,
            "training_history": history,
            "training_summary": recommender.training_summary,
            "train_time": round(train_time, 3),
            "evaluation_result": result,
        },
        checkpoint_path,
    )

    history_path = _project_path(args.history_output)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_output = {
        **history,
        "training_summary": recommender.training_summary,
    }
    history_path.write_text(
        json.dumps(history_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result_path = _project_path(args.result_output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    enabled_features = []
    if recommender.use_hour:
        enabled_features.append("hour")
    if recommender.use_time_slot:
        enabled_features.append("time_slot")
    if recommender.use_weekday:
        enabled_features.append("weekday")
    feature_slug = "_".join(enabled_features) or "no_time"
    prediction_output = args.prediction_output or (
        "results/predictions/"
        f"time_gru_{feature_slug}_seed{args.seed}_top10.csv"
    )
    prediction_path = _project_path(prediction_output)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame = recommendations_to_frame(recommendations)
    prediction_frame = prediction_frame.sort_values(
        ["event_id", "rank"],
        kind="stable",
    )
    prediction_frame.to_csv(prediction_path, index=False)

    print(f"训练完成，模型已保存到：{checkpoint_path}")
    print(f"损失历史已保存到：{history_path}")
    print(f"统一实验结果已保存到：{result_path}")
    print(f"Top-10 推荐明细已保存到：{prediction_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    """运行命令行 Time-GRU 训练入口。"""
    run(parse_args())


if __name__ == "__main__":
    main()
