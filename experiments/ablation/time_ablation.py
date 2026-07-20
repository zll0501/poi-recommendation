"""运行 GRU 时间信息消融实验。"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import random
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import POIDataBundle, load_data_bundle  # noqa: E402
from src.evaluator import evaluate_next_poi  # noqa: E402
from src.models.gru import GRURecommender  # noqa: E402
from src.models.time_gru import TimeGRURecommender  # noqa: E402
from src.torch_datasets import (  # noqa: E402
    NextPOITorchDataset,
    TimeAwareNextPOIDataset,
)
from src.utils.config import load_yaml  # noqa: E402


VARIANTS: dict[str, tuple[str, dict[str, bool]]] = {
    "gru": (
        "GRU",
        {"hour": False, "time_slot": False, "weekday": False},
    ),
    "hour": (
        "GRU+Hour",
        {"hour": True, "time_slot": False, "weekday": False},
    ),
    "time_slot": (
        "GRU+TimeSlot",
        {"hour": False, "time_slot": True, "weekday": False},
    ),
    "hour_weekday": (
        "GRU+Hour+Weekday",
        {"hour": True, "time_slot": False, "weekday": True},
    ),
}


def _project_path(path: str | Path) -> Path:
    """将相对路径解析到项目根目录。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _mapping_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """读取配置段并保证它是映射对象。"""
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise TypeError(f"配置字段 '{name}' 必须是映射对象")
    return section


def _resolve_device(device_name: str) -> torch.device:
    """解析实验使用的 CPU 或 GPU 设备。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已请求 CUDA 设备，但当前环境无法使用 GPU")
    return device


def _set_seed(seed: int) -> None:
    """在每个变体开始前重置随机状态。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _variant_config(
    base_config: Mapping[str, Any],
    variant_key: str,
    epochs_override: int | None,
) -> dict[str, Any]:
    """复制公共配置并应用一个消融变体。"""
    if variant_key not in VARIANTS:
        raise ValueError(f"未知实验变体：{variant_key}")
    config = copy.deepcopy(dict(base_config))
    model_config = dict(_mapping_section(config, "model"))
    training_config = dict(_mapping_section(config, "training"))
    _, time_features = VARIANTS[variant_key]

    if variant_key == "gru":
        model_config["name"] = "gru"
        model_config.pop("use_time", None)
    else:
        model_config["name"] = "time_gru"
        model_config["use_time"] = True
    if epochs_override is not None:
        training_config["epochs"] = epochs_override

    config["model"] = model_config
    config["training"] = training_config
    config["time_features"] = dict(time_features)
    return config


def _make_loaders(
    train_dataset: Any,
    valid_dataset: Any,
    test_dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """用相同参数创建训练、验证和测试 DataLoader。"""
    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            generator=train_generator,
            **options,
        ),
        DataLoader(valid_dataset, shuffle=False, **options),
        DataLoader(test_dataset, shuffle=False, **options),
    )


def _run_variant(
    *,
    model_name: str,
    recommender: GRURecommender | TimeGRURecommender,
    train_loader: Iterable[Any],
    valid_loader: Iterable[Any],
    test_loader: Iterable[Any],
    data_bundle: POIDataBundle,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """训练、测试并评价一个时间消融变体。"""
    print(f"\n开始实验：{model_name}")
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
        "model": model_name,
        "seed": seed,
    }
    result.update(metrics)
    result["train_time"] = round(train_time, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result, history


def _save_checkpoint(
    recommender: GRURecommender | TimeGRURecommender,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    history: Mapping[str, list[float]],
    path: Path,
) -> None:
    """保存便于跨 CPU/GPU 加载的模型权重与实验元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {
        name: parameter.detach().cpu()
        for name, parameter in recommender.network.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": cpu_state_dict,
            "config": dict(config),
            "num_pois": recommender.num_pois,
            "padding_idx": recommender.padding_idx,
            "training_history": dict(history),
            "training_summary": recommender.training_summary,
            "evaluation_result": dict(result),
        },
        path,
    )


def _write_results(results: list[dict[str, Any]], path: Path) -> None:
    """覆盖写入本次已完成变体的统一评价结果。"""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def _write_histories(
    histories: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    """保存本次已完成变体的训练和验证损失。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(histories, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """解析时间消融实验参数。"""
    parser = argparse.ArgumentParser(description="运行 GRU 时间信息消融实验")
    parser.add_argument(
        "--config",
        default="configs/time_gru.yaml",
        help="Time-GRU 公共配置路径",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="公共数据配置路径",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
        help="需要运行的变体，默认按顺序运行全部四组",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="临时覆盖配置中的训练轮数，适合冒烟测试",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="训练设备，例如 auto、cpu、cuda 或 cuda:0",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="数据加载进程数")
    parser.add_argument("--seed", type=int, default=42, help="所有变体共用的随机种子")
    parser.add_argument(
        "--result-output",
        default="results/metrics/time_ablation.csv",
        help="消融结果 CSV 保存路径",
    )
    parser.add_argument(
        "--history-output",
        default="results/metrics/time_ablation_history.json",
        help="各变体损失历史保存路径",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="results/checkpoints/time_ablation",
        help="各变体 checkpoint 保存目录",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """依次执行选定的时间消融变体。"""
    if args.num_workers < 0:
        raise ValueError("num_workers 不能为负数")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("variants 不能包含重复项")

    base_config = load_yaml(_project_path(args.config))
    data_config = _mapping_section(base_config, "data")
    training_config = _mapping_section(base_config, "training")
    max_history = int(data_config.get("max_history", 20))
    batch_size = int(training_config.get("batch_size", 128))
    if max_history <= 0:
        raise ValueError("max_history 必须为正整数")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")

    device = _resolve_device(args.device)
    print(f"使用设备：{device}")
    print(f"实验顺序：{', '.join(VARIANTS[key][0] for key in args.variants)}")
    print("正在加载公共数据……")
    data_bundle = load_data_bundle(_project_path(args.data_config))

    result_path = _project_path(args.result_output)
    history_path = _project_path(args.history_output)
    checkpoint_dir = _project_path(args.checkpoint_dir)
    results: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}

    for variant_key in args.variants:
        _set_seed(args.seed)
        model_name, _ = VARIANTS[variant_key]
        config = _variant_config(base_config, variant_key, args.epochs)

        if variant_key == "gru":
            train_dataset = NextPOITorchDataset(
                data_bundle, "train", max_history=max_history
            )
            valid_dataset = NextPOITorchDataset(
                data_bundle, "validation", max_history=max_history
            )
            test_dataset = NextPOITorchDataset(
                data_bundle,
                "test",
                max_history=max_history,
                include_metadata=True,
            )
            recommender: GRURecommender | TimeGRURecommender = GRURecommender(
                config=config,
                num_pois=train_dataset.num_pois,
                padding_idx=train_dataset.padding_idx,
                device=device,
                candidate_poi_ids=data_bundle.candidate_poi_ids,
            )
        else:
            train_dataset = TimeAwareNextPOIDataset(
                data_bundle, "train", max_history=max_history
            )
            valid_dataset = TimeAwareNextPOIDataset(
                data_bundle, "validation", max_history=max_history
            )
            test_dataset = TimeAwareNextPOIDataset(
                data_bundle, "test", max_history=max_history
            )
            recommender = TimeGRURecommender(
                config=config,
                num_pois=train_dataset.num_pois,
                padding_idx=train_dataset.padding_idx,
                device=device,
                candidate_poi_ids=data_bundle.candidate_poi_ids,
            )
            if recommender.model_name != model_name:
                raise RuntimeError(
                    f"变体名称不一致：预期 {model_name}，实际 {recommender.model_name}"
                )

        if not train_dataset or not valid_dataset or not test_dataset:
            raise ValueError(f"{model_name} 存在空数据分区")
        train_loader, valid_loader, test_loader = _make_loaders(
            train_dataset,
            valid_dataset,
            test_dataset,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed,
        )
        result, history = _run_variant(
            model_name=model_name,
            recommender=recommender,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            data_bundle=data_bundle,
            seed=args.seed,
        )
        results.append(result)
        histories[model_name] = {
            **history,
            "training_summary": recommender.training_summary,
        }

        checkpoint_path = checkpoint_dir / f"{variant_key}.pt"
        _save_checkpoint(recommender, config, result, history, checkpoint_path)
        _write_results(results, result_path)
        _write_histories(histories, history_path)
        print(f"已保存 checkpoint：{checkpoint_path}")

        del recommender, train_loader, valid_loader, test_loader
        del train_dataset, valid_dataset, test_dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n消融结果已保存到：{result_path}")
    print(f"损失历史已保存到：{history_path}")
    return results


def main() -> None:
    """运行命令行时间消融入口。"""
    run(parse_args())


if __name__ == "__main__":
    main()
