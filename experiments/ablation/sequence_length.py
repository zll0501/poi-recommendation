"""运行基础 GRU 的历史序列长度消融实验。"""

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
from src.torch_datasets import NextPOITorchDataset  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


DEFAULT_LENGTHS = (10, 20, 50)


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
    """在每个序列长度开始前重置随机状态。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _length_config(
    base_config: Mapping[str, Any],
    max_history: int,
    epochs_override: int | None,
) -> dict[str, Any]:
    """复制基础 GRU 配置并覆盖本次历史长度和可选训练轮数。"""
    config = copy.deepcopy(dict(base_config))
    data_config = dict(_mapping_section(config, "data"))
    training_config = dict(_mapping_section(config, "training"))
    data_config["max_history"] = max_history
    if epochs_override is not None:
        training_config["epochs"] = epochs_override
    config["data"] = data_config
    config["training"] = training_config
    return config


def _make_loaders(
    train_dataset: NextPOITorchDataset,
    valid_dataset: NextPOITorchDataset,
    test_dataset: NextPOITorchDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """用相同参数和随机种子创建三个数据分区的 DataLoader。"""
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


def _run_length(
    *,
    max_history: int,
    config: Mapping[str, Any],
    train_loader: Iterable[Any],
    valid_loader: Iterable[Any],
    test_loader: Iterable[Any],
    data_bundle: POIDataBundle,
    num_pois: int,
    padding_idx: int,
    device: torch.device,
    seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, list[float]],
    GRURecommender,
]:
    """训练、测试并评价一个历史序列长度。"""
    print(f"\n开始实验：GRU-L{max_history}")
    recommender = GRURecommender(
        config=config,
        num_pois=num_pois,
        padding_idx=padding_idx,
        device=device,
        candidate_poi_ids=data_bundle.candidate_poi_ids,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started_at = time.perf_counter()
    history = recommender.fit(train_data=train_loader, valid_data=valid_loader)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_time = time.perf_counter() - training_started_at
    if recommender.training_summary is None:
        raise RuntimeError("训练结束后未生成训练摘要")

    summary = recommender.training_summary
    trained_epochs = int(summary["trained_epochs"] or 0)
    if trained_epochs <= 0:
        raise RuntimeError("训练摘要中的 trained_epochs 必须为正整数")
    seconds_per_epoch = train_time / trained_epochs
    print(
        "训练摘要："
        + json.dumps(
            summary,
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
        "model": "GRU",
        "seed": seed,
        "max_history": max_history,
    }
    result.update(metrics)
    result.update(
        {
            "best_epoch": summary["best_epoch"],
            "best_valid_loss": summary["best_valid_loss"],
            "trained_epochs": trained_epochs,
            "stopped_early": summary["stopped_early"],
            "train_time": round(train_time, 3),
            "seconds_per_epoch": round(seconds_per_epoch, 3),
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result, history, recommender


def _save_checkpoint(
    recommender: GRURecommender,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    history: Mapping[str, list[float]],
    path: Path,
) -> None:
    """保存最佳模型权重及本次长度实验元数据。"""
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
    """覆盖写入本次已经完成的长度消融结果。"""
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
    """保存本次已经完成的损失历史和训练摘要。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(histories, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """解析序列长度消融实验参数。"""
    parser = argparse.ArgumentParser(description="运行基础 GRU 序列长度消融实验")
    parser.add_argument(
        "--config",
        default="configs/gru.yaml",
        help="基础 GRU 配置路径",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="公共数据配置路径",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_LENGTHS),
        help="需要比较的最大历史长度，默认运行 10、20、50",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="临时覆盖配置中的最大训练轮数，适合冒烟测试",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="训练设备，例如 auto、cpu、cuda 或 cuda:0",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="数据加载进程数")
    parser.add_argument("--seed", type=int, default=42, help="所有长度共用的随机种子")
    parser.add_argument(
        "--result-output",
        default="results/metrics/sequence_length.csv",
        help="长度消融结果 CSV 保存路径",
    )
    parser.add_argument(
        "--history-output",
        default="results/metrics/sequence_length_history.json",
        help="各长度损失历史保存路径",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="results/checkpoints/sequence_length",
        help="各长度 checkpoint 保存目录",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """依次执行选定的基础 GRU 历史长度实验。"""
    if args.num_workers < 0:
        raise ValueError("num_workers 不能为负数")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    if not args.lengths:
        raise ValueError("lengths 不得为空")
    if any(length <= 0 for length in args.lengths):
        raise ValueError("所有序列长度都必须为正整数")
    if len(set(args.lengths)) != len(args.lengths):
        raise ValueError("lengths 不能包含重复项")

    base_config = load_yaml(_project_path(args.config))
    model_config = _mapping_section(base_config, "model")
    if model_config.get("name") != "gru":
        raise ValueError("序列长度消融只接受 model.name 为 'gru' 的配置")
    training_config = _mapping_section(base_config, "training")
    batch_size = int(training_config.get("batch_size", 128))
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")

    device = _resolve_device(args.device)
    print(f"使用设备：{device}")
    print(f"实验长度：{', '.join(str(length) for length in args.lengths)}")
    print("正在加载公共数据……")
    data_bundle = load_data_bundle(_project_path(args.data_config))

    result_path = _project_path(args.result_output)
    history_path = _project_path(args.history_output)
    checkpoint_dir = _project_path(args.checkpoint_dir)
    results: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}

    expected_partition_sizes: tuple[int, int, int] | None = None
    for max_history in args.lengths:
        _set_seed(args.seed)
        config = _length_config(base_config, max_history, args.epochs)
        train_dataset = NextPOITorchDataset(
            data_bundle,
            "train",
            max_history=max_history,
        )
        valid_dataset = NextPOITorchDataset(
            data_bundle,
            "validation",
            max_history=max_history,
        )
        test_dataset = NextPOITorchDataset(
            data_bundle,
            "test",
            max_history=max_history,
            include_metadata=True,
        )
        if not train_dataset or not valid_dataset or not test_dataset:
            raise ValueError(f"GRU-L{max_history} 存在空数据分区")

        partition_sizes = (
            len(train_dataset),
            len(valid_dataset),
            len(test_dataset),
        )
        if expected_partition_sizes is None:
            expected_partition_sizes = partition_sizes
        elif partition_sizes != expected_partition_sizes:
            raise RuntimeError("不同序列长度产生了不一致的数据分区样本数")

        train_loader, valid_loader, test_loader = _make_loaders(
            train_dataset,
            valid_dataset,
            test_dataset,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed,
        )
        print(
            f"GRU-L{max_history}：训练样本 {len(train_dataset)}，"
            f"验证样本 {len(valid_dataset)}，测试样本 {len(test_dataset)}"
        )
        result, history, recommender = _run_length(
            max_history=max_history,
            config=config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            data_bundle=data_bundle,
            num_pois=train_dataset.num_pois,
            padding_idx=train_dataset.padding_idx,
            device=device,
            seed=args.seed,
        )
        results.append(result)
        histories[f"GRU-L{max_history}"] = {
            **history,
            "training_summary": recommender.training_summary,
        }

        checkpoint_path = checkpoint_dir / f"gru_len{max_history}.pt"
        _save_checkpoint(recommender, config, result, history, checkpoint_path)
        _write_results(results, result_path)
        _write_histories(histories, history_path)
        print(f"已保存 checkpoint：{checkpoint_path}")

        del recommender, train_loader, valid_loader, test_loader
        del train_dataset, valid_dataset, test_dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n长度消融结果已保存到：{result_path}")
    print(f"损失历史已保存到：{history_path}")
    return results


def main() -> None:
    """运行命令行序列长度消融入口。"""
    run(parse_args())


if __name__ == "__main__":
    main()
