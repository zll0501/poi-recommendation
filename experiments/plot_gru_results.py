"""根据 GRU 实验结果生成汇报所需的最小图表集合。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RANKING_METRICS = (
    "HitRate@5",
    "NDCG@5",
    "HitRate@10",
    "NDCG@10",
    "MRR@10",
)
SEQUENCE_METRICS = ("HitRate@10", "NDCG@10", "MRR@10")
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2")


def _project_path(path: str | Path) -> Path:
    """将相对路径解析到项目根目录。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _read_json(path: Path) -> Mapping[str, Any]:
    """读取 JSON 对象并检查顶层结构。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少绘图输入文件：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON 顶层必须是对象：{path}")
    return payload


def _read_csv(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    """读取 CSV 并检查绘图所需字段。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少绘图输入文件：{path}")
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise ValueError(f"{path} 缺少字段：{missing}")
    if frame.empty:
        raise ValueError(f"绘图输入不得为空：{path}")
    return frame


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """以统一清晰度保存并关闭图表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _loss_values(
    history: Mapping[str, Any],
    name: str,
) -> list[float]:
    """读取一条非空损失序列。"""
    values = history.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"训练历史必须包含非空列表：{name}")
    return [float(value) for value in values]


def _plot_loss_curve(
    history: Mapping[str, Any],
    title: str,
    output_path: Path,
) -> None:
    """绘制训练损失、验证损失和最佳轮次。"""
    train_loss = _loss_values(history, "train_loss")
    valid_loss = _loss_values(history, "valid_loss")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        range(1, len(train_loss) + 1),
        train_loss,
        marker="o",
        markersize=3,
        label="Train loss",
        color=COLORS[0],
    )
    axis.plot(
        range(1, len(valid_loss) + 1),
        valid_loss,
        marker="o",
        markersize=3,
        label="Validation loss",
        color=COLORS[1],
    )

    summary = history.get("training_summary", {})
    if isinstance(summary, Mapping) and summary.get("best_epoch") is not None:
        best_epoch = int(summary["best_epoch"])
        axis.axvline(
            best_epoch,
            linestyle="--",
            linewidth=1.5,
            color=COLORS[2],
            label=f"Best epoch ({best_epoch})",
        )

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_figure(figure, output_path)


def _plot_grouped_metrics(
    frame: pd.DataFrame,
    labels: Sequence[str],
    metrics: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    """绘制多个模型或设置的分组指标柱状图。"""
    figure, axis = plt.subplots(figsize=(11, 5.5))
    x_positions = list(range(len(labels)))
    bar_width = 0.8 / len(metrics)
    start_offset = -0.4 + bar_width / 2
    for metric_index, metric in enumerate(metrics):
        offsets = [
            position + start_offset + metric_index * bar_width
            for position in x_positions
        ]
        axis.bar(
            offsets,
            frame[metric].astype(float),
            width=bar_width,
            label=metric,
            color=COLORS[metric_index % len(COLORS)],
        )

    axis.set_title(title)
    axis.set_ylabel("Score")
    axis.set_xticks(x_positions, labels, rotation=12)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3)
    _save_figure(figure, output_path)


def _plot_time_information_contribution(
    gru_result: pd.Series,
    time_gru_result: pd.Series,
    output_path: Path,
) -> None:
    """对比完整轨迹与“轨迹＋时间”输入，并标注时间信息的相对贡献。"""
    metrics = list(RANKING_METRICS)
    gru_scores = [float(gru_result[metric]) for metric in metrics]
    time_gru_scores = [float(time_gru_result[metric]) for metric in metrics]
    x_positions = list(range(len(metrics)))
    bar_width = 0.36

    figure, axis = plt.subplots(figsize=(11, 5.8))
    gru_bars = axis.bar(
        [position - bar_width / 2 for position in x_positions],
        gru_scores,
        width=bar_width,
        label="GRU: trajectory",
        color=COLORS[0],
    )
    time_gru_bars = axis.bar(
        [position + bar_width / 2 for position in x_positions],
        time_gru_scores,
        width=bar_width,
        label="Time-GRU: trajectory + time",
        color=COLORS[1],
    )

    axis.bar_label(gru_bars, fmt="%.3f", padding=3, fontsize=8)
    for bar, gru_score, time_gru_score in zip(
        time_gru_bars,
        gru_scores,
        time_gru_scores,
    ):
        relative_change = (
            (time_gru_score - gru_score) / gru_score * 100
            if gru_score != 0
            else 0.0
        )
        axis.annotate(
            f"{time_gru_score:.3f}\n({relative_change:+.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS[1],
        )

    axis.set_title("Contribution of Time Information to GRU Recommendation")
    axis.set_xlabel("Evaluation metric")
    axis.set_ylabel("Score")
    axis.set_xticks(x_positions, metrics)
    axis.set_ylim(0, max(gru_scores + time_gru_scores) * 1.22)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.text(
        0.5,
        -0.14,
        "Percentage labels show the relative change from GRU to Time-GRU.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
        color="#555555",
    )
    _save_figure(figure, output_path)


def _plot_sequence_metrics(frame: pd.DataFrame, output_path: Path) -> None:
    """绘制序列长度与主要推荐指标的关系。"""
    ordered = frame.sort_values("max_history", kind="stable")
    figure, axis = plt.subplots(figsize=(8, 5))
    for metric_index, metric in enumerate(SEQUENCE_METRICS):
        axis.plot(
            ordered["max_history"],
            ordered[metric],
            marker="o",
            linewidth=2,
            label=metric,
            color=COLORS[metric_index],
        )
    axis.set_title("GRU Performance by Maximum History Length")
    axis.set_xlabel("Maximum history length")
    axis.set_ylabel("Score")
    axis.set_xticks(ordered["max_history"].astype(int).tolist())
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, output_path)


def _plot_sequence_time(frame: pd.DataFrame, output_path: Path) -> None:
    """分别绘制不同序列长度的总训练时间和每轮时间。"""
    ordered = frame.sort_values("max_history", kind="stable")
    labels = ordered["max_history"].astype(int).astype(str)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, ordered["train_time"], color=COLORS[0])
    axes[0].set_title("Total Training Time")
    axes[0].set_xlabel("Maximum history length")
    axes[0].set_ylabel("Seconds")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, ordered["seconds_per_epoch"], color=COLORS[1])
    axes[1].set_title("Training Time per Epoch")
    axes[1].set_xlabel("Maximum history length")
    axes[1].set_ylabel("Seconds per epoch")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("GRU Training Cost by Maximum History Length")
    _save_figure(figure, output_path)


def parse_args() -> argparse.Namespace:
    """解析绘图输入和输出路径。"""
    parser = argparse.ArgumentParser(description="生成 GRU 实验汇报图")
    parser.add_argument(
        "--gru-history",
        default="results/metrics/gru_training_history.json",
    )
    parser.add_argument(
        "--time-gru-history",
        default="results/metrics/time_gru_training_history.json",
    )
    parser.add_argument(
        "--time-ablation",
        default="results/metrics/time_ablation.csv",
    )
    parser.add_argument(
        "--sequence-length",
        default="results/metrics/sequence_length.csv",
    )
    parser.add_argument("--gru-result", default="results/metrics/gru.csv")
    parser.add_argument("--time-gru-result", default="results/metrics/time_gru.csv")
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="图片输出目录",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> list[Path]:
    """读取实验产物并生成固定的七张图片。"""
    output_dir = _project_path(args.output_dir)
    gru_history = _read_json(_project_path(args.gru_history))
    time_gru_history = _read_json(_project_path(args.time_gru_history))
    time_ablation = _read_csv(
        _project_path(args.time_ablation),
        ("model", *RANKING_METRICS),
    )
    sequence_length = _read_csv(
        _project_path(args.sequence_length),
        (
            "max_history",
            *SEQUENCE_METRICS,
            "train_time",
            "seconds_per_epoch",
        ),
    )
    gru_result = _read_csv(
        _project_path(args.gru_result),
        ("model", "seed", *RANKING_METRICS),
    )
    time_gru_result = _read_csv(
        _project_path(args.time_gru_result),
        ("model", "seed", *RANKING_METRICS),
    )
    if int(gru_result.iloc[0]["seed"]) != int(time_gru_result.iloc[0]["seed"]):
        raise ValueError("最终 GRU 与 Time-GRU 必须使用相同 seed")

    output_paths = [
        output_dir / "gru_loss_curve.png",
        output_dir / "time_gru_loss_curve.png",
        output_dir / "time_ablation_metrics.png",
        output_dir / "sequence_length_metrics.png",
        output_dir / "sequence_length_training_time.png",
        output_dir / "final_model_comparison.png",
        output_dir / "time_information_contribution.png",
    ]
    _plot_loss_curve(gru_history, "GRU Training Curve", output_paths[0])
    _plot_loss_curve(
        time_gru_history,
        f"{time_gru_result.iloc[0]['model']} Training Curve",
        output_paths[1],
    )
    _plot_grouped_metrics(
        time_ablation,
        time_ablation["model"].astype(str).tolist(),
        RANKING_METRICS,
        "Time Feature Ablation",
        output_paths[2],
    )
    _plot_sequence_metrics(sequence_length, output_paths[3])
    _plot_sequence_time(sequence_length, output_paths[4])

    final_results = pd.concat(
        [gru_result.iloc[[0]], time_gru_result.iloc[[0]]],
        ignore_index=True,
    )
    _plot_grouped_metrics(
        final_results,
        final_results["model"].astype(str).tolist(),
        RANKING_METRICS,
        "Final GRU Model Comparison",
        output_paths[5],
    )
    _plot_time_information_contribution(
        gru_result.iloc[0],
        time_gru_result.iloc[0],
        output_paths[6],
    )

    for path in output_paths:
        print(f"已生成图片：{path}")
    return output_paths


def main() -> None:
    """运行命令行绘图入口。"""
    run(parse_args())


if __name__ == "__main__":
    main()
