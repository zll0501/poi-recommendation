"""Create report-ready figures for weather and spatial extensions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "metrics"
FIGURES = PROJECT_ROOT / "results" / "figures"

COLORS = {
    "baseline": "#7A8793",
    "weather": "#3B82B8",
    "distance": "#3A9D73",
    "combined": "#E58A3A",
    "grid": "#D9DEE5",
    "text": "#263442",
    "muted": "#667684",
}


def _configure_style() -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "axes.titleweight": "normal",
            "font.size": 10,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finish(fig: plt.Figure, filename: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_four_model_comparison() -> Path:
    baseline = _read_json(RESULTS / "spatial_reranking" / "baseline_metrics.json")
    weather = _read_json(RESULTS / "spatial_reranking" / "weather_metrics.json")
    labels = ["Category-\nSASRec", "+ 天气", "+ 距离", "+ 天气\n+ 距离"]
    values = {
        "HitRate@10": [
            baseline["test_unreranked"]["HitRate@10"],
            weather["test_unreranked"]["HitRate@10"],
            baseline["test_distance"]["HitRate@10"],
            weather["test_distance"]["HitRate@10"],
        ],
        "NDCG@10": [
            baseline["test_unreranked"]["NDCG@10"],
            weather["test_unreranked"]["NDCG@10"],
            baseline["test_distance"]["NDCG@10"],
            weather["test_distance"]["NDCG@10"],
        ],
    }
    bar_colors = [
        COLORS["baseline"],
        COLORS["weather"],
        COLORS["distance"],
        COLORS["combined"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    limits = {"HitRate@10": (0.535, 0.5615), "NDCG@10": (0.376, 0.386)}
    for axis, (metric, metric_values) in zip(axes, values.items()):
        positions = np.arange(len(labels))
        bars = axis.bar(positions, metric_values, width=0.66, color=bar_colors)
        axis.set_xticks(positions, labels)
        axis.set_ylim(*limits[metric])
        axis.set_title(metric, pad=12)
        axis.set_ylabel("指标值（纵轴局部放大）")
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axhline(metric_values[0], color=COLORS["baseline"], linestyle="--", linewidth=1)
        for bar, value in zip(bars, metric_values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (limits[metric][1] - limits[metric][0]) * 0.018,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("天气与距离上下文的四组消融对比", fontsize=15, y=1.02)
    fig.text(
        0.5,
        0.0,
        "同一数据、候选集与 NDCG@10 选模协议；虚线为 Category-SASRec 基线",
        ha="center",
        color=COLORS["muted"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), w_pad=2.5)
    return _finish(fig, "context_four_model_comparison.png")


def plot_transition_distance() -> Path:
    data = pd.read_csv(RESULTS / "spatial_transition_by_time_gap.csv")
    labels = data["time_gap_bucket"].replace({">7d": ">7天"}).tolist()
    labels = [
        value.replace("0-1h", "0–1小时")
        .replace("1-6h", "1–6小时")
        .replace("6-24h", "6–24小时")
        .replace("1-7d", "1–7天")
        for value in labels
    ]
    positions = np.arange(len(data))
    median = data["median_distance_km"].to_numpy()
    p90 = data["p90_distance_km"].to_numpy()
    fig, axis = plt.subplots(figsize=(9.6, 5.4))
    bars = axis.bar(
        positions,
        median,
        width=0.58,
        color=COLORS["weather"],
        label="中位移动距离",
        zorder=2,
    )
    axis.plot(
        positions,
        p90,
        color=COLORS["combined"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="90%分位移动距离",
        zorder=3,
    )
    for bar, value in zip(bars, median):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for x, value in zip(positions, p90):
        axis.annotate(
            f"{value:.2f}",
            (x, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("相邻签到移动距离（公里）")
    axis.set_xlabel("相邻签到时间间隔")
    axis.set_ylim(0, max(p90) * 1.17)
    axis.set_title("时间间隔越长，用户真实移动距离越大", fontsize=15, pad=14)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    fig.text(
        0.5,
        0.01,
        "仅使用训练集 141,459 条连续签到转移统计，避免验证集和测试集信息泄漏",
        ha="center",
        color=COLORS["muted"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _finish(fig, "transition_distance_by_time_gap.png")


def _effect_rows() -> pd.DataFrame:
    weather = _read_json(
        RESULTS
        / "weather_query_time_category_sasrec_ndcg"
        / "bootstrap_significance.json"
    )
    distance = _read_json(RESULTS / "spatial_reranking" / "baseline_significance.json")
    combined = _read_json(RESULTS / "spatial_reranking" / "weather_significance.json")
    comparisons = [
        ("加入天气", weather, COLORS["weather"]),
        ("加入距离", distance, COLORS["distance"]),
        ("天气模型再加入距离", combined, COLORS["combined"]),
    ]
    rows: list[dict[str, Any]] = []
    for label, result, color in comparisons:
        for metric in ("HitRate@10", "NDCG@10"):
            item = result["metrics"][metric]
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "delta_pp": 100 * item["absolute_delta"],
                    "lower_pp": 100 * item["ci95_lower"],
                    "upper_pp": 100 * item["ci95_upper"],
                    "p": item["two_sided_bootstrap_p"],
                    "color": color,
                }
            )
    return pd.DataFrame(rows)


def plot_context_confidence_intervals() -> Path:
    data = _effect_rows()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4), sharey=True)
    labels = ["加入天气", "加入距离", "天气模型再加入距离"]
    y_positions = np.arange(len(labels))[::-1]
    for axis, metric in zip(axes, ("HitRate@10", "NDCG@10")):
        subset = data.loc[data["metric"] == metric].set_index("comparison").loc[labels]
        for y, (_, row) in zip(y_positions, subset.iterrows()):
            axis.errorbar(
                row["delta_pp"],
                y,
                xerr=np.array(
                    [[row["delta_pp"] - row["lower_pp"]], [row["upper_pp"] - row["delta_pp"]]]
                ),
                fmt="o",
                color=row["color"],
                ecolor=row["color"],
                elinewidth=2,
                capsize=4,
                markersize=7,
                zorder=3,
            )
            p_label = "p<0.001" if row["p"] < 0.001 else f"p={row['p']:.3f}"
            axis.text(
                row["upper_pp"] + (0.055 if metric == "HitRate@10" else 0.025),
                y,
                f"Δ {row['delta_pp']:.3f}  {p_label}",
                va="center",
                fontsize=8.5,
            )
        axis.axvline(0, color=COLORS["baseline"], linewidth=1.1, linestyle="--")
        axis.grid(axis="x", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.set_yticks(y_positions, labels)
        axis.tick_params(axis="y", length=0)
        axis.set_xlabel("绝对提升（百分点）")
        axis.set_title(metric, pad=12)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
    axes[0].set_xlim(-0.15, 1.85)
    axes[1].set_xlim(-0.08, 1.08)
    fig.suptitle("天气与距离带来的指标提升及95%置信区间", fontsize=15, y=1.02)
    fig.text(
        0.5,
        0.0,
        "点为绝对提升，横线为用户级配对 Bootstrap 95%置信区间；区间不跨0表示提升显著",
        ha="center",
        color=COLORS["muted"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), w_pad=2.8)
    return _finish(fig, "context_effect_confidence_intervals.png")


def main() -> int:
    _configure_style()
    paths = [
        plot_four_model_comparison(),
        plot_transition_distance(),
        plot_context_confidence_intervals(),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
