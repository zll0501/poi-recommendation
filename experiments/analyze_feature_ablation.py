"""Compare saved SASRec ablation metrics without loading or training models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("HitRate@10", "NDCG@10", "MRR@10")
DEFAULT_RESULTS = {
    "SASRec": "results/metrics/sasrec/final_metrics.json",
    "SASRec+Time": "results/metrics/sasrec_time/final_metrics.json",
    "SASRec+Category": "results/metrics/sasrec_category/final_metrics.json",
    "SASRec+Time+Category": (
        "results/metrics/sasrec_time_category/final_metrics.json"
    ),
}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"metrics file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    missing = [metric for metric in METRICS if metric not in payload]
    if missing:
        raise ValueError(f"{path} is missing metrics: {missing}")
    values = {metric: float(payload[metric]) for metric in METRICS}
    if not all(0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError(f"ranking metrics must be within [0, 1]: {path}")
    return values


def _gain(
    model_metrics: dict[str, float], baseline_metrics: dict[str, float]
) -> dict[str, dict[str, float]]:
    absolute = {
        metric: model_metrics[metric] - baseline_metrics[metric]
        for metric in METRICS
    }
    relative_percent = {
        metric: (
            absolute[metric] / baseline_metrics[metric] * 100.0
            if baseline_metrics[metric] != 0
            else 0.0
        )
        for metric in METRICS
    }
    return {
        "absolute": absolute,
        "relative_percent": relative_percent,
    }


def _plot_feature_gains(gains: dict[str, dict[str, Any]], path: Path) -> None:
    variant_labels = ["+ Time", "+ Category", "+ Time + Category"]
    keys = ["time_gain", "category_gain", "joint_gain"]
    model_labels = ["Base SASRec", *variant_labels]
    x_models = np.arange(len(model_labels))
    x_gains = np.arange(len(variant_labels))
    width = 0.24
    colors = ["#4472C4", "#ED7D31", "#70AD47"]

    fig, (absolute_axis, gain_axis) = plt.subplots(1, 2, figsize=(16, 6))
    for metric_index, (metric, color) in enumerate(zip(METRICS, colors)):
        baseline = gains["baseline"][metric]
        absolute_values = [
            baseline,
            *[
                baseline + gains[key]["absolute"][metric]
                for key in keys
            ],
        ]
        absolute_offsets = x_models + (metric_index - 1) * width
        absolute_bars = absolute_axis.bar(
            absolute_offsets,
            absolute_values,
            width=width,
            label=metric,
            color=color,
        )
        for bar, value in zip(absolute_bars, absolute_values):
            absolute_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.002,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

        gain_values = [gains[key]["absolute"][metric] for key in keys]
        gain_offsets = x_gains + (metric_index - 1) * width
        gain_bars = gain_axis.bar(
            gain_offsets,
            gain_values,
            width=width,
            label=metric,
            color=color,
        )
        for bar, value in zip(gain_bars, gain_values):
            gain_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.0003 if value >= 0 else -0.0003),
                f"{value:+.4f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
                rotation=90,
            )

    absolute_axis.set_xticks(x_models, model_labels)
    absolute_axis.set_ylabel("Metric value")
    absolute_axis.set_title("Absolute performance comparison")
    absolute_axis.set_ylim(0.25, 0.58)
    absolute_axis.legend()
    absolute_axis.grid(axis="y", alpha=0.25)

    gain_axis.axhline(0.0, color="black", linewidth=0.8)
    gain_axis.set_xticks(x_gains, variant_labels)
    gain_axis.set_ylabel("Absolute gain over base SASRec")
    gain_axis.set_title("Feature gain relative to base SASRec")
    gain_axis.legend()
    gain_axis.grid(axis="y", alpha=0.25)
    fig.suptitle("SASRec multi-source feature comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_feature_ablation(
    *,
    baseline_path: str | Path = DEFAULT_RESULTS["SASRec"],
    time_path: str | Path = DEFAULT_RESULTS["SASRec+Time"],
    category_path: str | Path = DEFAULT_RESULTS["SASRec+Category"],
    joint_path: str | Path = DEFAULT_RESULTS["SASRec+Time+Category"],
    output_directory: str | Path = "results/ablation_analysis",
) -> dict[str, Any]:
    paths = {
        "SASRec": _project_path(baseline_path),
        "SASRec+Time": _project_path(time_path),
        "SASRec+Category": _project_path(category_path),
        "SASRec+Time+Category": _project_path(joint_path),
    }
    results = {name: _read_metrics(path) for name, path in paths.items()}
    table = pd.DataFrame(
        [{"model": name, **results[name]} for name in DEFAULT_RESULTS]
    )

    baseline = results["SASRec"]
    gains: dict[str, Any] = {
        "definitions": {
            "absolute_gain": "variant metric - SASRec metric",
            "relative_percent": "absolute gain / SASRec metric * 100",
        },
        "feature_semantics": {
            "trajectory": "historical POI sequence",
            "time": "hour aligned with each historical POI",
            "category": "POI category aligned with each historical POI",
        },
        "source_files": {name: str(path) for name, path in paths.items()},
        "baseline": baseline,
        "time_gain": _gain(results["SASRec+Time"], baseline),
        "category_gain": _gain(results["SASRec+Category"], baseline),
        "joint_gain": _gain(results["SASRec+Time+Category"], baseline),
    }

    output_dir = _project_path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "ablation_table.csv", index=False, encoding="utf-8")
    (output_dir / "feature_gain.json").write_text(
        json.dumps(gains, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_feature_gains(gains, output_dir / "feature_gain.png")
    return gains


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze saved SASRec feature-ablation metrics"
    )
    parser.add_argument("--baseline", default=DEFAULT_RESULTS["SASRec"])
    parser.add_argument("--time", default=DEFAULT_RESULTS["SASRec+Time"])
    parser.add_argument("--category", default=DEFAULT_RESULTS["SASRec+Category"])
    parser.add_argument("--joint", default=DEFAULT_RESULTS["SASRec+Time+Category"])
    parser.add_argument("--output-dir", default="results/ablation_analysis")
    args = parser.parse_args()
    gains = analyze_feature_ablation(
        baseline_path=args.baseline,
        time_path=args.time,
        category_path=args.category,
        joint_path=args.joint,
        output_directory=args.output_dir,
    )
    print(json.dumps(gains, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
