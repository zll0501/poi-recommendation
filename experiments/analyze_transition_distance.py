"""Analyze leakage-safe POI transition distances on the training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.spatial_reranker import (
    build_training_transitions,
    summarize_training_transitions,
)
from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run(config_path: str | Path) -> tuple[dict[str, object], dict[str, Path]]:
    config = load_yaml(config_path)
    analysis = config["analysis"]
    train_path = project_path(analysis["train_path"])
    if not train_path.exists():
        raise FileNotFoundError(f"encoded training data not found: {train_path}")

    train = pd.read_csv(train_path)
    transitions = build_training_transitions(train)
    report, by_time_gap = summarize_training_transitions(
        transitions, analysis["distance_thresholds_km"]
    )
    report.update(
        {
            "source": str(train_path.relative_to(PROJECT_ROOT)),
            "definition": "consecutive events within each user in the training split",
        }
    )

    output = config["output"]
    report_path = project_path(output["analysis_json"])
    by_time_path = project_path(output["by_time_gap_csv"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    by_time_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    by_time_gap.to_csv(by_time_path, index=False, encoding="utf-8")
    return report, {"analysis_json": report_path, "by_time_gap_csv": by_time_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze training POI transition distances")
    parser.add_argument("--config", default="configs/spatial_reranker.yaml")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, paths = run(args.config)
    quantiles = report["distance_quantiles"]
    locality = report["distance_locality"]
    print(f"training transitions: {report['transition_count']:,}")
    print(
        f"median={quantiles['p50_km']:.3f} km, "
        f"p90={quantiles['p90_km']:.3f} km, "
        f"within_5km={locality['within_5km_ratio']:.2%}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
