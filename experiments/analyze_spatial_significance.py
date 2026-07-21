"""Paired user-cluster bootstrap for fixed spatial reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.analyze_weather_significance import paired_user_bootstrap


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def spatial_bootstrap(
    targets: pd.DataFrame,
    unreranked: pd.DataFrame,
    distance: pd.DataFrame,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 42,
) -> dict[str, object]:
    result = paired_user_bootstrap(
        targets,
        unreranked,
        distance,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    result["comparison"] = "distance_minus_unreranked"
    for metrics in result["metrics"].values():
        metrics["unreranked"] = metrics.pop("baseline")
        metrics["distance"] = metrics.pop("weather")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="data/processed/test_encoded.csv")
    parser.add_argument("--unreranked", required=True)
    parser.add_argument("--distance", required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = spatial_bootstrap(
        pd.read_csv(project_path(args.targets)),
        pd.read_csv(project_path(args.unreranked)),
        pd.read_csv(project_path(args.distance)),
        bootstrap_samples=args.samples,
        seed=args.seed,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
