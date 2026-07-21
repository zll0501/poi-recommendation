"""Paired user-cluster bootstrap for two next-POI prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _target_ranks(targets: pd.DataFrame, predictions: pd.DataFrame) -> pd.Series:
    expected = set(predictions["event_id"].astype(int))
    target = targets.loc[
        targets["event_id"].isin(expected), ["event_id", "poi_idx"]
    ].rename(columns={"poi_idx": "target_poi_idx"})
    if set(target["event_id"].astype(int)) != expected:
        raise ValueError("predictions contain event IDs absent from targets")
    ranked = predictions.merge(target, on="event_id", how="left", validate="many_to_one")
    hits = ranked.loc[
        ranked["poi_idx"].eq(ranked["target_poi_idx"]), ["event_id", "rank"]
    ]
    if hits["event_id"].duplicated().any():
        raise ValueError("a prediction contains the target POI more than once")
    return hits.set_index("event_id")["rank"].reindex(sorted(expected))


def _metric_contributions(ranks: pd.Series) -> dict[str, np.ndarray]:
    values = ranks.to_numpy(dtype="float64", na_value=np.nan)
    return {
        "HitRate@5": np.where(np.isfinite(values) & (values <= 5), 1.0, 0.0),
        "HitRate@10": np.where(np.isfinite(values) & (values <= 10), 1.0, 0.0),
        "NDCG@5": np.where(
            np.isfinite(values) & (values <= 5), 1.0 / np.log2(values + 1), 0.0
        ),
        "NDCG@10": np.where(
            np.isfinite(values) & (values <= 10), 1.0 / np.log2(values + 1), 0.0
        ),
        "MRR@10": np.where(
            np.isfinite(values) & (values <= 10), 1.0 / values, 0.0
        ),
    }


def paired_user_bootstrap(
    targets: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    weather_predictions: pd.DataFrame,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    baseline_events = set(baseline_predictions["event_id"].astype(int))
    weather_events = set(weather_predictions["event_id"].astype(int))
    if baseline_events != weather_events:
        raise ValueError("baseline and weather predictions must cover identical events")

    event_ids = sorted(baseline_events)
    event_frame = targets.loc[
        targets["event_id"].isin(event_ids), ["event_id", "user_idx"]
    ].drop_duplicates("event_id")
    if len(event_frame) != len(event_ids):
        raise ValueError("targets do not uniquely identify every prediction event")
    event_frame = event_frame.set_index("event_id").loc[event_ids]
    baseline = _metric_contributions(
        _target_ranks(targets, baseline_predictions).loc[event_ids]
    )
    weather = _metric_contributions(
        _target_ranks(targets, weather_predictions).loc[event_ids]
    )

    user_codes, users = pd.factorize(event_frame["user_idx"], sort=True)
    user_count = len(users)
    counts = np.bincount(user_codes, minlength=user_count).astype("float64")
    rng = np.random.default_rng(seed)
    result: dict[str, object] = {
        "events": len(event_ids),
        "users": user_count,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "metrics": {},
    }
    for name in baseline:
        differences = weather[name] - baseline[name]
        user_sums = np.bincount(
            user_codes, weights=differences, minlength=user_count
        )
        samples = np.empty(bootstrap_samples, dtype="float64")
        for index in range(bootstrap_samples):
            selected = rng.integers(0, user_count, size=user_count)
            samples[index] = user_sums[selected].sum() / counts[selected].sum()
        observed = float(differences.mean())
        lower, upper = np.quantile(samples, [0.025, 0.975])
        non_positive = (np.count_nonzero(samples <= 0) + 1) / (
            bootstrap_samples + 1
        )
        non_negative = (np.count_nonzero(samples >= 0) + 1) / (
            bootstrap_samples + 1
        )
        p_value = min(1.0, 2.0 * min(non_positive, non_negative))
        baseline_mean = float(baseline[name].mean())
        result["metrics"][name] = {
            "baseline": baseline_mean,
            "weather": float(weather[name].mean()),
            "absolute_delta": observed,
            "relative_delta_percent": (
                observed / baseline_mean * 100.0 if baseline_mean else None
            ),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "two_sided_bootstrap_p": float(p_value),
            "ci_excludes_zero": bool(lower > 0 or upper < 0),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="data/processed/test_encoded.csv")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()

    targets = pd.read_csv(_project_path(args.targets))
    baseline = pd.read_csv(_project_path(args.baseline))
    weather = pd.read_csv(_project_path(args.weather))
    result = paired_user_bootstrap(
        targets,
        baseline,
        weather,
        bootstrap_samples=args.samples,
        seed=args.seed,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = _project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
