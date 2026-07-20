"""Compare Time Popular with weather-conditioned popularity on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from src.datasets import load_data_bundle
from src.evaluator import evaluate_next_poi
from src.models.popular import TimePopular, TimeWeatherPopular
from src.utils.config import load_yaml
from src.weather_data import attach_weather, load_weather_sidecar


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run(config_path: str | Path = "configs/time_weather_popular.yaml") -> dict[str, Any]:
    config = load_yaml(config_path)
    model_config = config["model"]
    if model_config.get("name") != "TimeWeatherPopular":
        raise ValueError("model.name must be TimeWeatherPopular")
    if model_config.get("fit_on") != "train":
        raise ValueError("TimeWeatherPopular must be fitted on train only")
    top_k = int(model_config["top_k"])
    time_column = str(model_config["time_column"])
    weather_column = str(model_config["weather_column"])

    evaluation = load_yaml(config["evaluation_config"])
    if top_k != int(evaluation["ranking"]["output_k"]):
        raise ValueError("model.top_k must equal evaluation ranking.output_k")
    ks = tuple(int(k) for k in evaluation["ranking"]["top_k"])
    mrr_k = int(evaluation["ranking"]["mrr_k"])

    data = load_data_bundle(config["data_config"])
    sidecar = load_weather_sidecar(config["weather_config"])
    train = attach_weather(data.train, sidecar, [weather_column])
    test = attach_weather(data.test, sidecar, [weather_column])
    candidates = set(data.candidate_poi_ids)
    evaluable = test.loc[
        test["user_idx"].ne(data.unknown_id) & test["poi_idx"].isin(candidates)
    ].copy()

    started = perf_counter()
    baseline = TimePopular(time_column).fit(train)
    baseline_predictions = baseline.recommend(evaluable, top_k=top_k)
    baseline_metrics = evaluate_next_poi(
        test, baseline_predictions, data.candidate_poi_ids,
        unknown_id=data.unknown_id, ks=ks, mrr_k=mrr_k,
    )

    model = TimeWeatherPopular(
        time_column=time_column,
        weather_column=weather_column,
        min_bucket_events=int(model_config["min_bucket_events"]),
    ).fit(train)
    predictions = model.recommend(evaluable, top_k=top_k)
    metrics = evaluate_next_poi(
        test, predictions, data.candidate_poi_ids,
        unknown_id=data.unknown_id, ks=ks, mrr_k=mrr_k,
    )
    total_seconds = perf_counter() - started

    per_weather: dict[str, Any] = {}
    for group in sorted(test[weather_column].astype(str).unique()):
        targets = test.loc[test[weather_column].astype(str).eq(group)]
        ids = set(
            targets.loc[
                targets["user_idx"].ne(data.unknown_id)
                & targets["poi_idx"].isin(candidates),
                "event_id",
            ].astype(int)
        )
        if not ids:
            continue
        base_group = evaluate_next_poi(
            targets,
            baseline_predictions.loc[baseline_predictions["event_id"].isin(ids)],
            data.candidate_poi_ids,
            unknown_id=data.unknown_id, ks=ks, mrr_k=mrr_k,
        )
        weather_group_metrics = evaluate_next_poi(
            targets,
            predictions.loc[predictions["event_id"].isin(ids)],
            data.candidate_poi_ids,
            unknown_id=data.unknown_id, ks=ks, mrr_k=mrr_k,
        )
        per_weather[group] = {
            "test_events": int(len(targets)),
            "evaluable_events": int(len(ids)),
            "time_popular": base_group,
            "time_weather_popular": weather_group_metrics,
            "delta": {
                key: weather_group_metrics[key] - base_group[key]
                for key in weather_group_metrics
            },
        }

    output = {name: _resolve(path) for name, path in config["output"].items()}
    for path in output.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output["predictions"], index=False)
    model.ranking_frame().to_csv(output["ranking"], index=False)
    model.save(output["model"])
    report: dict[str, Any] = {
        "model": "TimeWeatherPopular",
        "research_question": "Does weather add signal beyond query-time popularity?",
        "protocol": {
            "fit_on": "train",
            "candidate_scope": "train_pois",
            "query_time_known": True,
            "context": [time_column, weather_column],
            "fallback_order": ["time_weather", "time", "global"],
            "min_bucket_events": model.min_bucket_events,
        },
        "data": {
            "train_events": int(len(train)),
            "test_events": int(len(test)),
            "evaluable_test_events": int(len(evaluable)),
            "candidate_pois": int(len(candidates)),
            "learned_weather_buckets": int(len(model.rankings_ or {})),
        },
        "time_popular_metrics": baseline_metrics,
        "time_weather_popular_metrics": metrics,
        "delta_vs_time_popular": {
            key: metrics[key] - baseline_metrics[key] for key in metrics
        },
        "per_weather_group": per_weather,
        "runtime_seconds": total_seconds,
        "artifacts": {name: str(path) for name, path in output.items()},
    }
    output["metrics"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/time_weather_popular.yaml")
    args = parser.parse_args(argv)
    report = run(args.config)
    print("Time Popular -> Time + Weather Popular")
    for key, value in report["time_weather_popular_metrics"].items():
        baseline = report["time_popular_metrics"][key]
        delta = report["delta_vs_time_popular"][key]
        print(f"{key}: {baseline:.6f} -> {value:.6f} ({delta:+.6f})")
    print(f"Runtime seconds: {report['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
