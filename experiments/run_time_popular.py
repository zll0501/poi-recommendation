"""Train and evaluate the time-aware popularity baseline on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from src.datasets import load_data_bundle
from src.evaluator import evaluate_next_poi
from src.models.popular import TimePopular
from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run(config_path: str | Path = "configs/time_popular.yaml") -> dict[str, Any]:
    config = load_yaml(config_path)
    model_config = config["model"]
    if model_config.get("name") != "TimePopular":
        raise ValueError("model.name must be TimePopular")
    if model_config.get("fit_on") != "train":
        raise ValueError("TimePopular must be fitted on train only")
    top_k = int(model_config["top_k"])
    time_column = str(model_config["time_column"])

    evaluation_config = load_yaml(config["evaluation_config"])
    output_k = int(evaluation_config["ranking"]["output_k"])
    if top_k != output_k:
        raise ValueError("model.top_k must equal evaluation ranking.output_k")
    metric_ks = tuple(int(k) for k in evaluation_config["ranking"]["top_k"])
    mrr_k = int(evaluation_config["ranking"]["mrr_k"])

    data = load_data_bundle(config["data_config"])
    candidate_set = set(data.candidate_poi_ids)
    evaluable_test = data.test.loc[
        data.test["user_idx"].ne(data.unknown_id)
        & data.test["poi_idx"].isin(candidate_set)
    ].copy()

    started = perf_counter()
    model = TimePopular(time_column=time_column).fit(data.train)
    fit_seconds = perf_counter() - started
    predictions = model.recommend(evaluable_test, top_k=top_k)
    metrics = evaluate_next_poi(
        targets=data.test,
        predictions=predictions,
        candidate_poi_ids=data.candidate_poi_ids,
        unknown_id=data.unknown_id,
        ks=metric_ks,
        mrr_k=mrr_k,
    )

    per_time_slot: dict[str, Any] = {}
    for slot in sorted(data.test[time_column].astype(str).unique()):
        slot_targets = data.test.loc[data.test[time_column].astype(str).eq(slot)]
        slot_evaluable_ids = set(
            slot_targets.loc[
                slot_targets["user_idx"].ne(data.unknown_id)
                & slot_targets["poi_idx"].isin(candidate_set),
                "event_id",
            ].astype(int)
        )
        if not slot_evaluable_ids:
            continue
        slot_predictions = predictions.loc[
            predictions["event_id"].isin(slot_evaluable_ids)
        ].copy()
        per_time_slot[slot] = {
            "test_events": int(len(slot_targets)),
            "evaluable_test_events": int(len(slot_evaluable_ids)),
            "metrics": evaluate_next_poi(
                targets=slot_targets,
                predictions=slot_predictions,
                candidate_poi_ids=data.candidate_poi_ids,
                unknown_id=data.unknown_id,
                ks=metric_ks,
                mrr_k=mrr_k,
            ),
        }
    total_seconds = perf_counter() - started

    output = {name: resolve_project_path(path) for name, path in config["output"].items()}
    for path in output.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output["predictions"], index=False, encoding="utf-8")
    model.ranking_frame().to_csv(output["ranking"], index=False, encoding="utf-8")
    model.save(output["model"])

    report: dict[str, Any] = {
        "model": "TimePopular",
        "protocol": {
            "fit_on": "train",
            "candidate_scope": "train_pois",
            "exclude_seen_pois": False,
            "query_feature": time_column,
            "fallback": "global_popular",
            "tie_break": "slot_visit_count_desc_then_poi_idx_asc",
            "top_k": top_k,
        },
        "data": {
            "train_events": int(len(data.train)),
            "test_events": int(len(data.test)),
            "evaluable_test_events": int(len(evaluable_test)),
            "candidate_pois": int(len(candidate_set)),
        },
        "metrics": metrics,
        "per_time_slot": per_time_slot,
        "runtime_seconds": {
            "fit": fit_seconds,
            "total_fit_recommend_evaluate": total_seconds,
        },
        "artifacts": {name: str(path) for name, path in output.items()},
    }
    output["metrics"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Time Popular baseline")
    parser.add_argument("--config", default="configs/time_popular.yaml")
    args = parser.parse_args(argv)
    report = run(args.config)
    print(f"Model: {report['model']}")
    for name, value in report["metrics"].items():
        print(f"{name}: {value:.6f}")
    for slot, values in report["per_time_slot"].items():
        metrics = values["metrics"]
        print(
            f"{slot}: events={values['test_events']}, "
            f"HitRate@10={metrics.get('HitRate@10', float('nan')):.6f}, "
            f"NDCG@10={metrics.get('NDCG@10', float('nan')):.6f}"
        )
    runtime = report["runtime_seconds"]
    print(f"Fit seconds: {runtime['fit']:.3f}")
    print(f"Total seconds: {runtime['total_fit_recommend_evaluate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
