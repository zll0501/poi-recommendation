"""Select and evaluate a soft-distance reranker without retraining SASRec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import POIDataBundle, load_data_bundle
from src.evaluator import evaluate_next_poi, recommendations_to_frame
from src.layers.weather import WeatherCategoryScorer, WeatherEncoder
from src.models.sasrec import SASRec
from src.models.weather_category_sasrec import WeatherCategorySASRec
from src.sasrec_data import SASRecCollator, SASRecDataset
from src.spatial_reranker import build_query_contexts, haversine_km, rerank_by_distance
from src.utils.config import load_yaml
from src.weather_features import load_weather_feature_store
from src.weather_sasrec_data import WeatherSASRecCollator, build_poi_category_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = ("Coverage", "HitRate@5", "HitRate@10", "NDCG@5", "NDCG@10", "MRR@10")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def device_from_config(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_sasrec(model_config: Mapping[str, Any], data: POIDataBundle) -> SASRec:
    return SASRec(
        num_pois=data.vocabulary_size("poi_id"),
        num_categories=data.vocabulary_size("category_id"),
        max_seq_len=int(model_config["max_seq_len"]),
        hidden_size=int(model_config["hidden_size"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        pad_id=data.pad_id,
        num_time_tokens=int(model_config.get("num_time_tokens", 25)),
        num_query_weekday_tokens=int(model_config.get("num_query_weekday_tokens", 8)),
        num_query_time_slot_tokens=int(model_config.get("num_query_time_slot_tokens", 5)),
        use_history_time=bool(model_config.get("use_history_time", False)),
        use_query_time=bool(model_config.get("use_query_time", False)),
        use_category=bool(model_config.get("use_category", False)),
    )


def build_model_and_collator(
    runner: str,
    experiment_config_path: Path,
    data: POIDataBundle,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    config = load_yaml(experiment_config_path)
    model_config = config["model"]
    base_model = build_sasrec(model_config, data)
    base_collator = SASRecCollator(
        int(model_config["max_seq_len"]),
        pad_id=data.pad_id,
        use_time=bool(model_config.get("use_history_time", False)),
        use_category=bool(model_config.get("use_category", False)),
        use_query_time=bool(model_config.get("use_query_time", False)),
    )
    if runner == "baseline":
        return base_model, base_collator, config
    if runner != "weather":
        raise ValueError(f"unknown runner: {runner}")

    weather_config_path = project_path(config["weather_model_config"])
    weather_config = load_yaml(weather_config_path)
    weather_store = load_weather_feature_store(weather_config_path)
    encoder = weather_config["encoder"]
    hidden_size = int(model_config["hidden_size"])
    if int(encoder["output_dim"]) != hidden_size:
        raise ValueError("weather output dimension must equal SASRec hidden size")
    model = WeatherCategorySASRec(
        base_model,
        WeatherEncoder(
            num_weather_groups=weather_store.weather_group_vocab_size,
            numeric_dim=weather_store.numeric_dim,
            output_dim=hidden_size,
            weather_embedding_dim=int(encoder["weather_embedding_dim"]),
            numeric_hidden_dim=int(encoder["numeric_hidden_dim"]),
            dropout=float(encoder["dropout"]),
        ),
        WeatherCategoryScorer(initial_weight=float(encoder["initial_weather_weight"])),
        build_poi_category_index(
            data.poi_metadata,
            num_pois=data.vocabulary_size("poi_id"),
            pad_id=data.pad_id,
            unknown_id=data.unknown_id,
        ),
    )
    return model, WeatherSASRecCollator(base_collator, weather_store), config


def load_checkpoint_model(
    runner: str,
    experiment_config_path: Path,
    checkpoint_path: Path,
    data: POIDataBundle,
    device: torch.device,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    model, collator, config = build_model_and_collator(runner, experiment_config_path, data)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, collator, config


def predict_shortlists(
    model: nn.Module,
    loader: DataLoader,
    candidate_poi_ids: Iterable[int],
    device: torch.device,
    top_m: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    candidates = torch.tensor(tuple(candidate_poi_ids), dtype=torch.long, device=device)
    if top_m < 10 or top_m > len(candidates):
        raise ValueError("candidate_top_m must be between 10 and candidate count")
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch["inputs"].items()}
            logits = model(**inputs).index_select(1, candidates)
            values, positions = logits.topk(top_m, dim=1)
            ids = candidates[positions].cpu().numpy()
            scores = values.cpu().numpy()
            for event_id, poi_ids, poi_scores in zip(batch["event_id"].tolist(), ids, scores):
                output[int(event_id)] = (poi_ids, poi_scores)
    return output


def partition_query_contexts(data: POIDataBundle, partition: str) -> pd.DataFrame:
    if partition == "validation":
        prior, targets = data.train, data.validation
    elif partition == "test":
        prior = pd.concat([data.train, data.validation], ignore_index=True)
        targets = data.test
    else:
        raise ValueError("spatial selection supports validation or test")
    return build_query_contexts(prior, targets)


def rerank_partition(
    shortlists: Mapping[int, tuple[np.ndarray, np.ndarray]],
    contexts: pd.DataFrame,
    poi_metadata: pd.DataFrame,
    penalty_weight: float,
    output_k: int,
) -> dict[int, list[int]]:
    coordinate_by_poi = poi_metadata.set_index("poi_idx")[["latitude", "longitude"]]
    context_by_event = contexts.set_index("event_id")
    recommendations: dict[int, list[int]] = {}
    for event_id, (poi_ids, scores) in shortlists.items():
        if event_id not in context_by_event.index:
            raise ValueError(f"missing query location for event {event_id}")
        query = context_by_event.loc[event_id]
        candidate_coordinates = coordinate_by_poi.loc[poi_ids]
        distances = haversine_km(
            np.full(len(poi_ids), float(query["last_latitude"])),
            np.full(len(poi_ids), float(query["last_longitude"])),
            candidate_coordinates["latitude"].to_numpy(),
            candidate_coordinates["longitude"].to_numpy(),
        )
        recommendations[event_id] = rerank_by_distance(
            poi_ids, scores, distances, penalty_weight, top_k=output_k
        )
    return recommendations


def metric_subset(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(metrics[name]) for name in METRIC_NAMES}


def run(config_path: str | Path, runner: str) -> dict[str, Any]:
    spatial = load_yaml(config_path)
    model_entry = spatial["models"][runner]
    experiment_config_path = project_path(model_entry["config"])
    checkpoint_path = project_path(model_entry["checkpoint"])
    experiment_config = load_yaml(experiment_config_path)
    data = load_data_bundle(project_path(experiment_config["data_config"]))
    training = experiment_config["training"]
    evaluation = experiment_config["evaluation"]
    device = device_from_config(str(training.get("device", "auto")))
    model, collator, _ = load_checkpoint_model(
        runner, experiment_config_path, checkpoint_path, data, device
    )
    batch_size = int(training["batch_size"])
    max_seq_len = int(experiment_config["model"]["max_seq_len"])
    loaders = {
        name: DataLoader(
            SASRecDataset(data, name, max_seq_len),
            batch_size=batch_size,
            collate_fn=collator,
        )
        for name in ("validation", "test")
    }
    top_m = int(spatial["reranking"]["candidate_top_m"])
    output_k = int(evaluation["output_k"])
    validation_shortlists = predict_shortlists(
        model, loaders["validation"], data.candidate_poi_ids, device, top_m
    )
    validation_contexts = partition_query_contexts(data, "validation")
    validation_rows: list[dict[str, float]] = []
    for penalty_weight in (float(value) for value in spatial["reranking"]["lambda_grid"]):
        recommendations = rerank_partition(
            validation_shortlists,
            validation_contexts,
            data.poi_metadata,
            penalty_weight,
            output_k,
        )
        metrics = evaluate_next_poi(
            data.validation,
            recommendations,
            data.candidate_poi_ids,
            unknown_id=data.unknown_id,
            ks=tuple(int(value) for value in evaluation["top_k"]),
            mrr_k=int(evaluation["mrr_k"]),
        )
        validation_rows.append({"lambda": penalty_weight, **metric_subset(metrics)})

    selection_metric = str(spatial["reranking"]["selection_metric"])
    selected = max(validation_rows, key=lambda row: (row[selection_metric], -row["lambda"]))
    test_shortlists = predict_shortlists(
        model, loaders["test"], data.candidate_poi_ids, device, top_m
    )
    test_contexts = partition_query_contexts(data, "test")
    test_results: dict[str, dict[str, float]] = {}
    test_predictions: dict[str, dict[int, list[int]]] = {}
    for label, penalty_weight in (("unreranked", 0.0), ("distance", selected["lambda"])):
        recommendations = rerank_partition(
            test_shortlists, test_contexts, data.poi_metadata, penalty_weight, output_k
        )
        metrics = evaluate_next_poi(
            data.test,
            recommendations,
            data.candidate_poi_ids,
            unknown_id=data.unknown_id,
            ks=tuple(int(value) for value in evaluation["top_k"]),
            mrr_k=int(evaluation["mrr_k"]),
        )
        test_results[label] = metric_subset(metrics)
        test_predictions[label] = recommendations

    deltas = {
        name: test_results["distance"][name] - test_results["unreranked"][name]
        for name in METRIC_NAMES
    }
    result = {
        "runner": runner,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "selection_partition": "validation",
        "selection_metric": selection_metric,
        "candidate_top_m": top_m,
        "selected_lambda": selected["lambda"],
        "best_validation_metric": selected[selection_metric],
        "validation_grid": validation_rows,
        "test_unreranked": test_results["unreranked"],
        "test_distance": test_results["distance"],
        "test_delta": deltas,
    }
    output_directory = project_path(spatial["output"]["reranking_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / f"{runner}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for label, recommendations in test_predictions.items():
        recommendations_to_frame(recommendations).to_csv(
            output_directory / f"{runner}_{label}_predictions.csv",
            index=False,
            encoding="utf-8",
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spatial_reranker.yaml")
    parser.add_argument("--runner", choices=("baseline", "weather"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.runner), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
