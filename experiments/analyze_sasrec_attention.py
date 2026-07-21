"""Extract interpretable SASRec attention from an existing checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.datasets import POIDataBundle, load_data_bundle
from src.layers.attention import AttentionExtractor, LastQueryAttention
from src.models.sasrec import SASRec
from src.sasrec_data import SASRecCollator, SASRecDataset, SASRecSample
from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_COLUMNS = [
    "event_id",
    "user_id",
    "user_idx",
    "target_poi_idx",
    "target_poi_id",
    "query_hour",
    "query_weekday",
    "query_time_slot",
    "layer",
    "head",
    "query_index",
    "history_position",
    "history_poi_idx",
    "history_poi_id",
    "history_category_idx",
    "history_hour",
    "attention_weight",
    "is_top_attention",
]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"SASRec checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain model_state_dict")
    return checkpoint


def _build_model(
    config: dict[str, Any], data: POIDataBundle, device: torch.device
) -> SASRec:
    model_config = config["model"]
    use_history_time = bool(
        model_config.get("use_history_time", model_config.get("use_time", False))
    )
    model = SASRec(
        num_pois=data.vocabulary_size("poi_id"),
        num_categories=data.vocabulary_size("category_id"),
        max_seq_len=int(model_config["max_seq_len"]),
        hidden_size=int(model_config["hidden_size"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        pad_id=data.pad_id,
        num_time_tokens=int(model_config.get("num_time_tokens", 25)),
        num_query_weekday_tokens=int(
            model_config.get("num_query_weekday_tokens", 8)
        ),
        num_query_time_slot_tokens=int(
            model_config.get("num_query_time_slot_tokens", 5)
        ),
        use_history_time=use_history_time,
        use_query_time=bool(model_config.get("use_query_time", False)),
        use_category=bool(model_config.get("use_category", False)),
    )
    return model.to(device)


def _select_distinct_users(
    dataset: SASRecDataset, count: int, seed: int
) -> list[SASRecSample]:
    if count < 1:
        raise ValueError("num_samples must be positive")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected: list[SASRecSample] = []
    seen_users: set[int] = set()
    for index in indices:
        sample = dataset[index]
        if sample.user_idx in seen_users:
            continue
        selected.append(sample)
        seen_users.add(sample.user_idx)
        if len(selected) == count:
            break
    if not selected:
        raise ValueError("test dataset contains no evaluable SASRec samples")
    return selected


def _move_inputs(inputs: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in inputs.items()}


def _inverse_vocabulary(data: POIDataBundle, column: str) -> dict[int, str]:
    vocabulary = data.mappings["vocabularies"][column]
    return {
        int(encoded): str(raw)
        for raw, encoded in vocabulary.get("token_to_id", {}).items()
    }


def _sample_records(
    *,
    samples: list[SASRecSample],
    data: POIDataBundle,
    attention: LastQueryAttention,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows = data.test.set_index("event_id", drop=False)
    poi_ids = _inverse_vocabulary(data, "poi_id")
    json_samples: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for batch_index, sample in enumerate(samples):
        event_row = event_rows.loc[sample.event_id]
        layers: dict[str, dict[str, list[float]]] = {}
        top_attention: dict[str, dict[str, dict[str, Any]]] = {}
        for layer_index, heads in attention.items():
            layer_weights: dict[str, list[float]] = {}
            layer_top: dict[str, dict[str, Any]] = {}
            for head_index, batch_weights in heads.items():
                weights = batch_weights[batch_index]
                values = [float(value) for value in weights.tolist()]
                top_position = int(weights.argmax().item())
                layer_weights[str(head_index)] = values
                layer_top[str(head_index)] = {
                    "history_position": top_position,
                    "history_poi_idx": int(sample.poi_sequence[top_position]),
                    "history_poi_id": poi_ids.get(
                        int(sample.poi_sequence[top_position]), "<UNK>"
                    ),
                    "attention_weight": values[top_position],
                }
                for history_position, weight in enumerate(values):
                    csv_rows.append(
                        {
                            "event_id": sample.event_id,
                            "user_id": str(event_row.user_id),
                            "user_idx": sample.user_idx,
                            "target_poi_idx": sample.target_poi_idx,
                            "target_poi_id": poi_ids.get(
                                sample.target_poi_idx, "<UNK>"
                            ),
                            "query_hour": sample.query_hour - 1,
                            "query_weekday": sample.query_weekday - 1,
                            "query_time_slot": sample.query_time_slot,
                            "layer": layer_index,
                            "head": head_index,
                            "query_index": len(sample.poi_sequence) - 1,
                            "history_position": history_position,
                            "history_poi_idx": sample.poi_sequence[history_position],
                            "history_poi_id": poi_ids.get(
                                sample.poi_sequence[history_position], "<UNK>"
                            ),
                            "history_category_idx": sample.category_sequence[
                                history_position
                            ],
                            "history_hour": sample.time_sequence[history_position] - 1,
                            "attention_weight": weight,
                            "is_top_attention": history_position == top_position,
                        }
                    )
            layers[str(layer_index)] = layer_weights
            top_attention[str(layer_index)] = layer_top

        json_samples.append(
            {
                "event_id": sample.event_id,
                "user_id": str(event_row.user_id),
                "user_idx": sample.user_idx,
                "history_poi_indices": list(sample.poi_sequence),
                "history_poi_ids": [
                    poi_ids.get(poi_idx, "<UNK>") for poi_idx in sample.poi_sequence
                ],
                "history_category_indices": list(sample.category_sequence),
                "history_hours": [value - 1 for value in sample.time_sequence],
                "target_poi_idx": sample.target_poi_idx,
                "target_poi_id": poi_ids.get(sample.target_poi_idx, "<UNK>"),
                "query_time": {
                    "hour": sample.query_hour - 1,
                    "weekday": sample.query_weekday - 1,
                    "time_slot_idx": sample.query_time_slot,
                    "time_slot": str(event_row.time_slot),
                },
                "last_query_index": len(sample.poi_sequence) - 1,
                "attention": layers,
                "top_attention": top_attention,
            }
        )
    return json_samples, csv_rows


def analyze(
    config_path: str | Path,
    *,
    num_samples: int = 5,
    seed: int = 42,
    device_name: str = "auto",
    output_root: str | Path = "results/attention_analysis",
) -> dict[str, Any]:
    config = load_yaml(_project_path(config_path))
    data = load_data_bundle(_project_path(config["data_config"]))
    model_config = config["model"]
    experiment_name = str(model_config["name"])
    device = _device(device_name)
    checkpoint_path = _project_path(config["output"]["checkpoint"])
    checkpoint = _load_checkpoint(checkpoint_path, device)

    model = _build_model(config, data, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    max_seq_len = int(model_config["max_seq_len"])
    use_history_time = bool(
        model_config.get("use_history_time", model_config.get("use_time", False))
    )
    use_query_time = bool(model_config.get("use_query_time", False))
    use_category = bool(model_config.get("use_category", False))
    dataset = SASRecDataset(data, "test", max_seq_len)
    samples = _select_distinct_users(dataset, num_samples, seed)
    collator = SASRecCollator(
        max_seq_len,
        pad_id=data.pad_id,
        use_time=use_history_time,
        use_category=use_category,
        use_query_time=use_query_time,
    )
    batch = collator(samples)
    inputs = _move_inputs(batch["inputs"], device)

    with torch.no_grad():
        logits_without_hooks = model(**inputs)
    extractor = AttentionExtractor().register(model)
    try:
        with torch.no_grad():
            analysis_logits = model(**inputs)
        last_attention = extractor.get_last_query_attention(
            inputs["attention_mask"]
        )
    finally:
        extractor.remove()
    # The hooked pass is used only for attention.  Removing hooks must restore
    # the exact prediction path and leave the trained model unchanged.
    with torch.no_grad():
        logits_after_analysis = model(**inputs)
    torch.testing.assert_close(
        logits_after_analysis,
        logits_without_hooks,
        rtol=0.0,
        atol=0.0,
    )
    analysis_logit_max_abs_difference = float(
        (analysis_logits - logits_without_hooks).abs().max().item()
    )

    json_samples, csv_rows = _sample_records(
        samples=samples,
        data=data,
        attention=last_attention,
    )
    output_directory = _project_path(output_root) / experiment_name
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "attention_weights.json"
    csv_path = output_directory / "attention_examples.csv"
    payload = {
        "experiment": experiment_name,
        "checkpoint": str(checkpoint_path),
        "partition": "test",
        "seed": seed,
        "num_samples": len(samples),
        "logits_preserved": True,
        "attention_pass_max_abs_logit_difference": analysis_logit_max_abs_difference,
        "samples": json_samples,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract last-query attention from a trained SASRec checkpoint"
    )
    parser.add_argument(
        "--config",
        default="configs/sasrec_query_time_category.yaml",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="results/attention_analysis",
    )
    args = parser.parse_args()
    payload = analyze(
        args.config,
        num_samples=args.num_samples,
        seed=args.seed,
        device_name=args.device,
        output_root=args.output_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
