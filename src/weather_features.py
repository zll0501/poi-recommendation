"""Leakage-safe weather feature preparation and model-facing batch interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from src.datasets import load_data_bundle
from src.utils.config import load_yaml
from src.weather_data import attach_weather, load_weather_sidecar


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _to_boolean(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        raise ValueError(f"{column} contains invalid boolean values")
    return normalized.isin({"true", "1"})


def _transform_numeric(series: pd.Series, transform: str, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{column} contains invalid numeric values")
    if transform == "identity":
        return values
    if transform == "log1p":
        if values.lt(0).any():
            raise ValueError(f"{column} cannot be negative before log1p")
        return np.log1p(values)
    raise ValueError(f"unsupported weather numeric transform: {transform}")


def fit_weather_statistics(
    train_weather: pd.DataFrame,
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    """Fit category vocabulary and numeric scaling on training events only."""
    if train_weather.empty:
        raise ValueError("training weather cannot be empty")
    category_column = str(preprocessing["categorical_column"])
    if train_weather[category_column].isna().any():
        raise ValueError("training weather groups cannot be missing")
    unknown_id = int(preprocessing["unknown_weather_group_id"])
    if unknown_id != 0:
        raise ValueError("unknown_weather_group_id must be 0")
    groups = sorted(train_weather[category_column].astype(str).unique())
    mapping = {group: index for index, group in enumerate(groups, start=1)}

    numeric_statistics: dict[str, dict[str, Any]] = {}
    numeric_output_columns: list[str] = []
    for feature in preprocessing["numeric_features"]:
        input_column = str(feature["input"])
        output_column = str(feature["output"])
        transform = str(feature["transform"])
        values = _transform_numeric(train_weather[input_column], transform, input_column)
        standard_deviation = float(values.std(ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation <= 0:
            standard_deviation = 1.0
        numeric_statistics[input_column] = {
            "output": output_column,
            "transform": transform,
            "mean": float(values.mean()),
            "std": standard_deviation,
        }
        numeric_output_columns.append(output_column)

    binary_features = [str(column) for column in preprocessing["binary_features"]]
    for column in binary_features:
        _to_boolean(train_weather[column], column)
    return {
        "fit_on": "train",
        "unknown_weather_group_id": unknown_id,
        "weather_group_mapping": mapping,
        "weather_group_vocab_size": len(mapping) + 1,
        "numeric_statistics": numeric_statistics,
        "numeric_output_columns": numeric_output_columns,
        "binary_output_columns": binary_features,
        "model_numeric_columns": numeric_output_columns + binary_features,
    }


def transform_weather_features(
    weather: pd.DataFrame,
    statistics: dict[str, Any],
    *,
    partition: str,
) -> pd.DataFrame:
    """Transform any split using immutable training statistics."""
    if "event_id" not in weather or weather["event_id"].duplicated().any():
        raise ValueError("weather events must contain unique event_id values")
    mapping = {
        str(group): int(index)
        for group, index in statistics["weather_group_mapping"].items()
    }
    result = pd.DataFrame({
        "event_id": pd.to_numeric(weather["event_id"], errors="raise").astype("int64"),
        "partition": partition,
        "weather_group_idx": weather["weather_group"]
        .astype(str)
        .map(mapping)
        .fillna(int(statistics["unknown_weather_group_id"]))
        .astype("int64"),
    })
    for input_column, values in statistics["numeric_statistics"].items():
        transformed = _transform_numeric(
            weather[input_column], str(values["transform"]), input_column
        )
        result[str(values["output"])] = (
            (transformed - float(values["mean"])) / float(values["std"])
        ).astype("float64")
    for column in statistics["binary_output_columns"]:
        result[str(column)] = _to_boolean(weather[column], str(column)).astype("int8")
    if not np.isfinite(result[statistics["model_numeric_columns"]].to_numpy()).all():
        raise ValueError("transformed weather features contain non-finite values")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_weather_model_features(
    config_path: str | Path = "configs/weather_model.yaml",
) -> dict[str, Any]:
    """Create an event-keyed model table without modifying canonical splits."""
    config = load_yaml(config_path)
    preprocessing = config["preprocessing"]
    if preprocessing.get("fit_on") != "train":
        raise ValueError("weather preprocessing must be fitted on train only")
    data = load_data_bundle(config["data_config"])
    sidecar = load_weather_sidecar(config["weather_config"])
    raw_columns = [
        str(preprocessing["categorical_column"]),
        *[str(item["input"]) for item in preprocessing["numeric_features"]],
        *[str(column) for column in preprocessing["binary_features"]],
    ]
    raw_partitions = {
        "train": attach_weather(data.train, sidecar, raw_columns),
        "validation": attach_weather(data.validation, sidecar, raw_columns),
        "test": attach_weather(data.test, sidecar, raw_columns),
    }
    statistics = fit_weather_statistics(raw_partitions["train"], preprocessing)
    transformed = [
        transform_weather_features(frame, statistics, partition=name)
        for name, frame in raw_partitions.items()
    ]
    features = pd.concat(transformed, ignore_index=True).sort_values(
        "event_id", kind="stable"
    )
    if features["event_id"].duplicated().any():
        raise ValueError("event IDs overlap across weather partitions")

    output = config["output"]
    output_dir = _resolve(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / output["features_file"]
    statistics_path = output_dir / output["statistics_file"]
    temporary_features = features_path.with_suffix(features_path.suffix + ".tmp")
    temporary_statistics = statistics_path.with_suffix(statistics_path.suffix + ".tmp")
    features.to_csv(temporary_features, index=False)
    statistics.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": int(len(features)),
        "partition_events": {
            name: int(len(frame)) for name, frame in raw_partitions.items()
        },
        "canonical_partitions_modified": False,
        "features_file": features_path.name,
        "features_file_sha256": _file_sha256(temporary_features),
    })
    temporary_statistics.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_features.replace(features_path)
    temporary_statistics.replace(statistics_path)
    return statistics


@dataclass(frozen=True)
class WeatherBatch:
    group_idx: torch.LongTensor
    numeric: torch.FloatTensor


class WeatherFeatureStore:
    """Immutable event-to-weather tensor adapter used by any sequence model."""

    def __init__(self, frame: pd.DataFrame, statistics: dict[str, Any]) -> None:
        required = {
            "event_id",
            "weather_group_idx",
            *statistics["model_numeric_columns"],
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"weather model features are missing columns: {missing}")
        if frame["event_id"].duplicated().any():
            raise ValueError("weather model feature event IDs must be unique")
        self.frame = frame.set_index("event_id", drop=False)
        self.statistics = statistics
        self.numeric_columns = tuple(statistics["model_numeric_columns"])

    @property
    def weather_group_vocab_size(self) -> int:
        return int(self.statistics["weather_group_vocab_size"])

    @property
    def numeric_dim(self) -> int:
        return len(self.numeric_columns)

    def get_by_event_ids(
        self,
        event_ids: Sequence[int],
        *,
        device: torch.device | str | None = None,
    ) -> WeatherBatch:
        ids = [int(event_id) for event_id in event_ids]
        missing = [event_id for event_id in ids if event_id not in self.frame.index]
        if missing:
            raise KeyError(f"weather features missing for {len(missing)} event IDs")
        selected = self.frame.loc[ids]
        group_idx = torch.as_tensor(
            selected["weather_group_idx"].to_numpy(copy=True),
            dtype=torch.long,
            device=device,
        )
        numeric = torch.as_tensor(
            selected[list(self.numeric_columns)].to_numpy(dtype="float32", copy=True),
            dtype=torch.float32,
            device=device,
        )
        return WeatherBatch(group_idx=group_idx, numeric=numeric)


def load_weather_feature_store(
    config_path: str | Path = "configs/weather_model.yaml",
) -> WeatherFeatureStore:
    config = load_yaml(config_path)
    output = config["output"]
    output_dir = _resolve(output["directory"])
    features_path = output_dir / output["features_file"]
    statistics_path = output_dir / output["statistics_file"]
    missing = [str(path) for path in (features_path, statistics_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "weather model features are missing; run python -m src.weather_features: "
            + ", ".join(missing)
        )
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if _file_sha256(features_path) != statistics["features_file_sha256"]:
        raise ValueError("weather model feature checksum does not match statistics")
    return WeatherFeatureStore(pd.read_csv(features_path), statistics)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/weather_model.yaml")
    args = parser.parse_args(argv)
    statistics = prepare_weather_model_features(args.config)
    print(f"Weather model events: {statistics['events']:,}")
    print(f"Weather group vocabulary: {statistics['weather_group_vocab_size']}")
    print(f"Numeric features: {', '.join(statistics['model_numeric_columns'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
