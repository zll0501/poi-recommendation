"""Match hourly weather to recommendation events through a sidecar table."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEATHER_TIME_COLUMN = "weather_time_utc"
WEATHER_FEATURES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def weather_group(code: int | float | None) -> str:
    """Collapse WMO weather codes into interpretable, stable groups."""
    if code is None or pd.isna(code):
        return "unknown"
    value = int(code)
    if value == 0:
        return "clear"
    if value in {1, 2, 3}:
        return "cloudy"
    if value in {45, 48}:
        return "fog"
    if value in {51, 53, 55, 56, 57}:
        return "drizzle"
    if value in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "rain"
    if value in {71, 73, 75, 77, 85, 86}:
        return "snow"
    if value in {95, 96, 99}:
        return "thunderstorm"
    return "other"


def match_events_to_weather(
    events: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    event_id_column: str = "event_id",
    timestamp_column: str = "utc_time",
    partition: str,
) -> pd.DataFrame:
    """Floor event timestamps to UTC hours and perform a many-to-one join."""
    required_events = {event_id_column, timestamp_column}
    missing_events = sorted(required_events.difference(events.columns))
    if missing_events:
        raise ValueError(f"event data is missing columns: {missing_events}")
    required_weather = {WEATHER_TIME_COLUMN, *WEATHER_FEATURES}
    missing_weather = sorted(required_weather.difference(weather.columns))
    if missing_weather:
        raise ValueError(f"weather data is missing columns: {missing_weather}")

    event_frame = events[[event_id_column, timestamp_column]].copy()
    event_frame[timestamp_column] = pd.to_datetime(
        event_frame[timestamp_column], utc=True, errors="coerce"
    )
    if event_frame[timestamp_column].isna().any():
        raise ValueError(f"{partition} contains invalid event timestamps")
    if event_frame[event_id_column].duplicated().any():
        raise ValueError(f"{partition} contains duplicate event IDs")
    event_frame[WEATHER_TIME_COLUMN] = event_frame[timestamp_column].dt.floor("h")

    weather_frame = weather[[WEATHER_TIME_COLUMN, *WEATHER_FEATURES]].copy()
    weather_frame[WEATHER_TIME_COLUMN] = pd.to_datetime(
        weather_frame[WEATHER_TIME_COLUMN], utc=True, errors="coerce"
    )
    if weather_frame[WEATHER_TIME_COLUMN].isna().any():
        raise ValueError("weather table contains invalid timestamps")
    if weather_frame[WEATHER_TIME_COLUMN].duplicated().any():
        raise ValueError("weather table contains duplicate hourly timestamps")

    result = event_frame.merge(
        weather_frame,
        on=WEATHER_TIME_COLUMN,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result.insert(1, "partition", partition)
    result["weather_missing"] = result[list(WEATHER_FEATURES)].isna().any(axis=1)
    result["weather_group"] = result["weather_code"].map(weather_group)
    result["has_precipitation"] = result["precipitation"].fillna(0).gt(0)
    return result.drop(columns=[timestamp_column])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_weather_sidecar(
    weather_config_path: str | Path = "configs/weather.yaml",
) -> dict[str, Any]:
    """Build one event-keyed weather table without changing canonical splits."""
    weather_config = load_yaml(weather_config_path)
    matching = weather_config["matching"]
    if matching.get("hour_alignment") != "floor":
        raise ValueError("only causal floor-hour alignment is supported")

    data_config = load_yaml(_resolve_project_path(matching["data_config"]))
    data_output = data_config["output"]
    processed_dir = _resolve_project_path(data_output["directory"])
    partition_files = {
        "train": processed_dir / data_output["encoded_train_file"],
        "validation": processed_dir / data_output["encoded_validation_file"],
        "test": processed_dir / data_output["encoded_test_file"],
    }
    weather_output = weather_config["output"]
    weather_path = (
        _resolve_project_path(weather_output["directory"])
        / weather_output["weather_file"]
    )
    required_paths = [weather_path, *partition_files.values()]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("required inputs are missing: " + ", ".join(missing_paths))

    weather = pd.read_csv(weather_path)
    event_id_column = str(matching.get("event_id_column", "event_id"))
    timestamp_column = str(matching.get("timestamp_column", "utc_time"))
    matched_parts: list[pd.DataFrame] = []
    partition_report: dict[str, Any] = {}
    for name, path in partition_files.items():
        events = pd.read_csv(path, usecols=[event_id_column, timestamp_column])
        matched = match_events_to_weather(
            events,
            weather,
            event_id_column=event_id_column,
            timestamp_column=timestamp_column,
            partition=name,
        )
        missing = int(matched["weather_missing"].sum())
        coverage = float(1.0 - missing / len(matched)) if len(matched) else 0.0
        partition_report[name] = {
            "events": int(len(matched)),
            "matched_events": int(len(matched) - missing),
            "missing_events": missing,
            "coverage": coverage,
        }
        matched_parts.append(matched)

    sidecar = pd.concat(matched_parts, ignore_index=True)
    if sidecar[event_id_column].duplicated().any():
        raise ValueError("event IDs overlap across data partitions")
    overall_missing = int(sidecar["weather_missing"].sum())
    overall_coverage = float(1.0 - overall_missing / len(sidecar))
    minimum_coverage = float(matching.get("minimum_coverage", 0.99))
    if overall_coverage < minimum_coverage:
        raise ValueError(
            f"weather coverage {overall_coverage:.2%} is below "
            f"required {minimum_coverage:.2%}"
        )

    sidecar = sidecar.sort_values(event_id_column, kind="stable").reset_index(drop=True)
    output_dir = _resolve_project_path(matching["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / matching["features_file"]
    report_path = output_dir / matching["report_file"]
    temporary_features = features_path.with_suffix(features_path.suffix + ".tmp")
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    sidecar.to_csv(
        temporary_features,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )

    report = {
        "purpose": "optional_event_keyed_weather_sidecar",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "query_time_known": bool(data_config["split"]["query_time_known"]),
            "timestamp_column": timestamp_column,
            "hour_alignment": "floor_utc_hour",
            "canonical_partitions_modified": False,
        },
        "inputs": {
            "weather_file": str(weather_path.relative_to(PROJECT_ROOT)),
            "weather_file_sha256": _file_sha256(weather_path),
            "partition_files": {
                name: str(path.relative_to(PROJECT_ROOT))
                for name, path in partition_files.items()
            },
        },
        "overall": {
            "events": int(len(sidecar)),
            "matched_events": int(len(sidecar) - overall_missing),
            "missing_events": overall_missing,
            "coverage": overall_coverage,
            "unique_weather_hours_used": int(sidecar[WEATHER_TIME_COLUMN].nunique()),
        },
        "partitions": partition_report,
        "weather_group_counts": {
            str(key): int(value)
            for key, value in sidecar["weather_group"].value_counts().items()
        },
        "features_file": features_path.name,
    }
    report["features_file_sha256"] = _file_sha256(temporary_features)
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_features.replace(features_path)
    temporary_report.replace(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/weather.yaml")
    args = parser.parse_args()
    report = build_weather_sidecar(args.config)
    print(f"Events: {report['overall']['events']:,}")
    print(f"Weather coverage: {report['overall']['coverage']:.2%}")
    for name, values in report["partitions"].items():
        print(f"{name}: {values['matched_events']:,}/{values['events']:,}")


if __name__ == "__main__":
    main()

