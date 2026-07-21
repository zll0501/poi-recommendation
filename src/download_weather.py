"""Download and validate reproducible hourly NYC weather context data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIME_COLUMN = "weather_time_utc"
NUMERIC_RANGES: dict[str, tuple[float, float | None]] = {
    "temperature_2m": (-90.0, 65.0),
    "relative_humidity_2m": (0.0, 100.0),
    "precipitation": (0.0, None),
    "weather_code": (0.0, 99.0),
    "wind_speed_10m": (0.0, None),
}


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def build_request_url(config: dict[str, Any]) -> str:
    """Build the complete, reproducible Open-Meteo request URL."""
    source = config["source"]
    location = config["location"]
    time_config = config["time"]
    features = list(config["hourly_features"])
    if not features:
        raise ValueError("hourly_features cannot be empty")

    params = {
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
        "start_date": str(time_config["start_date"]),
        "end_date": str(time_config["end_date"]),
        "hourly": ",".join(features),
        "timezone": str(time_config.get("timezone", "UTC")),
        "models": str(source["dataset"]),
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
    }
    return f"{str(source['api_url']).rstrip('/')}?{urlencode(params)}"


def fetch_json(url: str, timeout_seconds: int, max_attempts: int) -> dict[str, Any]:
    """Fetch JSON with bounded retries and a descriptive user agent."""
    if timeout_seconds < 1 or max_attempts < 1:
        raise ValueError("timeout_seconds and max_attempts must be positive")

    request = Request(
        url,
        headers={"User-Agent": "poi-recommendation-course-project/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError("weather API response root must be an object")
            if "error" in payload:
                raise ValueError(f"weather API error: {payload.get('reason', payload['error'])}")
            return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"weather download failed after {max_attempts} attempts") from last_error


def validate_hourly_payload(
    payload: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert an API payload to a strictly validated hourly table."""
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise ValueError("weather response is missing hourly data or units")

    features = list(config["hourly_features"])
    required = ["time", *features]
    missing = [column for column in required if column not in hourly]
    if missing:
        raise ValueError(f"weather response is missing fields: {missing}")

    lengths = {column: len(hourly[column]) for column in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"hourly arrays have inconsistent lengths: {lengths}")

    frame = pd.DataFrame({column: hourly[column] for column in required}).rename(
        columns={"time": TIME_COLUMN}
    )
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], utc=True, errors="coerce")
    if frame[TIME_COLUMN].isna().any():
        raise ValueError("weather data contains invalid timestamps")
    if frame[TIME_COLUMN].duplicated().any():
        raise ValueError("weather data contains duplicate timestamps")
    if not frame[TIME_COLUMN].is_monotonic_increasing:
        raise ValueError("weather timestamps are not increasing")

    start = pd.Timestamp(config["time"]["start_date"], tz="UTC")
    end = pd.Timestamp(config["time"]["end_date"], tz="UTC") + pd.Timedelta(hours=23)
    expected_index = pd.date_range(start, end, freq="h")
    if len(frame) != len(expected_index):
        raise ValueError(
            f"expected {len(expected_index)} hourly rows, received {len(frame)}"
        )
    actual_index = pd.DatetimeIndex(frame[TIME_COLUMN])
    if not actual_index.equals(expected_index):
        absent = expected_index.difference(actual_index)
        unexpected = actual_index.difference(expected_index)
        raise ValueError(
            "weather timeline is not complete: "
            f"missing={len(absent)}, unexpected={len(unexpected)}"
        )

    for column in features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_counts = {column: int(frame[column].isna().sum()) for column in features}
    if any(missing_counts.values()):
        raise ValueError(f"weather features contain missing values: {missing_counts}")

    range_report: dict[str, dict[str, float]] = {}
    for column in features:
        minimum = float(frame[column].min())
        maximum = float(frame[column].max())
        range_report[column] = {"min": minimum, "max": maximum}
        allowed = NUMERIC_RANGES.get(column)
        if allowed is not None:
            lower, upper = allowed
            if minimum < lower or (upper is not None and maximum > upper):
                raise ValueError(
                    f"{column} outside plausible range: [{minimum}, {maximum}]"
                )

    frame["weather_code"] = frame["weather_code"].astype("int64")
    report = {
        "rows": int(len(frame)),
        "start_utc": frame[TIME_COLUMN].iloc[0].isoformat(),
        "end_utc": frame[TIME_COLUMN].iloc[-1].isoformat(),
        "unique_timestamps": int(frame[TIME_COLUMN].nunique()),
        "duplicate_timestamps": int(frame[TIME_COLUMN].duplicated().sum()),
        "missing_values": missing_counts,
        "feature_ranges": range_report,
        "units": {feature: units.get(feature) for feature in features},
    }
    return frame, report


def download_weather(config_path: str | Path = "configs/weather.yaml") -> dict[str, Any]:
    """Download, validate and atomically persist hourly weather plus metadata."""
    config = load_yaml(config_path)
    url = build_request_url(config)
    source = config["source"]
    payload = fetch_json(
        url,
        timeout_seconds=int(source.get("timeout_seconds", 60)),
        max_attempts=int(source.get("max_attempts", 3)),
    )
    frame, validation = validate_hourly_payload(payload, config)

    output = config["output"]
    output_dir = _resolve_project_path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    weather_path = output_dir / output["weather_file"]
    metadata_path = output_dir / output["metadata_file"]
    temporary_weather = weather_path.with_suffix(weather_path.suffix + ".tmp")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")

    frame.to_csv(temporary_weather, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    checksum = hashlib.sha256(temporary_weather.read_bytes()).hexdigest()
    metadata = {
        "source": {
            "provider": source["provider"],
            "dataset": source["dataset"],
            "api_url": source["api_url"],
            "request_url": url,
        },
        "requested_location": config["location"],
        "returned_location": {
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "elevation": payload.get("elevation"),
            "timezone": payload.get("timezone"),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weather_file": weather_path.name,
        "weather_file_sha256": checksum,
        "validation": validation,
    }
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_weather.replace(weather_path)
    temporary_metadata.replace(metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/weather.yaml")
    args = parser.parse_args()
    metadata = download_weather(args.config)
    validation = metadata["validation"]
    print(f"Weather rows: {validation['rows']:,}")
    print(f"UTC range: {validation['start_utc']} -> {validation['end_utc']}")
    print(f"SHA256: {metadata['weather_file_sha256']}")


if __name__ == "__main__":
    main()

