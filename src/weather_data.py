"""Read and attach optional event-keyed weather context features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_weather_sidecar(
    config_path: str | Path = "configs/weather.yaml",
) -> pd.DataFrame:
    config = load_yaml(config_path)
    matching = config["matching"]
    path = _resolve(matching["output_directory"]) / matching["features_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"weather sidecar is missing: {path}; run python -m src.match_weather"
        )
    frame = pd.read_csv(path)
    if "event_id" not in frame or frame["event_id"].duplicated().any():
        raise ValueError("weather sidecar must contain unique event_id values")
    if "weather_missing" not in frame or frame["weather_missing"].astype(bool).any():
        raise ValueError("weather sidecar contains missing weather matches")
    return frame


def attach_weather(
    events: pd.DataFrame,
    sidecar: pd.DataFrame,
    columns: Iterable[str] = ("weather_group",),
) -> pd.DataFrame:
    selected = list(columns)
    missing = sorted({"event_id", *selected}.difference(sidecar.columns))
    if missing:
        raise ValueError(f"weather sidecar is missing columns: {missing}")
    overlap = sorted(set(selected).intersection(events.columns))
    if overlap:
        raise ValueError(f"event data already contains weather columns: {overlap}")
    result = events.merge(
        sidecar[["event_id", *selected]],
        on="event_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if result[selected].isna().any().any():
        raise ValueError("some events have no weather sidecar row")
    return result
