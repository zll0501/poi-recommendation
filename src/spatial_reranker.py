"""Spatial utilities for leakage-safe next-POI reranking.

This module deliberately stays independent of the SASRec implementation.  It
uses the last observed check-in as the query location and only training events
when estimating mobility statistics.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088
TRANSITION_COLUMNS = {
    "event_id",
    "user_idx",
    "poi_idx",
    "timestamp",
    "latitude",
    "longitude",
}


def haversine_km(
    latitude_a: Iterable[float] | np.ndarray,
    longitude_a: Iterable[float] | np.ndarray,
    latitude_b: Iterable[float] | np.ndarray,
    longitude_b: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return element-wise great-circle distances in kilometres."""
    lat_a = np.radians(np.asarray(latitude_a, dtype=float))
    lon_a = np.radians(np.asarray(longitude_a, dtype=float))
    lat_b = np.radians(np.asarray(latitude_b, dtype=float))
    lon_b = np.radians(np.asarray(longitude_b, dtype=float))
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    value = np.clip(value, 0.0, 1.0)
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(value))


def build_training_transitions(train: pd.DataFrame) -> pd.DataFrame:
    """Build consecutive transitions using training events only.

    Sorting is performed within each user.  The function never joins validation
    or test targets, so its output is safe for selecting spatial priors.
    """
    missing = sorted(TRANSITION_COLUMNS.difference(train.columns))
    if missing:
        raise ValueError(f"training data is missing spatial columns: {missing}")

    data = train.loc[:, sorted(TRANSITION_COLUMNS)].copy()
    data["timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce")
    numeric_columns = ["latitude", "longitude", "user_idx", "poi_idx", "event_id"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    invalid = (
        data[list(TRANSITION_COLUMNS)].isna().any(axis=1)
        | ~data["latitude"].between(-90.0, 90.0)
        | ~data["longitude"].between(-180.0, 180.0)
    )
    if invalid.any():
        raise ValueError(f"training data contains {int(invalid.sum())} invalid spatial rows")

    data = data.sort_values(
        ["user_idx", "timestamp", "event_id"], kind="stable"
    ).reset_index(drop=True)
    grouped = data.groupby("user_idx", sort=False, observed=True)
    data["previous_event_id"] = grouped["event_id"].shift()
    data["previous_poi_idx"] = grouped["poi_idx"].shift()
    data["previous_timestamp"] = grouped["timestamp"].shift()
    data["previous_latitude"] = grouped["latitude"].shift()
    data["previous_longitude"] = grouped["longitude"].shift()

    transitions = data.loc[data["previous_event_id"].notna()].copy()
    transitions["time_gap_hours"] = (
        transitions["timestamp"] - transitions["previous_timestamp"]
    ) / 3600.0
    if transitions["time_gap_hours"].le(0).any():
        raise ValueError("training transitions must have strictly increasing timestamps")

    transitions["distance_km"] = haversine_km(
        transitions["previous_latitude"],
        transitions["previous_longitude"],
        transitions["latitude"],
        transitions["longitude"],
    )
    transitions["same_poi"] = transitions["poi_idx"].eq(
        transitions["previous_poi_idx"]
    )
    columns = [
        "user_idx",
        "previous_event_id",
        "event_id",
        "previous_poi_idx",
        "poi_idx",
        "previous_timestamp",
        "timestamp",
        "time_gap_hours",
        "distance_km",
        "same_poi",
    ]
    return transitions.loc[:, columns].reset_index(drop=True)


def summarize_training_transitions(
    transitions: pd.DataFrame,
    distance_thresholds_km: Iterable[float] = (1, 3, 5, 10, 20),
) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarize overall and time-gap-conditioned mobility patterns."""
    required = {"user_idx", "distance_km", "time_gap_hours", "same_poi"}
    missing = sorted(required.difference(transitions.columns))
    if missing:
        raise ValueError(f"transitions are missing columns: {missing}")
    if transitions.empty:
        raise ValueError("at least one training transition is required")

    distances = pd.to_numeric(transitions["distance_km"], errors="raise")
    quantiles = {
        f"p{int(probability * 100):02d}_km": float(distances.quantile(probability))
        for probability in (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    }
    thresholds = [float(value) for value in distance_thresholds_km]
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("distance thresholds must be positive")

    within = {
        f"within_{value:g}km_ratio": float(distances.le(value).mean())
        for value in thresholds
    }
    report: dict[str, object] = {
        "fit_partition": "train_only",
        "transition_count": int(len(transitions)),
        "user_count": int(transitions["user_idx"].nunique()),
        "same_poi_ratio": float(transitions["same_poi"].mean()),
        "mean_distance_km": float(distances.mean()),
        "maximum_distance_km": float(distances.max()),
        "distance_quantiles": quantiles,
        "distance_locality": within,
    }

    bins = [-np.inf, 1.0, 6.0, 24.0, 24.0 * 7.0, np.inf]
    labels = ["0-1h", "1-6h", "6-24h", "1-7d", ">7d"]
    working = transitions.copy()
    working["time_gap_bucket"] = pd.cut(
        working["time_gap_hours"], bins=bins, labels=labels, right=True
    )
    rows: list[dict[str, object]] = []
    for label in labels:
        group = working.loc[working["time_gap_bucket"] == label]
        if group.empty:
            continue
        values = group["distance_km"]
        row: dict[str, object] = {
            "time_gap_bucket": label,
            "transition_count": int(len(group)),
            "mean_distance_km": float(values.mean()),
            "median_distance_km": float(values.median()),
            "p90_distance_km": float(values.quantile(0.90)),
        }
        row.update(
            {
                f"within_{value:g}km_ratio": float(values.le(value).mean())
                for value in thresholds
            }
        )
        rows.append(row)
    by_time_gap = pd.DataFrame(rows)
    return report, by_time_gap
