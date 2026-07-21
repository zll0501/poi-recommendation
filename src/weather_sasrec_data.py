"""Adapters that attach event-level weather to existing SASRec batches."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
import torch

from src.weather_features import WeatherFeatureStore


def build_poi_category_index(
    poi_metadata: pd.DataFrame,
    *,
    num_pois: int,
    pad_id: int = 0,
    unknown_id: int = 1,
) -> torch.LongTensor:
    """Build a complete POI-vocabulary to category-vocabulary lookup tensor."""
    required = {"poi_idx", "category_idx"}
    missing = sorted(required.difference(poi_metadata.columns))
    if missing:
        raise ValueError(f"POI metadata is missing columns: {missing}")
    if num_pois < 2 or not 0 <= pad_id < num_pois or not 0 <= unknown_id < num_pois:
        raise ValueError("invalid POI vocabulary or special-token IDs")

    metadata = poi_metadata[["poi_idx", "category_idx"]].copy()
    metadata["poi_idx"] = pd.to_numeric(metadata["poi_idx"], errors="raise").astype(
        "int64"
    )
    metadata["category_idx"] = pd.to_numeric(
        metadata["category_idx"], errors="raise"
    ).astype("int64")
    if metadata["poi_idx"].duplicated().any():
        raise ValueError("POI metadata contains duplicate poi_idx values")
    if metadata["poi_idx"].lt(0).any() or metadata["poi_idx"].ge(num_pois).any():
        raise ValueError("POI metadata contains an out-of-vocabulary poi_idx")
    if metadata["category_idx"].lt(0).any():
        raise ValueError("POI metadata contains a negative category_idx")

    lookup = torch.full((num_pois,), int(unknown_id), dtype=torch.long)
    lookup[pad_id] = int(pad_id)
    if not metadata.empty:
        lookup[torch.as_tensor(metadata["poi_idx"].to_numpy(copy=True))] = torch.as_tensor(
            metadata["category_idx"].to_numpy(copy=True), dtype=torch.long
        )
    return lookup


class WeatherSASRecCollator:
    """Wrap an existing SASRec collator and append target-query weather."""

    def __init__(
        self,
        base_collator: Callable[[Sequence[Any]], dict[str, Any]],
        weather_store: WeatherFeatureStore,
    ) -> None:
        self.base_collator = base_collator
        self.weather_store = weather_store

    def __call__(self, samples: Sequence[Any]) -> dict[str, Any]:
        batch = self.base_collator(samples)
        if "inputs" not in batch or "event_id" not in batch:
            raise KeyError("base SASRec batch must contain inputs and event_id")
        inputs = batch["inputs"]
        event_ids = batch["event_id"]
        if not isinstance(inputs, dict) or not isinstance(event_ids, torch.Tensor):
            raise TypeError("base SASRec collator returned an invalid batch")
        weather = self.weather_store.get_by_event_ids(event_ids.tolist())
        if weather.group_idx.size(0) != event_ids.size(0):
            raise RuntimeError("weather batch is not aligned with SASRec events")
        inputs["weather_group_idx"] = weather.group_idx
        inputs["weather_numeric"] = weather.numeric
        return batch
