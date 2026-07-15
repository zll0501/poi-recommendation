"""Deterministic global-popularity baseline for next-POI recommendation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluator import recommendations_to_frame
from src.models.base import BaseRecommender


class GlobalPopular(BaseRecommender):
    """Rank every train POI by visit count and reuse that ranking for all events."""

    def __init__(self) -> None:
        self.ranking_: tuple[int, ...] | None = None
        self.visit_counts_: dict[int, int] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.ranking_ is not None and self.visit_counts_ is not None

    def fit(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame | None = None,
    ) -> "GlobalPopular":
        """Fit POI counts on training events only; validation is intentionally unused."""
        del valid_data
        if "poi_idx" not in train_data.columns:
            raise ValueError("train_data is missing column: poi_idx")
        if train_data.empty:
            raise ValueError("train_data cannot be empty")
        poi_ids = pd.to_numeric(train_data["poi_idx"], errors="coerce")
        if poi_ids.isna().any() or (poi_ids % 1 != 0).any():
            raise ValueError("train_data.poi_idx must contain integers")
        counts = (
            poi_ids.astype("int64")
            .value_counts()
            .rename_axis("poi_idx")
            .reset_index(name="visit_count")
            .sort_values(
                ["visit_count", "poi_idx"],
                ascending=[False, True],
                kind="stable",
            )
        )
        self.ranking_ = tuple(counts["poi_idx"].astype(int))
        self.visit_counts_ = {
            int(row.poi_idx): int(row.visit_count)
            for row in counts.itertuples(index=False)
        }
        return self

    def recommend(self, test_data: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """Return the shared event_id/rank/poi_idx long prediction table."""
        if not self.is_fitted:
            raise RuntimeError("GlobalPopular must be fitted before recommendation")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if top_k > len(self.ranking_):
            raise ValueError("top_k cannot exceed the number of train POIs")
        if "event_id" not in test_data.columns:
            raise ValueError("test_data is missing column: event_id")
        if test_data["event_id"].duplicated().any():
            raise ValueError("test_data.event_id values must be unique")
        top_pois = self.ranking_[:top_k]
        recommendations = {
            int(event_id): top_pois for event_id in test_data["event_id"]
        }
        return recommendations_to_frame(recommendations)

    def ranking_frame(self) -> pd.DataFrame:
        """Return the fitted full ranking for audit and reporting."""
        if not self.is_fitted:
            raise RuntimeError("GlobalPopular must be fitted before exporting ranking")
        return pd.DataFrame({
            "rank": range(1, len(self.ranking_) + 1),
            "poi_idx": self.ranking_,
            "visit_count": [self.visit_counts_[poi] for poi in self.ranking_],
        })

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("GlobalPopular must be fitted before saving")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model": "GlobalPopular",
            "ranking": list(self.ranking_),
            "visit_counts": {str(k): v for k, v in self.visit_counts_.items()},
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, path: str | Path) -> "GlobalPopular":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model") != "GlobalPopular":
            raise ValueError("model file is not a GlobalPopular artifact")
        ranking = tuple(int(poi) for poi in payload["ranking"])
        counts = {int(poi): int(count) for poi, count in payload["visit_counts"].items()}
        if not ranking or set(ranking) != set(counts):
            raise ValueError("invalid GlobalPopular model artifact")
        self.ranking_ = ranking
        self.visit_counts_ = counts
        return self
