"""Deterministic popularity baselines for next-POI recommendation."""

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


class TimePopular(BaseRecommender):
    """Rank POIs within each query-time bucket with a global fallback.

    Every bucket ranking contains the complete train-POI universe: POIs observed
    in that bucket are ordered by bucket count, then unseen POIs follow the
    global popularity order. This keeps the candidate protocol identical to
    GlobalPopular and guarantees enough recommendations for sparse buckets.
    """

    def __init__(self, time_column: str = "time_slot") -> None:
        if not isinstance(time_column, str) or not time_column.strip():
            raise ValueError("time_column must be a non-empty string")
        self.time_column = time_column
        self.global_model_: GlobalPopular | None = None
        self.rankings_: dict[str, tuple[int, ...]] | None = None
        self.visit_counts_: dict[str, dict[int, int]] | None = None

    @property
    def is_fitted(self) -> bool:
        return (
            self.global_model_ is not None
            and self.global_model_.is_fitted
            and self.rankings_ is not None
            and self.visit_counts_ is not None
        )

    def fit(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame | None = None,
    ) -> "TimePopular":
        """Fit all counts on training events only."""
        del valid_data
        required = {"poi_idx", self.time_column}
        missing = sorted(required.difference(train_data.columns))
        if missing:
            raise ValueError(f"train_data is missing columns: {missing}")
        if train_data.empty:
            raise ValueError("train_data cannot be empty")
        if train_data[self.time_column].isna().any():
            raise ValueError(f"train_data.{self.time_column} cannot contain missing values")

        global_model = GlobalPopular().fit(train_data)
        global_ranking = global_model.ranking_
        assert global_ranking is not None
        rankings: dict[str, tuple[int, ...]] = {}
        slot_counts: dict[str, dict[int, int]] = {}

        working = train_data[["poi_idx", self.time_column]].copy()
        working[self.time_column] = working[self.time_column].astype(str)
        for slot, group in working.groupby(self.time_column, sort=True, observed=True):
            poi_ids = pd.to_numeric(group["poi_idx"], errors="coerce")
            if poi_ids.isna().any() or (poi_ids % 1 != 0).any():
                raise ValueError("train_data.poi_idx must contain integers")
            counts = {
                int(poi): int(count)
                for poi, count in poi_ids.astype("int64").value_counts().items()
            }
            observed = sorted(counts, key=lambda poi: (-counts[poi], poi))
            observed_set = set(observed)
            fallback = [poi for poi in global_ranking if poi not in observed_set]
            rankings[str(slot)] = tuple(observed + fallback)
            slot_counts[str(slot)] = counts

        self.global_model_ = global_model
        self.rankings_ = rankings
        self.visit_counts_ = slot_counts
        return self

    def recommend(self, test_data: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """Recommend from the matching time bucket, falling back globally."""
        if not self.is_fitted:
            raise RuntimeError("TimePopular must be fitted before recommendation")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        required = {"event_id", self.time_column}
        missing = sorted(required.difference(test_data.columns))
        if missing:
            raise ValueError(f"test_data is missing columns: {missing}")
        if test_data["event_id"].duplicated().any():
            raise ValueError("test_data.event_id values must be unique")
        if test_data[self.time_column].isna().any():
            raise ValueError(f"test_data.{self.time_column} cannot contain missing values")

        assert self.global_model_ is not None
        assert self.global_model_.ranking_ is not None
        assert self.rankings_ is not None
        if top_k > len(self.global_model_.ranking_):
            raise ValueError("top_k cannot exceed the number of train POIs")
        recommendations = {
            int(row.event_id): self.rankings_.get(
                str(getattr(row, self.time_column)), self.global_model_.ranking_
            )[:top_k]
            for row in test_data.itertuples(index=False)
        }
        return recommendations_to_frame(recommendations)

    def ranking_frame(self) -> pd.DataFrame:
        """Return all time-bucket rankings, including global fallback rows."""
        if not self.is_fitted:
            raise RuntimeError("TimePopular must be fitted before exporting ranking")
        assert self.rankings_ is not None
        assert self.visit_counts_ is not None
        rows = []
        for slot, ranking in sorted(self.rankings_.items()):
            counts = self.visit_counts_[slot]
            rows.extend(
                {
                    "time_slot": slot,
                    "rank": rank,
                    "poi_idx": poi,
                    "visit_count": counts.get(poi, 0),
                    "is_global_fallback": poi not in counts,
                }
                for rank, poi in enumerate(ranking, start=1)
            )
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("TimePopular must be fitted before saving")
        assert self.global_model_ is not None
        assert self.global_model_.ranking_ is not None
        assert self.global_model_.visit_counts_ is not None
        assert self.rankings_ is not None
        assert self.visit_counts_ is not None
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model": "TimePopular",
            "time_column": self.time_column,
            "global_ranking": list(self.global_model_.ranking_),
            "global_visit_counts": {
                str(k): v for k, v in self.global_model_.visit_counts_.items()
            },
            "rankings": {slot: list(values) for slot, values in self.rankings_.items()},
            "visit_counts": {
                slot: {str(k): v for k, v in counts.items()}
                for slot, counts in self.visit_counts_.items()
            },
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, path: str | Path) -> "TimePopular":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model") != "TimePopular":
            raise ValueError("model file is not a TimePopular artifact")
        global_model = GlobalPopular()
        global_model.ranking_ = tuple(int(poi) for poi in payload["global_ranking"])
        global_model.visit_counts_ = {
            int(poi): int(count)
            for poi, count in payload["global_visit_counts"].items()
        }
        rankings = {
            str(slot): tuple(int(poi) for poi in values)
            for slot, values in payload["rankings"].items()
        }
        counts = {
            str(slot): {int(poi): int(count) for poi, count in values.items()}
            for slot, values in payload["visit_counts"].items()
        }
        if not global_model.is_fitted or not rankings:
            raise ValueError("invalid TimePopular model artifact")
        candidate_set = set(global_model.ranking_)
        if any(set(ranking) != candidate_set for ranking in rankings.values()):
            raise ValueError("time rankings must contain the global candidate universe")
        self.time_column = str(payload["time_column"])
        self.global_model_ = global_model
        self.rankings_ = rankings
        self.visit_counts_ = counts
        return self


class TimeWeatherPopular(BaseRecommender):
    """Popularity conditioned on query time and weather with safe fallbacks."""

    def __init__(
        self,
        time_column: str = "time_slot",
        weather_column: str = "weather_group",
        min_bucket_events: int = 50,
    ) -> None:
        if not time_column or not weather_column:
            raise ValueError("time_column and weather_column must be non-empty")
        if min_bucket_events < 1:
            raise ValueError("min_bucket_events must be positive")
        self.time_column = time_column
        self.weather_column = weather_column
        self.min_bucket_events = int(min_bucket_events)
        self.time_model_: TimePopular | None = None
        self.rankings_: dict[tuple[str, str], tuple[int, ...]] | None = None
        self.visit_counts_: dict[tuple[str, str], dict[int, int]] | None = None
        self.bucket_events_: dict[tuple[str, str], int] | None = None

    @property
    def is_fitted(self) -> bool:
        return (
            self.time_model_ is not None
            and self.time_model_.is_fitted
            and self.rankings_ is not None
            and self.visit_counts_ is not None
            and self.bucket_events_ is not None
        )

    def fit(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame | None = None,
    ) -> "TimeWeatherPopular":
        """Fit exact context buckets exclusively from training events."""
        del valid_data
        required = {"poi_idx", self.time_column, self.weather_column}
        missing = sorted(required.difference(train_data.columns))
        if missing:
            raise ValueError(f"train_data is missing columns: {missing}")
        if train_data.empty:
            raise ValueError("train_data cannot be empty")
        if train_data[[self.time_column, self.weather_column]].isna().any().any():
            raise ValueError("time and weather context cannot contain missing values")

        time_model = TimePopular(self.time_column).fit(train_data)
        assert time_model.global_model_ is not None
        assert time_model.global_model_.ranking_ is not None
        assert time_model.rankings_ is not None
        global_ranking = time_model.global_model_.ranking_
        rankings: dict[tuple[str, str], tuple[int, ...]] = {}
        visit_counts: dict[tuple[str, str], dict[int, int]] = {}
        bucket_events: dict[tuple[str, str], int] = {}

        working = train_data[["poi_idx", self.time_column, self.weather_column]].copy()
        working[self.time_column] = working[self.time_column].astype(str)
        working[self.weather_column] = working[self.weather_column].astype(str)
        grouped = working.groupby(
            [self.time_column, self.weather_column], sort=True, observed=True
        )
        for (time_value, weather_value), group in grouped:
            key = (str(time_value), str(weather_value))
            bucket_events[key] = int(len(group))
            if len(group) < self.min_bucket_events:
                continue
            poi_ids = pd.to_numeric(group["poi_idx"], errors="coerce")
            if poi_ids.isna().any() or (poi_ids % 1 != 0).any():
                raise ValueError("train_data.poi_idx must contain integers")
            counts = {
                int(poi): int(count)
                for poi, count in poi_ids.astype("int64").value_counts().items()
            }
            observed = sorted(counts, key=lambda poi: (-counts[poi], poi))
            observed_set = set(observed)
            time_ranking = time_model.rankings_.get(str(time_value), global_ranking)
            fallback = [poi for poi in time_ranking if poi not in observed_set]
            rankings[key] = tuple(observed + fallback)
            visit_counts[key] = counts

        self.time_model_ = time_model
        self.rankings_ = rankings
        self.visit_counts_ = visit_counts
        self.bucket_events_ = bucket_events
        return self

    def recommend(self, test_data: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """Use exact context, then time-only, then global popularity."""
        if not self.is_fitted:
            raise RuntimeError("TimeWeatherPopular must be fitted before recommendation")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        required = {"event_id", self.time_column, self.weather_column}
        missing = sorted(required.difference(test_data.columns))
        if missing:
            raise ValueError(f"test_data is missing columns: {missing}")
        if test_data["event_id"].duplicated().any():
            raise ValueError("test_data.event_id values must be unique")
        if test_data[[self.time_column, self.weather_column]].isna().any().any():
            raise ValueError("time and weather context cannot contain missing values")

        assert self.time_model_ is not None
        assert self.time_model_.global_model_ is not None
        assert self.time_model_.global_model_.ranking_ is not None
        assert self.time_model_.rankings_ is not None
        assert self.rankings_ is not None
        global_ranking = self.time_model_.global_model_.ranking_
        if top_k > len(global_ranking):
            raise ValueError("top_k cannot exceed the number of train POIs")

        recommendations: dict[int, tuple[int, ...]] = {}
        for row in test_data.itertuples(index=False):
            time_value = str(getattr(row, self.time_column))
            weather_value = str(getattr(row, self.weather_column))
            ranking = self.rankings_.get((time_value, weather_value))
            if ranking is None:
                ranking = self.time_model_.rankings_.get(time_value, global_ranking)
            recommendations[int(row.event_id)] = ranking[:top_k]
        return recommendations_to_frame(recommendations)

    def ranking_frame(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("TimeWeatherPopular must be fitted before exporting ranking")
        assert self.rankings_ is not None
        assert self.visit_counts_ is not None
        assert self.bucket_events_ is not None
        rows = []
        for key, ranking in sorted(self.rankings_.items()):
            time_value, weather_value = key
            counts = self.visit_counts_[key]
            rows.extend(
                {
                    self.time_column: time_value,
                    self.weather_column: weather_value,
                    "bucket_events": self.bucket_events_[key],
                    "rank": rank,
                    "poi_idx": poi,
                    "visit_count": counts.get(poi, 0),
                    "is_fallback": poi not in counts,
                }
                for rank, poi in enumerate(ranking, start=1)
            )
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("TimeWeatherPopular must be fitted before saving")
        assert self.time_model_ is not None
        assert self.time_model_.global_model_ is not None
        assert self.time_model_.global_model_.ranking_ is not None
        assert self.time_model_.global_model_.visit_counts_ is not None
        assert self.time_model_.rankings_ is not None
        assert self.time_model_.visit_counts_ is not None
        assert self.rankings_ is not None
        assert self.visit_counts_ is not None
        assert self.bucket_events_ is not None
        payload: dict[str, Any] = {
            "model": "TimeWeatherPopular",
            "time_column": self.time_column,
            "weather_column": self.weather_column,
            "min_bucket_events": self.min_bucket_events,
            "global_ranking": list(self.time_model_.global_model_.ranking_),
            "global_visit_counts": {
                str(k): v for k, v in self.time_model_.global_model_.visit_counts_.items()
            },
            "time_rankings": {
                slot: list(ranking) for slot, ranking in self.time_model_.rankings_.items()
            },
            "time_visit_counts": {
                slot: {str(k): v for k, v in counts.items()}
                for slot, counts in self.time_model_.visit_counts_.items()
            },
            "weather_buckets": [
                {
                    "time": key[0],
                    "weather": key[1],
                    "events": self.bucket_events_[key],
                    "ranking": list(ranking),
                    "visit_counts": {
                        str(k): v for k, v in self.visit_counts_[key].items()
                    },
                }
                for key, ranking in sorted(self.rankings_.items())
            ],
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, path: str | Path) -> "TimeWeatherPopular":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model") != "TimeWeatherPopular":
            raise ValueError("model file is not a TimeWeatherPopular artifact")
        global_model = GlobalPopular()
        global_model.ranking_ = tuple(int(poi) for poi in payload["global_ranking"])
        global_model.visit_counts_ = {
            int(poi): int(count)
            for poi, count in payload["global_visit_counts"].items()
        }
        time_model = TimePopular(str(payload["time_column"]))
        time_model.global_model_ = global_model
        time_model.rankings_ = {
            str(slot): tuple(int(poi) for poi in ranking)
            for slot, ranking in payload["time_rankings"].items()
        }
        time_model.visit_counts_ = {
            str(slot): {int(poi): int(count) for poi, count in counts.items()}
            for slot, counts in payload["time_visit_counts"].items()
        }
        rankings = {}
        counts = {}
        events = {}
        for bucket in payload["weather_buckets"]:
            key = (str(bucket["time"]), str(bucket["weather"]))
            rankings[key] = tuple(int(poi) for poi in bucket["ranking"])
            counts[key] = {
                int(poi): int(count) for poi, count in bucket["visit_counts"].items()
            }
            events[key] = int(bucket["events"])
        if not time_model.is_fitted:
            raise ValueError("invalid TimeWeatherPopular time fallback artifact")
        candidate_set = set(global_model.ranking_)
        if any(set(ranking) != candidate_set for ranking in rankings.values()):
            raise ValueError("weather rankings must contain the global candidate universe")
        self.time_column = str(payload["time_column"])
        self.weather_column = str(payload["weather_column"])
        self.min_bucket_events = int(payload["min_bucket_events"])
        self.time_model_ = time_model
        self.rankings_ = rankings
        self.visit_counts_ = counts
        self.bucket_events_ = events
        return self
