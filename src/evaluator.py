"""Shared evaluator and prediction format for every recommendation model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.metrics import evaluation_coverage, hit_rate_at_k, mrr_at_k, ndcg_at_k


TARGET_COLUMNS = {"event_id", "user_idx", "poi_idx"}
PREDICTION_COLUMNS = {"event_id", "rank", "poi_idx"}


def recommendations_to_frame(
    recommendations: Mapping[int, Sequence[int]],
) -> pd.DataFrame:
    """Convert {event_id: ranked POI IDs} to the shared long-table format."""
    rows = [
        {"event_id": int(event_id), "rank": rank, "poi_idx": int(poi_idx)}
        for event_id, poi_ids in recommendations.items()
        for rank, poi_idx in enumerate(poi_ids, start=1)
    ]
    return pd.DataFrame(rows, columns=["event_id", "rank", "poi_idx"])


def validate_predictions(
    predictions: pd.DataFrame,
    expected_event_ids: set[int],
    candidate_poi_ids: set[int],
    max_k: int,
) -> pd.DataFrame:
    """Reject malformed rankings instead of silently producing misleading scores."""
    missing_columns = sorted(PREDICTION_COLUMNS.difference(predictions.columns))
    if missing_columns:
        raise ValueError(f"predictions are missing columns: {missing_columns}")
    frame = predictions[["event_id", "rank", "poi_idx"]].copy()
    if frame.isna().any().any():
        raise ValueError("predictions cannot contain missing values")
    for column in ("event_id", "rank", "poi_idx"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"predictions.{column} must contain integers")
        frame[column] = numeric.astype("int64")
    if frame["rank"].lt(1).any() or frame["rank"].gt(max_k).any():
        raise ValueError(f"prediction ranks must be between 1 and {max_k}")
    if frame.duplicated(["event_id", "rank"]).any():
        raise ValueError("each event_id/rank pair must be unique")
    if frame.duplicated(["event_id", "poi_idx"]).any():
        raise ValueError("a POI cannot appear twice for the same event")
    unknown_candidates = set(frame["poi_idx"]).difference(candidate_poi_ids)
    if unknown_candidates:
        raise ValueError("predictions contain POIs outside the train candidate set")

    predicted_event_ids = set(frame["event_id"])
    missing_events = expected_event_ids.difference(predicted_event_ids)
    extra_events = predicted_event_ids.difference(expected_event_ids)
    if missing_events or extra_events:
        raise ValueError(
            "prediction event IDs must exactly match evaluable targets: "
            f"missing={len(missing_events)}, extra={len(extra_events)}"
        )
    rank_lists = frame.groupby("event_id", observed=True)["rank"].apply(sorted)
    if any(len(ranks) != max_k for ranks in rank_lists):
        raise ValueError(f"each event must contain exactly {max_k} recommendations")
    invalid_rankings = sum(ranks != list(range(1, len(ranks) + 1)) for ranks in rank_lists)
    if invalid_rankings:
        raise ValueError("ranks for each event must be contiguous and start at 1")
    return frame.sort_values(["event_id", "rank"], kind="stable").reset_index(drop=True)


def evaluate_next_poi(
    targets: pd.DataFrame,
    predictions: pd.DataFrame | Mapping[int, Sequence[int]],
    candidate_poi_ids: Sequence[int],
    *,
    unknown_id: int = 1,
    ks: Sequence[int] = (5, 10),
    mrr_k: int = 10,
) -> dict[str, Any]:
    """Evaluate one-target rankings under the shared closed-world protocol.

    Coverage is the fraction of all target rows with a train-known user and POI.
    Ranking metrics are computed only on those evaluable rows.
    """
    missing_columns = sorted(TARGET_COLUMNS.difference(targets.columns))
    if missing_columns:
        raise ValueError(f"targets are missing columns: {missing_columns}")
    if targets.empty:
        raise ValueError("targets cannot be empty")
    if targets["event_id"].duplicated().any():
        raise ValueError("target event_id values must be unique")
    cutoffs = sorted({int(k) for k in ks})
    if not cutoffs or any(k < 1 for k in cutoffs) or mrr_k < 1:
        raise ValueError("ks and mrr_k must contain positive integers")

    candidates = {int(poi_idx) for poi_idx in candidate_poi_ids}
    if not candidates or unknown_id in candidates:
        raise ValueError("candidate_poi_ids must be non-empty and exclude UNK")
    evaluable = targets.loc[
        targets["user_idx"].ne(unknown_id)
        & targets["poi_idx"].isin(candidates),
        ["event_id", "poi_idx"],
    ].copy()
    if evaluable.empty:
        raise ValueError("targets contain no evaluable closed-world events")
    expected_events = set(evaluable["event_id"].astype(int))

    prediction_frame = (
        recommendations_to_frame(predictions)
        if isinstance(predictions, Mapping)
        else predictions
    )
    max_k = max(max(cutoffs), int(mrr_k))
    if len(candidates) < max_k:
        raise ValueError("candidate set must contain at least max_k POIs")
    prediction_frame = validate_predictions(
        prediction_frame, expected_events, candidates, max_k
    )
    truth = evaluable.rename(columns={"poi_idx": "target_poi_idx"})
    ranked = prediction_frame.merge(truth, on="event_id", how="left", validate="many_to_one")
    hits = ranked.loc[ranked["poi_idx"].eq(ranked["target_poi_idx"]), ["event_id", "rank"]]
    rank_by_event = hits.set_index("event_id")["rank"].to_dict()
    target_ranks = [rank_by_event.get(event_id) for event_id in evaluable["event_id"]]
    metrics: dict[str, Any] = {
        "Coverage": evaluation_coverage(len(evaluable), len(targets)),
    }
    for k in cutoffs:
        metrics[f"HitRate@{k}"] = hit_rate_at_k(target_ranks, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(target_ranks, k)
    metrics[f"MRR@{mrr_k}"] = mrr_at_k(target_ranks, mrr_k)
    return metrics
