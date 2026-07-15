"""Pure ranking metrics for one-target next-POI recommendation."""

from __future__ import annotations

from collections.abc import Iterable
import math


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")


def _clean_ranks(ranks: Iterable[int | None]) -> list[int | None]:
    values = list(ranks)
    for rank in values:
        if rank is not None and (
            not isinstance(rank, int) or isinstance(rank, bool) or rank < 1
        ):
            raise ValueError("ranks must contain positive integers or None")
    return values


def hit_rate_at_k(ranks: Iterable[int | None], k: int) -> float:
    """Fraction of events whose one true POI appears in the first k positions."""
    _validate_k(k)
    values = _clean_ranks(ranks)
    if not values:
        raise ValueError("ranks cannot be empty")
    return sum(rank is not None and rank <= k for rank in values) / len(values)


def ndcg_at_k(ranks: Iterable[int | None], k: int) -> float:
    """Mean NDCG@k for a single relevant POI per event."""
    _validate_k(k)
    values = _clean_ranks(ranks)
    if not values:
        raise ValueError("ranks cannot be empty")
    gains = [
        1.0 / math.log2(rank + 1)
        if rank is not None and rank <= k
        else 0.0
        for rank in values
    ]
    return sum(gains) / len(gains)


def mrr_at_k(ranks: Iterable[int | None], k: int) -> float:
    """Mean truncated reciprocal rank of the one true POI."""
    _validate_k(k)
    values = _clean_ranks(ranks)
    if not values:
        raise ValueError("ranks cannot be empty")
    reciprocal = [
        1.0 / rank if rank is not None and rank <= k else 0.0
        for rank in values
    ]
    return sum(reciprocal) / len(reciprocal)


def evaluation_coverage(evaluable_events: int, total_events: int) -> float:
    """Share of all target events that satisfy the shared closed-world protocol."""
    if total_events < 1:
        raise ValueError("total_events must be positive")
    if not 0 <= evaluable_events <= total_events:
        raise ValueError("evaluable_events must be between zero and total_events")
    return evaluable_events / total_events
