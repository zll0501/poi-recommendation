"""Tests for the deterministic Global Popular baseline."""

from pathlib import Path

import pandas as pd
import pytest

from src.models.popular import GlobalPopular


def make_train() -> pd.DataFrame:
    return pd.DataFrame({"poi_idx": [4, 2, 3, 2, 4, 2, 3]})


def test_global_popular_ranks_by_count_then_poi_id() -> None:
    model = GlobalPopular().fit(make_train())

    assert model.ranking_ == (2, 3, 4)
    assert model.ranking_frame().to_dict(orient="records") == [
        {"rank": 1, "poi_idx": 2, "visit_count": 3},
        {"rank": 2, "poi_idx": 3, "visit_count": 2},
        {"rank": 3, "poi_idx": 4, "visit_count": 2},
    ]


def test_global_popular_returns_shared_prediction_format() -> None:
    predictions = GlobalPopular().fit(make_train()).recommend(
        pd.DataFrame({"event_id": [10, 20]}), top_k=2
    )

    assert predictions.to_dict(orient="records") == [
        {"event_id": 10, "rank": 1, "poi_idx": 2},
        {"event_id": 10, "rank": 2, "poi_idx": 3},
        {"event_id": 20, "rank": 1, "poi_idx": 2},
        {"event_id": 20, "rank": 2, "poi_idx": 3},
    ]


def test_global_popular_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "global_popular.json"
    GlobalPopular().fit(make_train()).save(path)
    restored = GlobalPopular().load(path)

    assert restored.ranking_ == (2, 3, 4)
    assert restored.visit_counts_ == {2: 3, 3: 2, 4: 2}


def test_global_popular_rejects_invalid_usage() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        GlobalPopular().recommend(pd.DataFrame({"event_id": [1]}))
    with pytest.raises(ValueError, match="top_k"):
        GlobalPopular().fit(make_train()).recommend(
            pd.DataFrame({"event_id": [1]}), top_k=4
        )
