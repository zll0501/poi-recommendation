import pandas as pd
import pytest

from experiments.analyze_weather_significance import paired_user_bootstrap


def _predictions(first_rank: list[int]) -> pd.DataFrame:
    rows = []
    for event_id, target_rank in enumerate(first_rank, start=1):
        ranking = list(range(100, 110))
        ranking[target_rank - 1] = event_id + 10
        rows.extend(
            {"event_id": event_id, "rank": rank, "poi_idx": poi}
            for rank, poi in enumerate(ranking, start=1)
        )
    return pd.DataFrame(rows)


def test_paired_bootstrap_reports_positive_weather_delta() -> None:
    targets = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "user_idx": [2, 2, 3, 3],
            "poi_idx": [11, 12, 13, 14],
        }
    )
    baseline = _predictions([10, 10, 10, 10])
    weather = _predictions([1, 1, 1, 1])
    result = paired_user_bootstrap(
        targets, baseline, weather, bootstrap_samples=100, seed=7
    )
    assert result["events"] == 4
    assert result["users"] == 2
    assert result["metrics"]["NDCG@10"]["absolute_delta"] > 0


def test_paired_bootstrap_requires_matching_events() -> None:
    targets = pd.DataFrame(
        {"event_id": [1, 2], "user_idx": [2, 3], "poi_idx": [11, 12]}
    )
    with pytest.raises(ValueError, match="identical"):
        paired_user_bootstrap(
            targets,
            _predictions([1, 1]),
            _predictions([1]),
            bootstrap_samples=100,
        )
