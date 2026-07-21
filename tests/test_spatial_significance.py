import pandas as pd

from experiments.analyze_spatial_significance import spatial_bootstrap


def predictions(ranks: list[int]) -> pd.DataFrame:
    rows = []
    for event_id, target_rank in enumerate(ranks, start=1):
        pois = list(range(100, 110))
        pois[target_rank - 1] = event_id + 10
        rows.extend(
            {"event_id": event_id, "rank": rank, "poi_idx": poi}
            for rank, poi in enumerate(pois, start=1)
        )
    return pd.DataFrame(rows)


def test_spatial_bootstrap_uses_generic_result_labels() -> None:
    targets = pd.DataFrame(
        {"event_id": [1, 2], "user_idx": [2, 3], "poi_idx": [11, 12]}
    )
    result = spatial_bootstrap(
        targets, predictions([10, 10]), predictions([1, 1]), bootstrap_samples=100
    )
    metric = result["metrics"]["NDCG@10"]
    assert result["comparison"] == "distance_minus_unreranked"
    assert metric["distance"] > metric["unreranked"]
    assert "weather" not in metric
