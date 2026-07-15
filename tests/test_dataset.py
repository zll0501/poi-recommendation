"""Tests for the shared rolling next-POI interface."""

import pandas as pd

from src.datasets import POIDataBundle


def make_frame(event_ids: list[int], poi_ids: list[int]) -> pd.DataFrame:
    size = len(event_ids)
    return pd.DataFrame(
        {
            "event_id": event_ids,
            "user_id": ["u1"] * size,
            "user_idx": [2] * size,
            "poi_idx": poi_ids,
            "category_idx": [2] * size,
            "utc_time": pd.date_range(
                "2024-01-01", periods=size, freq="h", tz="UTC"
            )
            + pd.Timedelta(hours=event_ids[0]),
            "hour": [8] * size,
            "weekday": [0] * size,
            "time_slot": ["morning"] * size,
        }
    )


def make_bundle() -> POIDataBundle:
    mappings = {
        "special_tokens": {"PAD": 0, "UNK": 1},
        "vocabularies": {
            "user_id": {"embedding_size": 3},
            "poi_id": {"embedding_size": 8},
            "category_id": {"embedding_size": 3},
        },
    }
    metadata = pd.DataFrame({"poi_idx": [2, 3, 4, 5, 6, 7]})
    return POIDataBundle(
        train=make_frame([0, 1, 2], [2, 3, 4]),
        validation=make_frame([3, 4], [5, 1]),
        test=make_frame([5, 6], [6, 7]),
        mappings=mappings,
        poi_metadata=metadata,
    )


def test_training_and_rolling_evaluation_histories() -> None:
    bundle = make_bundle()

    train_samples = list(bundle.iter_next_poi_samples("train"))
    validation_samples = list(bundle.iter_next_poi_samples("validation"))
    test_samples = list(bundle.iter_next_poi_samples("test"))

    assert [sample.target_poi_idx for sample in train_samples] == [3, 4]
    assert train_samples[0].history == (2,)
    assert [sample.target_poi_idx for sample in validation_samples] == [5]
    assert validation_samples[0].history == (2, 3, 4)
    assert test_samples[0].history == (2, 3, 4, 5, 1)
    assert test_samples[1].history == (2, 3, 4, 5, 1, 6)
    assert bundle.candidate_poi_ids == (2, 3, 4, 5, 6, 7)


def test_max_history_keeps_only_most_recent_events() -> None:
    sample = next(make_bundle().iter_next_poi_samples("test", max_history=3))
    assert sample.history == (4, 5, 1)


def test_unknown_users_do_not_share_a_history() -> None:
    bundle = make_bundle()
    bundle.validation = pd.DataFrame({
        "event_id": [20, 21, 22, 23],
        "user_id": ["new_a", "new_b", "new_a", "new_b"],
        "user_idx": [1, 1, 1, 1],
        "poi_idx": [2, 3, 4, 5],
        "category_idx": [2, 2, 2, 2],
        "utc_time": pd.date_range("2024-02-01", periods=4, tz="UTC"),
        "hour": [0, 1, 2, 3],
        "weekday": [3, 3, 3, 3],
        "time_slot": ["night"] * 4,
    })
    bundle.__post_init__()

    assert list(bundle.iter_next_poi_samples("validation")) == []
    samples = list(
        bundle.iter_next_poi_samples("validation", include_unknown_users=True)
    )
    assert [sample.history for sample in samples] == [(2,), (3,)]
