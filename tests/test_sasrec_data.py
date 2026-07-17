"""验证 SASRec 的 POI、时间和类别历史始终严格对齐。"""

import pandas as pd

from src.datasets import POIDataBundle
from src.sasrec_data import SASRecCollator, SASRecDataset


def _frame(event_ids, poi_ids, categories, hours):
    size = len(event_ids)
    return pd.DataFrame(
        {
            "event_id": event_ids,
            "user_id": ["u1"] * size,
            "user_idx": [2] * size,
            "poi_idx": poi_ids,
            "category_idx": categories,
            "utc_time": pd.date_range("2024-01-01", periods=size, freq="h", tz="UTC")
            + pd.Timedelta(hours=event_ids[0]),
            "hour": hours,
            "weekday": [0] * size,
            "time_slot": ["morning"] * size,
        }
    )


def _bundle():
    mappings = {
        "special_tokens": {"PAD": 0, "UNK": 1},
        "vocabularies": {
            "user_id": {"embedding_size": 3},
            "poi_id": {"embedding_size": 10},
            "category_id": {"embedding_size": 8},
        },
    }
    return POIDataBundle(
        train=_frame([0, 1, 2], [2, 3, 4], [2, 3, 4], [0, 1, 23]),
        validation=_frame([3], [5], [5], [8]),
        test=_frame([4], [6], [6], [9]),
        mappings=mappings,
        poi_metadata=pd.DataFrame({"poi_idx": [2, 3, 4, 5, 6]}),
    )


def test_sequences_are_aligned_and_hours_reserve_zero_for_padding():
    dataset = SASRecDataset(_bundle(), "train", max_seq_len=2)

    assert len(dataset) == 2
    assert dataset[0].poi_sequence == (2,)
    assert dataset[0].category_sequence == (2,)
    assert dataset[0].time_sequence == (1,)
    assert dataset[1].poi_sequence == (2, 3)
    assert dataset[1].time_sequence == (1, 2)
    assert dataset[1].target_time_idx == 24


def test_validation_and_test_include_previous_partition_histories():
    data = _bundle()
    validation = SASRecDataset(data, "validation", max_seq_len=10)
    test = SASRecDataset(data, "test", max_seq_len=10)

    assert validation[0].poi_sequence == (2, 3, 4)
    assert test[0].poi_sequence == (2, 3, 4, 5)
    assert test[0].category_sequence == (2, 3, 4, 5)


def test_collator_right_pads_all_enabled_features_consistently():
    samples = list(SASRecDataset(_bundle(), "train", max_seq_len=2))
    batch = SASRecCollator(max_seq_len=3, pad_id=0)(samples)
    inputs = batch["inputs"]

    assert inputs["poi_sequence"].tolist() == [[2, 0, 0], [2, 3, 0]]
    assert inputs["time_sequence"].tolist() == [[1, 0, 0], [1, 2, 0]]
    assert inputs["category_sequence"].tolist() == [[2, 0, 0], [2, 3, 0]]
    assert inputs["attention_mask"].tolist() == [
        [True, False, False],
        [True, True, False],
    ]
    assert batch["target"].tolist() == [3, 4]
    assert batch["event_id"].tolist() == [1, 2]
