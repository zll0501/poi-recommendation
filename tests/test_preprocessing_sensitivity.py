"""Tests for preprocessing sensitivity evaluation helpers."""

import pandas as pd

from experiments.preprocessing_sensitivity import evaluate_popularity


def test_popularity_evaluation_uses_train_candidates_and_time_slots() -> None:
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "poi_id": ["p1", "p1", "p1", "p2", "p2"],
            "time_slot": ["morning", "morning", "evening", "evening", "evening"],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "new_user"],
            "poi_id": ["p1", "p2", "unseen", "p1"],
            "time_slot": ["evening", "evening", "morning", "morning"],
        }
    )

    result = evaluate_popularity(train, test, [1, 2])

    assert result["test_rows"] == 4
    assert result["evaluable_rows"] == 2
    assert result["candidate_coverage"] == 0.5
    assert result["global_acc@1"] == 0.5
    assert result["global_acc@2"] == 1.0
    assert result["time_acc@1"] == 0.5
    assert result["time_acc@2"] == 1.0
    assert result["global_mrr"] == 0.75
