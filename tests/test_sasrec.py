"""验证基础、时间与类别 SASRec 的前向和反向传播。"""

import torch

from src.models.sasrec import SASRec


def _inputs():
    return {
        "poi_sequence": torch.tensor([[2, 3, 0, 0], [4, 5, 6, 0]]),
        "time_sequence": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
        "category_sequence": torch.tensor([[2, 3, 0, 0], [4, 5, 6, 0]]),
        "attention_mask": torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        ),
    }


def test_base_sasrec_returns_full_poi_logits():
    model = SASRec(num_pois=10, max_seq_len=4, hidden_size=8, dropout=0.0)
    inputs = _inputs()
    logits = model(inputs["poi_sequence"], inputs["attention_mask"])

    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_time_and_category_variant_supports_backward_pass():
    model = SASRec(
        num_pois=10,
        num_categories=8,
        max_seq_len=4,
        hidden_size=8,
        dropout=0.0,
        use_time=True,
        use_category=True,
    )
    logits = model(**_inputs())
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([4, 7]))
    loss.backward()

    assert torch.isfinite(loss)
    assert model.poi_embedding.weight.grad is not None
    assert model.time_embedding.weight.grad is not None
    assert model.category_embedding.weight.grad is not None


def test_enabled_features_are_required():
    model = SASRec(
        num_pois=10,
        num_categories=8,
        max_seq_len=4,
        hidden_size=8,
        use_time=True,
        use_category=True,
    )
    inputs = _inputs()

    try:
        model(inputs["poi_sequence"], inputs["attention_mask"])
    except ValueError as error:
        assert "time_sequence" in str(error)
    else:
        raise AssertionError("enabled time input should be required")


def test_left_padded_input_is_rejected():
    model = SASRec(num_pois=10, max_seq_len=4, hidden_size=8)
    poi_sequence = torch.tensor([[0, 0, 2, 3]])
    attention_mask = torch.tensor([[False, False, True, True]])

    try:
        model(poi_sequence, attention_mask)
    except ValueError as error:
        assert "right-padded" in str(error)
    else:
        raise AssertionError("left-padded input should be rejected")
