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


def test_query_time_variant_uses_batch_shaped_context_and_supports_backward():
    model = SASRec(
        num_pois=10,
        max_seq_len=4,
        hidden_size=8,
        dropout=0.0,
        use_query_time=True,
    )
    inputs = _inputs()
    logits = model(
        poi_sequence=inputs["poi_sequence"],
        attention_mask=inputs["attention_mask"],
        query_hour=torch.tensor([8, 20]),
        query_weekday=torch.tensor([1, 7]),
        query_time_slot=torch.tensor([2, 4]),
    )
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([4, 7]))
    loss.backward()

    assert logits.shape == (2, 10)
    assert model.query_hour_embedding.weight.grad is not None
    assert model.query_weekday_embedding.weight.grad is not None
    assert model.query_time_slot_embedding.weight.grad is not None


def test_query_time_requires_one_value_per_batch_item():
    model = SASRec(
        num_pois=10, max_seq_len=4, hidden_size=8, use_query_time=True
    )
    inputs = _inputs()

    try:
        model(
            inputs["poi_sequence"],
            inputs["attention_mask"],
            query_hour=torch.tensor([[8, 9], [20, 21]]),
            query_weekday=torch.tensor([1, 7]),
            query_time_slot=torch.tensor([2, 4]),
        )
    except ValueError as error:
        assert "shape [B]" in str(error)
    else:
        raise AssertionError("query time must not be expanded to [B, L]")


def test_disabling_query_time_preserves_old_checkpoint_structure_and_output():
    torch.manual_seed(42)
    old_style = SASRec(
        num_pois=10, max_seq_len=4, hidden_size=8, dropout=0.0
    )
    compatible = SASRec(
        num_pois=10,
        max_seq_len=4,
        hidden_size=8,
        dropout=0.0,
        use_query_time=False,
    )
    compatible.load_state_dict(old_style.state_dict(), strict=True)
    inputs = _inputs()

    old_style.eval()
    compatible.eval()
    expected = old_style(inputs["poi_sequence"], inputs["attention_mask"])
    actual = compatible(inputs["poi_sequence"], inputs["attention_mask"])

    assert not any("query_" in key for key in compatible.state_dict())
    assert torch.equal(actual, expected)
