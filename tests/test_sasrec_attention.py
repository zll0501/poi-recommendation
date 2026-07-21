"""Tests for non-invasive SASRec attention extraction."""

import pytest
import torch

from src.layers.attention import AttentionExtractor
from src.models.sasrec import SASRec


def _model_and_inputs():
    torch.manual_seed(42)
    model = SASRec(
        num_pois=12,
        max_seq_len=5,
        hidden_size=8,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()
    poi_sequence = torch.tensor(
        [
            [2, 3, 4, 0, 0],
            [5, 6, 7, 8, 0],
        ]
    )
    attention_mask = poi_sequence.ne(0)
    return model, poi_sequence, attention_mask


def test_hooks_preserve_logits_and_capture_every_layer_and_head():
    model, poi_sequence, attention_mask = _model_and_inputs()
    with torch.no_grad():
        expected = model(poi_sequence, attention_mask)

    extractor = AttentionExtractor().register(model)
    with torch.no_grad():
        actual = model(poi_sequence, attention_mask)
    attention = extractor.get_attention(attention_mask)
    extractor.remove()
    with torch.no_grad():
        restored = model(poi_sequence, attention_mask)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)
    assert set(attention) == {0, 1}
    assert set(attention[0]) == {0, 1}
    assert attention[0][0].shape == (2, 5, 5)


def test_padding_is_zeroed_and_last_real_query_is_selected():
    model, poi_sequence, attention_mask = _model_and_inputs()
    extractor = AttentionExtractor().register(model)
    with torch.no_grad():
        model(poi_sequence, attention_mask)

    full = extractor.get_attention(attention_mask)
    last = extractor.get_last_query_attention(attention_mask)
    extractor.remove()

    for heads in full.values():
        for matrices in heads.values():
            assert torch.count_nonzero(matrices[0, :, 3:]) == 0
            assert torch.count_nonzero(matrices[0, 3:, :]) == 0
            assert torch.count_nonzero(matrices[1, :, 4:]) == 0
            assert torch.count_nonzero(matrices[1, 4:, :]) == 0

    for heads in last.values():
        for sample_weights in heads.values():
            assert sample_weights[0].shape == (3,)
            assert sample_weights[1].shape == (4,)
            torch.testing.assert_close(sample_weights[0].sum(), torch.tensor(1.0))
            torch.testing.assert_close(sample_weights[1].sum(), torch.tensor(1.0))


def test_remove_stops_capture_and_clears_previous_weights():
    model, poi_sequence, attention_mask = _model_and_inputs()
    extractor = AttentionExtractor().register(model)
    with torch.no_grad():
        model(poi_sequence, attention_mask)
    extractor.remove()

    assert not extractor.is_registered
    with pytest.raises(RuntimeError, match="not registered"):
        extractor.get_attention(attention_mask)
