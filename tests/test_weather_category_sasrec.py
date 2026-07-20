import pytest
import torch

from src.layers.weather import WeatherCategoryScorer, WeatherEncoder
from src.models.sasrec import SASRec
from src.models.weather_category_sasrec import WeatherCategorySASRec


def _model() -> WeatherCategorySASRec:
    base = SASRec(
        num_pois=8,
        num_categories=6,
        max_seq_len=3,
        hidden_size=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        use_query_time=True,
        use_category=True,
    )
    return WeatherCategorySASRec(
        base,
        WeatherEncoder(4, 3, 8, dropout=0.0),
        WeatherCategoryScorer(0.1),
        torch.tensor([0, 1, 2, 2, 3, 3, 4, 5]),
    )


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "poi_sequence": torch.tensor([[2, 3, 0], [4, 5, 6]]),
        "attention_mask": torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
        "category_sequence": torch.tensor([[2, 2, 0], [3, 3, 4]]),
        "query_hour": torch.tensor([9, 18]),
        "query_weekday": torch.tensor([1, 5]),
        "query_time_slot": torch.tensor([2, 4]),
        "weather_group_idx": torch.tensor([1, 2]),
        "weather_numeric": torch.randn(2, 3),
    }


def test_weather_wrapper_returns_full_logits_and_backpropagates() -> None:
    model = _model()
    logits = model(**_inputs())
    assert logits.shape == (2, 8)
    logits.sum().backward()
    assert model.weather_scorer.weight_logit.grad is not None
    assert model.weather_encoder.fusion[0].weight.grad is not None
    assert model.base_model.poi_embedding.weight.grad is not None


def test_weather_wrapper_requires_weather_and_matching_dimensions() -> None:
    model = _model()
    inputs = _inputs()
    del inputs["weather_numeric"]
    with pytest.raises(ValueError, match="required"):
        model(**inputs)

    base = SASRec(
        num_pois=8,
        num_categories=6,
        max_seq_len=3,
        hidden_size=8,
        num_heads=2,
        num_layers=1,
        use_category=True,
    )
    with pytest.raises(ValueError, match="dimension"):
        WeatherCategorySASRec(
            base,
            WeatherEncoder(4, 3, 4),
            WeatherCategoryScorer(),
            torch.tensor([0, 1, 2, 2, 3, 3, 4, 5]),
        )
