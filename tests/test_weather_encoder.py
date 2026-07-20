import pytest
import torch

from src.layers.weather import WeatherCategoryScorer, WeatherEncoder


def test_weather_encoder_output_shape() -> None:
    encoder = WeatherEncoder(
        num_weather_groups=6,
        numeric_dim=5,
        output_dim=32,
        dropout=0.0,
    )
    output = encoder(
        torch.tensor([1, 2, 0]),
        torch.randn(3, 5),
    )
    assert output.shape == (3, 32)
    assert torch.isfinite(output).all()


def test_weather_encoder_rejects_bad_inputs() -> None:
    encoder = WeatherEncoder(6, 5, 32)
    with pytest.raises(ValueError, match="batch sizes"):
        encoder(torch.tensor([1, 2]), torch.randn(3, 5))
    with pytest.raises(ValueError, match="vocabulary"):
        encoder(torch.tensor([6]), torch.randn(1, 5))


def test_weather_category_scorer_shapes_and_initial_weight() -> None:
    scorer = WeatherCategoryScorer(initial_weight=0.1)
    weather = torch.randn(4, 16)
    categories = torch.randn(10, 16)
    scores = scorer(weather, categories)
    assert scores.shape == (4, 10)
    assert scorer.weather_weight.item() == pytest.approx(0.1, abs=1e-6)


def test_weather_category_scorer_supports_batched_candidates() -> None:
    scorer = WeatherCategoryScorer()
    scores = scorer(torch.randn(4, 16), torch.randn(4, 10, 16))
    assert scores.shape == (4, 10)
