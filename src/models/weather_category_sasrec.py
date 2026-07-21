"""Weather residual wrapper around the existing Category-SASRec model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.layers.weather import WeatherCategoryScorer, WeatherEncoder


class WeatherCategorySASRec(nn.Module):
    """Add a weak query-weather/category residual without changing SASRec.

    The wrapped SASRec remains responsible for user preference and POI-level
    ranking. Weather only adjusts the scores of known POI categories.
    """

    def __init__(
        self,
        base_model: nn.Module,
        weather_encoder: WeatherEncoder,
        weather_scorer: WeatherCategoryScorer,
        poi_category_idx: Tensor,
    ) -> None:
        super().__init__()
        category_embedding = getattr(base_model, "category_embedding", None)
        if not isinstance(category_embedding, nn.Embedding):
            raise ValueError(
                "base_model must be a category-enabled SASRec with category_embedding"
            )
        if poi_category_idx.ndim != 1:
            raise ValueError("poi_category_idx must have shape [num_pois]")
        num_pois = int(getattr(base_model, "num_pois", -1))
        if num_pois < 1 or poi_category_idx.numel() != num_pois:
            raise ValueError("poi_category_idx must cover the complete POI vocabulary")
        category_ids = poi_category_idx.detach().to(dtype=torch.long, device="cpu")
        if category_ids.numel() and (
            category_ids.min().item() < 0
            or category_ids.max().item() >= category_embedding.num_embeddings
        ):
            raise ValueError("poi_category_idx contains an invalid category ID")
        if weather_encoder.output_dim != category_embedding.embedding_dim:
            raise ValueError(
                "weather output dimension must equal the category embedding dimension"
            )

        self.base_model = base_model
        self.weather_encoder = weather_encoder
        self.weather_scorer = weather_scorer
        self.register_buffer("poi_category_idx", category_ids, persistent=True)

    @property
    def weather_weight(self) -> Tensor:
        return self.weather_scorer.weather_weight

    def forward(
        self,
        poi_sequence: Tensor,
        attention_mask: Tensor,
        time_sequence: Tensor | None = None,
        category_sequence: Tensor | None = None,
        query_hour: Tensor | None = None,
        query_weekday: Tensor | None = None,
        query_time_slot: Tensor | None = None,
        weather_group_idx: Tensor | None = None,
        weather_numeric: Tensor | None = None,
    ) -> Tensor:
        if weather_group_idx is None or weather_numeric is None:
            raise ValueError(
                "weather_group_idx and weather_numeric are required for weather fusion"
            )
        base_logits = self.base_model(
            poi_sequence=poi_sequence,
            attention_mask=attention_mask,
            time_sequence=time_sequence,
            category_sequence=category_sequence,
            query_hour=query_hour,
            query_weekday=query_weekday,
            query_time_slot=query_time_slot,
        )
        weather_state = self.weather_encoder(weather_group_idx, weather_numeric)
        category_embeddings = self.base_model.category_embedding(
            self.poi_category_idx
        )
        weather_logits = self.weather_scorer(weather_state, category_embeddings)
        if weather_logits.shape != base_logits.shape:
            raise RuntimeError("weather and SASRec logits are not aligned")
        fused_logits = base_logits + weather_logits
        if not torch.isfinite(fused_logits).all():
            raise FloatingPointError("non-finite weather-aware SASRec logits")
        return fused_logits
