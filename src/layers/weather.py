"""Small weather encoder and category-aware residual scorer."""

from __future__ import annotations

import math

import torch
from torch import nn


class WeatherEncoder(nn.Module):
    """Encode categorical and continuous weather into a configurable vector."""

    def __init__(
        self,
        num_weather_groups: int,
        numeric_dim: int,
        output_dim: int,
        weather_embedding_dim: int = 8,
        numeric_hidden_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dimensions = {
            "num_weather_groups": num_weather_groups,
            "numeric_dim": numeric_dim,
            "output_dim": output_dim,
            "weather_embedding_dim": weather_embedding_dim,
            "numeric_hidden_dim": numeric_hidden_dim,
        }
        if any(value < 1 for value in dimensions.values()):
            raise ValueError("all WeatherEncoder dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.num_weather_groups = int(num_weather_groups)
        self.numeric_dim = int(numeric_dim)
        self.output_dim = int(output_dim)
        self.group_embedding = nn.Embedding(
            num_weather_groups, weather_embedding_dim, padding_idx=0
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_dim, numeric_hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(weather_embedding_dim + numeric_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        nn.init.normal_(self.group_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.group_embedding.weight[0].zero_()

    def forward(
        self,
        weather_group_idx: torch.Tensor,
        weather_numeric: torch.Tensor,
    ) -> torch.Tensor:
        if weather_group_idx.ndim != 1:
            raise ValueError("weather_group_idx must have shape [batch]")
        if weather_numeric.ndim != 2 or weather_numeric.shape[1] != self.numeric_dim:
            raise ValueError(
                f"weather_numeric must have shape [batch, {self.numeric_dim}]"
            )
        if weather_group_idx.shape[0] != weather_numeric.shape[0]:
            raise ValueError("weather categorical and numeric batch sizes must match")
        if weather_group_idx.numel() and (
            weather_group_idx.min().item() < 0
            or weather_group_idx.max().item() >= self.num_weather_groups
        ):
            raise ValueError("weather_group_idx is outside the configured vocabulary")
        group_state = self.group_embedding(weather_group_idx.long())
        numeric_state = self.numeric_projection(weather_numeric.float())
        return self.fusion(torch.cat([group_state, numeric_state], dim=-1))


class WeatherCategoryScorer(nn.Module):
    """Return a small weather-category residual score for candidate POIs."""

    def __init__(self, initial_weight: float = 0.1) -> None:
        super().__init__()
        if not 0 < initial_weight < 1:
            raise ValueError("initial_weight must be between 0 and 1")
        initial_logit = math.log(initial_weight / (1 - initial_weight))
        self.weight_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    @property
    def weather_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.weight_logit)

    def forward(
        self,
        weather_state: torch.Tensor,
        candidate_category_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if weather_state.ndim != 2:
            raise ValueError("weather_state must have shape [batch, dim]")
        if candidate_category_embeddings.ndim not in {2, 3}:
            raise ValueError("candidate category embeddings must be [candidates, dim] or [batch, candidates, dim]")
        if weather_state.shape[-1] != candidate_category_embeddings.shape[-1]:
            raise ValueError("weather and category embedding dimensions must match")
        if candidate_category_embeddings.ndim == 2:
            scores = weather_state @ candidate_category_embeddings.transpose(0, 1)
        else:
            if weather_state.shape[0] != candidate_category_embeddings.shape[0]:
                raise ValueError("batched category embeddings must match weather batch size")
            scores = torch.einsum("bd,bcd->bc", weather_state, candidate_category_embeddings)
        return self.weather_weight * scores / math.sqrt(weather_state.shape[-1])
