"""GRU-based sequential recommender model."""

from collections.abc import Iterable
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.optim import Adam

from src.models.base import BaseRecommender
from src.trainer import Trainer


class GRUNetwork(nn.Module):
    """Encode POI histories with a GRU and predict the next POI.

    Args:
        num_pois: Number of POIs in the model vocabulary.
        embedding_dim: Size of each POI embedding.
        hidden_dim: Size of the GRU hidden state.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers. PyTorch applies it only when
            ``num_layers`` is greater than one.
        padding_idx: Optional vocabulary index used for sequence padding.
    """

    def __init__(
        self,
        num_pois: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        if num_pois <= 0:
            raise ValueError("num_pois must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.poi_embedding = nn.Embedding(
            num_embeddings=num_pois,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, num_pois)

    def forward(self, history: Tensor) -> Tensor:
        """Return next-POI logits for a batch of POI histories.

        Args:
            history: Integer POI indices with shape ``[batch_size,
                sequence_length]``. Each sequence must contain at least one
                item.

        Returns:
            Unnormalized logits with shape ``[batch_size, num_pois]``.
        """
        if history.ndim != 2:
            raise ValueError(
                "history must have shape [batch_size, sequence_length]"
            )
        if history.size(1) == 0:
            raise ValueError("history sequences must contain at least one POI")

        embedded_history = self.poi_embedding(history)
        _, hidden_state = self.gru(embedded_history)
        final_hidden_state = hidden_state[-1]
        return self.output_layer(final_hidden_state)


class GRURecommender(BaseRecommender):
    """GRU 下一 POI 推荐模型封装。

    模型词表大小由数据接口动态提供，不属于静态配置。训练过程委托给公共
    ``Trainer``，当前推荐接口仍等待后续推理流程实现。
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        num_pois: int,
        padding_idx: Optional[int] = 0,
        device: torch.device | str | None = None,
    ) -> None:
        """使用模型配置和数据接口提供的词表信息初始化推荐器。"""
        self.config = dict(config)
        nested_model_config = self.config.get("model", {})
        if not isinstance(nested_model_config, Mapping):
            raise TypeError("config['model'] must be a mapping when provided")
        model_config = {**self.config, **nested_model_config}

        nested_training_config = self.config.get("training", {})
        if not isinstance(nested_training_config, Mapping):
            raise TypeError("config['training'] must be a mapping when provided")
        self.training_config = {**self.config, **nested_training_config}

        self.num_pois = int(num_pois)
        self.padding_idx = padding_idx
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.network = GRUNetwork(
            num_pois=self.num_pois,
            embedding_dim=int(model_config.get("embedding_dim", 128)),
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            num_layers=int(model_config.get("num_layers", 1)),
            dropout=float(model_config.get("dropout", 0.2)),
            padding_idx=self.padding_idx,
        )
        self.trainer: Optional[Trainer] = None
        self.training_history: Optional[dict[str, list[float]]] = None

    def fit(
        self,
        train_data: Iterable[Any],
        valid_data: Optional[Iterable[Any]] = None,
    ) -> dict[str, list[float]]:
        """使用已组批的训练数据调用公共训练器完成模型训练。"""
        optimizer = Adam(
            self.network.parameters(),
            lr=float(self.training_config.get("learning_rate", 0.001)),
            weight_decay=float(self.training_config.get("weight_decay", 0.0001)),
        )
        loss_fn = nn.CrossEntropyLoss()
        self.trainer = Trainer(
            model=self.network,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=self.device,
        )
        self.training_history = self.trainer.fit(
            train_loader=train_data,
            valid_loader=valid_data,
            epochs=int(self.training_config.get("epochs", 50)),
        )
        return self.training_history

    def recommend(self, test_data: Any, top_k: int = 10) -> Any:
        """Generate Top-K recommendations; inference is not implemented yet."""
        raise NotImplementedError("GRU recommendation is not implemented yet")
