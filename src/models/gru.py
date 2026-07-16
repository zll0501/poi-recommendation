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

    模型词表大小和候选 POI 由数据接口动态提供，不属于静态配置。训练过程
    委托给公共 ``Trainer``，推理结果按事件编号返回 Top-K POI。
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        num_pois: int,
        padding_idx: Optional[int] = 0,
        device: torch.device | str | None = None,
        candidate_poi_ids: Optional[Iterable[int]] = None,
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
        self.candidate_poi_ids = tuple(
            dict.fromkeys(
                int(poi_id)
                for poi_id in (
                    candidate_poi_ids if candidate_poi_ids is not None else ()
                )
            )
        )
        if any(
            poi_id < 0 or poi_id >= self.num_pois
            for poi_id in self.candidate_poi_ids
        ):
            raise ValueError("candidate_poi_ids 包含词表范围外的 POI")
        if self.padding_idx in self.candidate_poi_ids:
            raise ValueError("candidate_poi_ids 不能包含 PAD")
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

    def recommend(
        self,
        test_data: Iterable[Any],
        top_k: int = 10,
    ) -> dict[int, list[int]]:
        """为测试事件生成限定在训练候选集中的 Top-K POI。"""
        if not self.candidate_poi_ids:
            raise ValueError("测试集推理前必须提供 candidate_poi_ids")
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        if top_k > len(self.candidate_poi_ids):
            raise ValueError("top_k 不能大于候选 POI 数量")

        self.network.to(self.device)
        self.network.eval()
        candidate_mask = torch.zeros(
            self.num_pois,
            dtype=torch.bool,
            device=self.device,
        )
        candidate_mask[list(self.candidate_poi_ids)] = True
        recommendations: dict[int, list[int]] = {}

        with torch.inference_mode():
            for batch in test_data:
                if not isinstance(batch, Mapping):
                    raise TypeError("测试批次必须是包含 history/event_id 的字典")
                if "history" not in batch or "event_id" not in batch:
                    raise KeyError("测试批次必须包含 history 和 event_id 字段")

                history = batch["history"].to(self.device)
                logits = self.network(history)
                logits = logits.masked_fill(~candidate_mask.unsqueeze(0), -torch.inf)
                top_poi_ids = torch.topk(logits, k=top_k, dim=1).indices.cpu()
                event_ids = batch["event_id"].cpu().tolist()

                for event_id, poi_ids in zip(event_ids, top_poi_ids.tolist()):
                    event_id = int(event_id)
                    if event_id in recommendations:
                        raise ValueError(f"测试批次包含重复 event_id：{event_id}")
                    recommendations[event_id] = [int(poi_id) for poi_id in poi_ids]

        if not recommendations:
            raise ValueError("test_data 不得为空")
        return recommendations
