"""融合已知请求时间上下文的 GRU 下一 POI 推荐模型。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.optim import Adam

from src.models.base import BaseRecommender
from src.trainer import Trainer


class TimeGRUNetwork(nn.Module):
    """使用请求小时、时间段和星期信息增强 GRU 隐藏状态。"""

    def __init__(
        self,
        num_pois: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        padding_idx: Optional[int] = 0,
        use_hour: bool = False,
        use_time_slot: bool = False,
        use_weekday: bool = False,
    ) -> None:
        super().__init__()
        if num_pois <= 0:
            raise ValueError("num_pois 必须为正整数")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须为正整数")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim 必须为正整数")
        if num_layers <= 0:
            raise ValueError("num_layers 必须为正整数")

        self.use_hour = use_hour
        self.use_time_slot = use_time_slot
        self.use_weekday = use_weekday
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
        # 在时间 Embedding 前初始化，保证共享层在相同 seed 下与基础 GRU 一致。
        self.output_layer = nn.Linear(hidden_dim, num_pois)
        self.hour_embedding = (
            nn.Embedding(25, hidden_dim, padding_idx=0) if use_hour else None
        )
        self.time_slot_embedding = (
            nn.Embedding(5, hidden_dim, padding_idx=0) if use_time_slot else None
        )
        self.weekday_embedding = (
            nn.Embedding(8, hidden_dim, padding_idx=0) if use_weekday else None
        )

    @staticmethod
    def _validate_context(
        name: str,
        value: Optional[Tensor],
        batch_size: int,
    ) -> Tensor:
        """检查请求时间特征是否是一维批次张量。"""
        if value is None:
            raise ValueError(f"启用 {name} 时必须提供 {name}")
        if value.ndim != 1 or value.size(0) != batch_size:
            raise ValueError(f"{name} 必须具有形状 [batch_size]")
        return value

    def forward(
        self,
        history: Tensor,
        hour: Optional[Tensor] = None,
        time_slot: Optional[Tensor] = None,
        weekday: Optional[Tensor] = None,
    ) -> Tensor:
        """返回形状为 ``[batch_size, num_pois]`` 的下一 POI logits。"""
        if history.ndim != 2:
            raise ValueError("history 必须具有形状 [batch_size, sequence_length]")
        if history.size(1) == 0:
            raise ValueError("history 序列不得为空")

        embedded_history = self.poi_embedding(history)
        _, hidden_state = self.gru(embedded_history)
        combined_hidden = hidden_state[-1]
        batch_size = history.size(0)

        if self.use_hour:
            hour = self._validate_context("hour", hour, batch_size)
            assert self.hour_embedding is not None
            combined_hidden = combined_hidden + self.hour_embedding(hour)
        if self.use_time_slot:
            time_slot = self._validate_context("time_slot", time_slot, batch_size)
            assert self.time_slot_embedding is not None
            combined_hidden = combined_hidden + self.time_slot_embedding(time_slot)
        if self.use_weekday:
            weekday = self._validate_context("weekday", weekday, batch_size)
            assert self.weekday_embedding is not None
            combined_hidden = combined_hidden + self.weekday_embedding(weekday)

        return self.output_layer(combined_hidden)


def time_gru_batch_forward(
    model: nn.Module,
    batch: Any,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """将 Time-GRU 字典批次移动到设备并执行模型前向传播。"""
    if not isinstance(batch, Mapping):
        raise TypeError("Time-GRU 批次必须是字典")
    required = {"history", "hour", "time_slot", "weekday", "target"}
    missing = sorted(required.difference(batch))
    if missing:
        raise KeyError(f"Time-GRU 批次缺少字段：{missing}")

    output = model(
        history=batch["history"].to(device),
        hour=batch["hour"].to(device),
        time_slot=batch["time_slot"].to(device),
        weekday=batch["weekday"].to(device),
    )
    return output, batch["target"].to(device)


class TimeGRURecommender(BaseRecommender):
    """负责 Time-GRU 训练、候选集过滤和 Top-K 推理。"""

    def __init__(
        self,
        config: Mapping[str, Any],
        num_pois: int,
        padding_idx: Optional[int] = 0,
        device: torch.device | str | None = None,
        candidate_poi_ids: Optional[Iterable[int]] = None,
    ) -> None:
        self.config = dict(config)
        model_config = self._config_section("model")
        time_config = self._config_section("time_features")
        self.training_config = {
            **self.config,
            **self._config_section("training"),
        }

        master_time_switch = self._bool_config(model_config, "use_time", True)
        self.use_hour = master_time_switch and self._bool_config(
            time_config, "hour", False
        )
        self.use_time_slot = master_time_switch and self._bool_config(
            time_config, "time_slot", False
        )
        self.use_weekday = master_time_switch and self._bool_config(
            time_config, "weekday", False
        )

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
        self.network = TimeGRUNetwork(
            num_pois=self.num_pois,
            embedding_dim=int(model_config.get("embedding_dim", 128)),
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            num_layers=int(model_config.get("num_layers", 1)),
            dropout=float(model_config.get("dropout", 0.2)),
            padding_idx=self.padding_idx,
            use_hour=self.use_hour,
            use_time_slot=self.use_time_slot,
            use_weekday=self.use_weekday,
        )
        self.trainer: Optional[Trainer] = None
        self.training_history: Optional[dict[str, list[float]]] = None
        self.training_summary: Optional[
            dict[str, int | float | bool | None]
        ] = None

    def _config_section(self, name: str) -> Mapping[str, Any]:
        section = self.config.get(name, {})
        if not isinstance(section, Mapping):
            raise TypeError(f"config['{name}'] 必须是映射对象")
        return section

    @staticmethod
    def _bool_config(
        section: Mapping[str, Any],
        name: str,
        default: bool,
    ) -> bool:
        value = section.get(name, default)
        if not isinstance(value, bool):
            raise TypeError(f"配置字段 '{name}' 必须是布尔值")
        return value

    @property
    def model_name(self) -> str:
        """根据启用的请求时间特征返回实验名称。"""
        features = []
        if self.use_hour:
            features.append("Hour")
        if self.use_time_slot:
            features.append("TimeSlot")
        if self.use_weekday:
            features.append("Weekday")
        return "GRU" if not features else "GRU+" + "+".join(features)

    def fit(
        self,
        train_data: Iterable[Any],
        valid_data: Optional[Iterable[Any]] = None,
    ) -> dict[str, list[float]]:
        """使用公共训练循环拟合 Time-GRU。"""
        optimizer = Adam(
            self.network.parameters(),
            lr=float(self.training_config.get("learning_rate", 0.001)),
            weight_decay=float(self.training_config.get("weight_decay", 0.0001)),
        )
        early_stopping_config = self.training_config.get("early_stopping", {})
        if not isinstance(early_stopping_config, Mapping):
            raise TypeError("training.early_stopping 必须是映射对象")
        self.trainer = Trainer(
            model=self.network,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            device=self.device,
            batch_forward=time_gru_batch_forward,
            early_stopping=early_stopping_config.get("enabled", False),
            patience=int(early_stopping_config.get("patience", 5)),
            min_delta=float(early_stopping_config.get("min_delta", 0.001)),
            restore_best_weights=early_stopping_config.get(
                "restore_best_weights", True
            ),
        )
        self.training_history = self.trainer.fit(
            train_loader=train_data,
            valid_loader=valid_data,
            epochs=int(self.training_config.get("epochs", 50)),
        )
        self.training_summary = self.trainer.training_summary
        return self.training_history

    def recommend(
        self,
        test_data: Iterable[Any],
        top_k: int = 10,
    ) -> dict[int, list[int]]:
        """生成限定在训练 POI 候选集中的 Top-K 推荐。"""
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
                    raise TypeError("Time-GRU 测试批次必须是字典")
                required = {"history", "hour", "time_slot", "weekday", "event_id"}
                missing = sorted(required.difference(batch))
                if missing:
                    raise KeyError(f"Time-GRU 测试批次缺少字段：{missing}")

                logits = self.network(
                    history=batch["history"].to(self.device),
                    hour=batch["hour"].to(self.device),
                    time_slot=batch["time_slot"].to(self.device),
                    weekday=batch["weekday"].to(self.device),
                )
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
