"""用于下一 POI 预测的 SASRec 及其时间、类别特征变体。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SASRec(nn.Module):
    """支持可选时间与 POI 类别嵌入的自注意力序列模型。"""

    def __init__(
        self,
        *,
        num_pois: int,
        max_seq_len: int,
        hidden_size: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0,
        num_time_tokens: int = 25,
        num_categories: int | None = None,
        use_time: bool = False,
        use_category: bool = False,
    ) -> None:
        super().__init__()
        if num_pois < 2 or max_seq_len < 1 or hidden_size < 1:
            raise ValueError("num_pois, max_seq_len, and hidden_size are invalid")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if use_category and (num_categories is None or num_categories < 2):
            raise ValueError("num_categories is required when use_category is true")

        self.num_pois = int(num_pois)
        self.max_seq_len = int(max_seq_len)
        self.pad_id = int(pad_id)
        self.use_time = bool(use_time)
        self.use_category = bool(use_category)

        self.poi_embedding = nn.Embedding(
            self.num_pois, hidden_size, padding_idx=self.pad_id
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, hidden_size)
        self.time_embedding = (
            # 小时编码为 1～24，0 专门保留给 PAD。
            nn.Embedding(num_time_tokens, hidden_size, padding_idx=self.pad_id)
            if self.use_time
            else None
        )
        self.category_embedding = (
            nn.Embedding(
                int(num_categories), hidden_size, padding_idx=self.pad_id
            )
            if self.use_category
            else None
        )
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.poi_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        if self.time_embedding is not None:
            nn.init.normal_(self.time_embedding.weight, std=0.02)
        if self.category_embedding is not None:
            nn.init.normal_(self.category_embedding.weight, std=0.02)
        with torch.no_grad():
            self.poi_embedding.weight[self.pad_id].zero_()
            if self.time_embedding is not None:
                self.time_embedding.weight[self.pad_id].zero_()
            if self.category_embedding is not None:
                self.category_embedding.weight[self.pad_id].zero_()

    def forward(
        self,
        poi_sequence: Tensor,
        attention_mask: Tensor,
        time_sequence: Tensor | None = None,
        category_sequence: Tensor | None = None,
    ) -> Tensor:
        """为每条历史输出完整 POI 词表上的下一地点 logits。"""
        if poi_sequence.ndim != 2 or attention_mask.shape != poi_sequence.shape:
            raise ValueError("poi_sequence and attention_mask must have shape [B, L]")
        if poi_sequence.size(1) > self.max_seq_len:
            raise ValueError("sequence length exceeds configured max_seq_len")
        attention_mask = attention_mask.bool()
        if not attention_mask.any(dim=1).all():
            raise ValueError("every sequence must contain at least one real event")
        if (
            attention_mask.size(1) > 1
            and ((~attention_mask[:, :-1]) & attention_mask[:, 1:]).any()
        ):
            raise ValueError("attention_mask must describe right-padded sequences")

        # 位置只按真实事件递增，右侧 PAD 不占用新的有效位置。
        position_ids = attention_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        hidden = self.poi_embedding(poi_sequence) + self.position_embedding(
            position_ids
        )

        if self.use_time:
            if time_sequence is None or time_sequence.shape != poi_sequence.shape:
                raise ValueError("aligned time_sequence is required")
            hidden = hidden + self.time_embedding(time_sequence)
        if self.use_category:
            if (
                category_sequence is None
                or category_sequence.shape != poi_sequence.shape
            ):
                raise ValueError("aligned category_sequence is required")
            hidden = hidden + self.category_embedding(category_sequence)

        hidden = self.input_dropout(self.input_norm(hidden))
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("non-finite hidden values before encoder")
        length = poi_sequence.size(1)
        # 上三角为 True，阻止任一位置看到它之后的未来行为。
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=poi_sequence.device),
            diagonal=1,
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            # PyTorch 的 padding mask 中 True 表示需要屏蔽，和输入协议相反。
            src_key_padding_mask=~attention_mask,
        )
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("non-finite hidden values after encoder")

        # 右侧 Padding 后，根据每行真实长度提取最近一次历史事件。
        last_positions = attention_mask.long().sum(dim=1).sub(1)
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = self.output_norm(hidden[batch_indices, last_positions])
        # 与 POI Embedding 共享输出权重，返回原始 logits，不提前做 softmax。
        logits = F.linear(last_hidden, self.poi_embedding.weight)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite SASRec logits")
        return logits
