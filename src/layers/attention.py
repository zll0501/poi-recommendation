"""Non-invasive attention extraction utilities for Transformer encoders."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


AttentionByHead = dict[int, Tensor]
AttentionByLayer = dict[int, AttentionByHead]
LastQueryAttention = dict[int, dict[int, tuple[Tensor, ...]]]


class AttentionExtractor:
    """Capture per-layer, per-head self-attention without changing the model.

    ``nn.TransformerEncoderLayer`` normally calls ``MultiheadAttention`` with
    ``need_weights=False``.  The hook leaves that prediction call untouched,
    then performs a detached second call solely to obtain per-head weights.
    """

    def __init__(self) -> None:
        self._handles: list[RemovableHandle] = []
        self._captured: dict[int, Tensor] = {}
        self._model: nn.Module | None = None
        self._recomputing_layers: set[int] = set()

    @property
    def is_registered(self) -> bool:
        return bool(self._handles)

    def register(self, model: nn.Module) -> "AttentionExtractor":
        """Register hooks on every ``model.encoder.layers[*].self_attn``."""
        self.remove()
        if model.training:
            raise ValueError("attention extraction requires model.eval()")
        encoder = getattr(model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None or len(layers) == 0:
            raise TypeError("model must expose non-empty encoder.layers")

        self._model = model
        for layer_index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            if not isinstance(attention, nn.MultiheadAttention):
                self.remove()
                raise TypeError(
                    f"encoder layer {layer_index} has no MultiheadAttention self_attn"
                )
            self._handles.append(
                attention.register_forward_hook(
                    self._capture_hook(layer_index),
                    with_kwargs=True,
                )
            )
        return self

    def _capture_hook(self, layer_index: int):
        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: tuple[Tensor, Tensor | None],
        ) -> None:
            del output
            # Keep the model's original need_weights=False call untouched.  A
            # second, detached call computes weights only for interpretation.
            if layer_index in self._recomputing_layers:
                return
            updated = dict(kwargs)
            updated["need_weights"] = True
            updated["average_attn_weights"] = False
            self._recomputing_layers.add(layer_index)
            try:
                with torch.no_grad():
                    recomputed = module(*args, **updated)
            finally:
                self._recomputing_layers.remove(layer_index)
            if (
                not isinstance(recomputed, tuple)
                or len(recomputed) < 2
                or recomputed[1] is None
            ):
                raise RuntimeError(
                    f"attention weights were not returned for encoder layer {layer_index}"
                )
            weights = recomputed[1]
            if weights.ndim != 4:
                raise RuntimeError(
                    "expected attention weights with shape [B, H, L, L], "
                    f"got {tuple(weights.shape)}"
                )
            self._captured[layer_index] = weights.detach()

        return hook

    def clear(self) -> None:
        """Discard weights from the previous forward pass, keeping hooks active."""
        self._captured.clear()

    def remove(self) -> None:
        """Remove all hooks and discard captured tensors."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured.clear()
        self._model = None
        self._recomputing_layers.clear()

    def _validate_mask(self, attention_mask: Tensor) -> Tensor:
        if not self.is_registered:
            raise RuntimeError("AttentionExtractor is not registered")
        if not self._captured:
            raise RuntimeError("no attention captured; run a model forward pass first")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B, L]")
        mask = attention_mask.detach().bool()
        if not mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one real position")
        first = next(iter(self._captured.values()))
        if first.shape[0] != mask.shape[0] or first.shape[-1] != mask.shape[1]:
            raise ValueError("attention_mask does not match captured attention shape")
        return mask.to(first.device)

    def get_attention(self, attention_mask: Tensor) -> AttentionByLayer:
        """Return full attention matrices with all PAD query/key entries zeroed.

        The returned tensors have shape ``[B, L, L]`` for each layer and head.
        """
        mask = self._validate_mask(attention_mask)
        valid_pairs = mask[:, None, :, None] & mask[:, None, None, :]
        result: AttentionByLayer = {}
        for layer_index, weights in sorted(self._captured.items()):
            filtered = weights.masked_fill(~valid_pairs, 0.0).cpu()
            result[layer_index] = {
                head_index: filtered[:, head_index].clone()
                for head_index in range(filtered.size(1))
            }
        return result

    def get_last_query_attention(
        self, attention_mask: Tensor
    ) -> LastQueryAttention:
        """Return the last real query's weights over real history keys only.

        Each head maps to one variable-length tensor per batch sample.  PAD keys
        are sliced away rather than merely assigned a zero value.
        """
        mask = self._validate_mask(attention_mask)
        lengths = mask.long().sum(dim=1)
        last_positions = lengths - 1
        batch_indices = torch.arange(mask.size(0), device=mask.device)
        result: LastQueryAttention = {}
        for layer_index, weights in sorted(self._captured.items()):
            layer_result: dict[int, tuple[Tensor, ...]] = {}
            for head_index in range(weights.size(1)):
                last_rows = weights[
                    batch_indices,
                    head_index,
                    last_positions,
                    :,
                ]
                layer_result[head_index] = tuple(
                    last_rows[index, : int(length)].detach().cpu().clone()
                    for index, length in enumerate(lengths.tolist())
                )
            result[layer_index] = layer_result
        return result

    def __enter__(self) -> "AttentionExtractor":
        if not self.is_registered:
            raise RuntimeError("register(model) must be called before entering context")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.remove()
