"""供深度推荐模型复用的基础训练器。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class Trainer:
    """执行基础的单设备模型训练与损失验证。

    批次支持 ``(history, target)`` 二元序列，也支持包含
    ``history`` 和 ``target`` 字段的映射对象。

    Args:
        model: 需要训练的 PyTorch 模型。
        optimizer: 用于更新模型参数的优化器。
        loss_fn: 接收模型输出和目标值的损失函数。
        device: 模型和批次数据所在的计算设备。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable[[Any, Any], Tensor],
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.model.to(self.device)

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[Any, Any]:
        """从受支持的批次格式中提取历史序列和目标值。"""
        if isinstance(batch, Mapping):
            try:
                return batch["history"], batch["target"]
            except KeyError as exc:
                raise KeyError(
                    "字典批次必须包含 'history' 和 'target' 字段"
                ) from exc

        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]

        raise TypeError(
            "批次必须是 (history, target) 或包含 history/target 的字典"
        )

    def _prepare_batch(self, batch: Any) -> tuple[Any, Any]:
        """解析批次，并将输入和目标移动到训练设备。"""
        history, target = self._unpack_batch(batch)
        try:
            return history.to(self.device), target.to(self.device)
        except AttributeError as exc:
            raise TypeError("history 和 target 必须支持 to(device) 操作") from exc

    def train_one_epoch(self, train_loader: Iterable[Any]) -> float:
        """训练一个轮次并返回各批次的平均损失。"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            history, target = self._prepare_batch(batch)
            self.optimizer.zero_grad()
            output = self.model(history)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().item())
            batch_count += 1

        if batch_count == 0:
            raise ValueError("train_loader 不得为空")
        return total_loss / batch_count

    def evaluate_loss(self, valid_loader: Iterable[Any]) -> float:
        """在不更新参数的情况下计算验证集平均损失。"""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch in valid_loader:
                history, target = self._prepare_batch(batch)
                output = self.model(history)
                loss = self.loss_fn(output, target)

                total_loss += float(loss.item())
                batch_count += 1

        if batch_count == 0:
            raise ValueError("valid_loader 不得为空")
        return total_loss / batch_count

    def fit(
        self,
        train_loader: Iterable[Any],
        valid_loader: Iterable[Any] | None = None,
        epochs: int = 50,
    ) -> dict[str, list[float]]:
        """执行指定轮数的训练，并返回训练与验证损失历史。"""
        if epochs <= 0:
            raise ValueError("epochs 必须为正整数")

        history: dict[str, list[float]] = {
            "train_loss": [],
            "valid_loss": [],
        }
        for _ in range(epochs):
            history["train_loss"].append(self.train_one_epoch(train_loader))
            if valid_loader is not None:
                history["valid_loss"].append(self.evaluate_loss(valid_loader))

        return history
