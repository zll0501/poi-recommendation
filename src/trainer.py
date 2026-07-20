"""供深度推荐模型复用的基础训练器。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


BatchForward = Callable[[nn.Module, Any, torch.device], tuple[Tensor, Tensor]]


class Trainer:
    """执行基础的单设备模型训练与损失验证。

    批次支持 ``(history, target)`` 二元序列，也支持包含
    ``history`` 和 ``target`` 字段的映射对象。

    Args:
        model: 需要训练的 PyTorch 模型。
        optimizer: 用于更新模型参数的优化器。
        loss_fn: 接收模型输出和目标值的损失函数。
        device: 模型和批次数据所在的计算设备。
        batch_forward: 可选的自定义批次前向函数，用于多输入模型。
        early_stopping: 是否根据验证损失提前停止训练。
        patience: 验证损失连续多少轮未改善后停止。
        min_delta: 判定验证损失改善所需的最小下降量。
        restore_best_weights: 是否在训练结束后恢复最佳轮次权重。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable[[Any, Any], Tensor],
        device: torch.device | str,
        batch_forward: BatchForward | None = None,
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
    ) -> None:
        if not isinstance(early_stopping, bool):
            raise TypeError("early_stopping 必须是布尔值")
        if not isinstance(restore_best_weights, bool):
            raise TypeError("restore_best_weights 必须是布尔值")
        if patience <= 0:
            raise ValueError("patience 必须为正整数")
        if min_delta < 0:
            raise ValueError("min_delta 不能为负数")

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.batch_forward = batch_forward
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_epoch: int | None = None
        self.best_valid_loss: float | None = None
        self.stopped_early = False
        self.trained_epochs = 0
        self.model.to(self.device)

    @property
    def training_summary(self) -> dict[str, int | float | bool | None]:
        """返回最近一次训练的 Early Stopping 摘要。"""
        return {
            "early_stopping_enabled": self.early_stopping,
            "best_epoch": self.best_epoch,
            "best_valid_loss": self.best_valid_loss,
            "stopped_early": self.stopped_early,
            "trained_epochs": self.trained_epochs,
        }

    def _copy_model_state(self) -> dict[str, Tensor]:
        """将当前模型权重复制到 CPU，避免长期占用额外显存。"""
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.state_dict().items()
        }

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

    def _forward_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        """使用默认单输入路径或自定义函数完成一次批次前向传播。"""
        if self.batch_forward is not None:
            return self.batch_forward(self.model, batch, self.device)

        history, target = self._prepare_batch(batch)
        return self.model(history), target

    def train_one_epoch(self, train_loader: Iterable[Any]) -> float:
        """训练一个轮次并返回各批次的平均损失。"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            self.optimizer.zero_grad()
            output, target = self._forward_batch(batch)
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
                output, target = self._forward_batch(batch)
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
        if self.early_stopping and valid_loader is None:
            raise ValueError("启用 Early Stopping 时必须提供 valid_loader")

        history: dict[str, list[float]] = {
            "train_loss": [],
            "valid_loss": [],
        }
        self.best_epoch = None
        self.best_valid_loss = None
        self.stopped_early = False
        self.trained_epochs = 0
        best_state: dict[str, Tensor] | None = None
        epochs_without_improvement = 0

        for epoch_index in range(epochs):
            history["train_loss"].append(self.train_one_epoch(train_loader))
            if valid_loader is not None:
                valid_loss = self.evaluate_loss(valid_loader)
                history["valid_loss"].append(valid_loss)
                if (
                    self.best_valid_loss is None
                    or valid_loss < self.best_valid_loss - self.min_delta
                ):
                    self.best_valid_loss = valid_loss
                    self.best_epoch = epoch_index + 1
                    epochs_without_improvement = 0
                    if self.early_stopping and self.restore_best_weights:
                        best_state = self._copy_model_state()
                else:
                    epochs_without_improvement += 1

            self.trained_epochs = epoch_index + 1
            if (
                self.early_stopping
                and epochs_without_improvement >= self.patience
            ):
                self.stopped_early = True
                break

        if (
            self.early_stopping
            and self.restore_best_weights
            and best_state is not None
        ):
            self.model.load_state_dict(best_state)

        return history
