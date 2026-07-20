"""SASRec 专用训练器与闭集候选集推理工具。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class SASRecTrainer:
    """在不修改成员 2 公共 Trainer 的前提下训练多输入 SASRec。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.model.to(self.device)

    def _move_to_device(self, value: Any) -> Any:
        # 递归处理 inputs 字典，保证所有嵌套张量位于同一计算设备。
        if isinstance(value, Tensor):
            return value.to(self.device)
        if isinstance(value, Mapping):
            return {
                key: self._move_to_device(item) for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        return value

    def _prepare_batch(self, batch: Mapping[str, Any]) -> tuple[dict[str, Tensor], Tensor]:
        if "inputs" not in batch or "target" not in batch:
            raise KeyError("SASRec batch must contain 'inputs' and 'target'")
        inputs = self._move_to_device(batch["inputs"])
        target = self._move_to_device(batch["target"])
        if not isinstance(inputs, Mapping) or not isinstance(target, Tensor):
            raise TypeError("invalid SASRec inputs or target")
        return dict(inputs), target

    def train_one_epoch(self, train_loader: Iterable[Mapping[str, Any]]) -> float:
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            inputs, target = self._prepare_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(**inputs)
            loss = self.loss_fn(logits, target)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach().item())
            batch_count += 1
        if batch_count == 0:
            raise ValueError("train_loader cannot be empty")
        return total_loss / batch_count

    def evaluate_loss(self, valid_loader: Iterable[Mapping[str, Any]]) -> float:
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, target = self._prepare_batch(batch)
                loss = self.loss_fn(self.model(**inputs), target)
                total_loss += float(loss.item())
                batch_count += 1
        if batch_count == 0:
            raise ValueError("valid_loader cannot be empty")
        return total_loss / batch_count

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        valid_loader: Iterable[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 50,
        patience: int | None = None,
        epoch_evaluator: Callable[[int], Mapping[str, float]] | None = None,
        selection_metric: str = "valid_loss",
        maximize_selection_metric: bool = False,
    ) -> dict[str, Any]:
        """训练模型，并在结束时恢复指定验证指标对应的最佳参数。"""
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if patience is not None and patience < 1:
            raise ValueError("patience must be positive or None")

        history: dict[str, Any] = {
            "train_loss": [],
            "valid_loss": [],
            "epochs": [],
        }
        best_state: dict[str, Tensor] | None = None
        best_score = float("-inf") if maximize_selection_metric else float("inf")
        stale_epochs = 0
        for epoch_index in range(epochs):
            epoch_number = epoch_index + 1
            train_loss = self.train_one_epoch(train_loader)
            history["train_loss"].append(train_loss)
            epoch_record: dict[str, float | int] = {
                "epoch": epoch_number,
                "train_loss": train_loss,
            }
            if valid_loader is None:
                history["epochs"].append(epoch_record)
                continue
            valid_loss = self.evaluate_loss(valid_loader)
            history["valid_loss"].append(valid_loss)
            epoch_record["valid_loss"] = valid_loss
            if epoch_evaluator is not None:
                epoch_record.update(
                    {
                        key: float(value)
                        for key, value in epoch_evaluator(epoch_number).items()
                    }
                )
            history["epochs"].append(epoch_record)
            if selection_metric not in epoch_record:
                raise KeyError(
                    f"selection metric {selection_metric!r} is missing from epoch record"
                )
            score = float(epoch_record[selection_metric])
            improved = (
                score > best_score if maximize_selection_metric else score < best_score
            )
            if improved:
                best_score = score
                best_state = deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if patience is not None and stale_epochs >= patience:
                    break

        if best_state is not None:
            # 测试必须使用验证集选择的最佳状态，而不是最后一轮状态。
            self.model.load_state_dict(best_state)
        return history

    def predict(
        self,
        data_loader: Iterable[Mapping[str, Any]],
        candidate_poi_ids: Sequence[int],
        *,
        top_k: int = 10,
    ) -> dict[int, list[int]]:
        """仅在训练集冻结的 POI 候选集合中为每个事件生成排序。"""
        candidates = tuple(dict.fromkeys(int(value) for value in candidate_poi_ids))
        if top_k < 1 or len(candidates) < top_k:
            raise ValueError("candidate set must contain at least top_k POIs")
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=self.device)

        recommendations: dict[int, list[int]] = {}
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if "event_id" not in batch:
                    raise KeyError("prediction batch must contain 'event_id'")
                inputs, _ = self._prepare_batch(batch)
                logits = self.model(**inputs)
                # PAD、UNK 和训练集外 POI 不在 candidate_poi_ids 中，不会被推荐。
                candidate_logits = logits.index_select(1, candidate_tensor)
                positions = candidate_logits.topk(top_k, dim=1).indices.cpu()
                ranked_pois = candidate_tensor[positions.to(self.device)].cpu().tolist()
                event_ids = batch["event_id"].tolist()
                for event_id, poi_ids in zip(event_ids, ranked_pois):
                    event_id = int(event_id)
                    if event_id in recommendations:
                        raise ValueError(f"duplicate prediction event_id: {event_id}")
                    recommendations[event_id] = [int(value) for value in poi_ids]
        return recommendations
