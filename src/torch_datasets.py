"""将公共下一 POI 数据接口适配为 PyTorch Dataset。"""

from __future__ import annotations

from torch import Tensor, long, tensor
from torch.utils.data import Dataset

from src.datasets import POIDataBundle, PartitionName


TIME_SLOT_TO_INDEX = {
    "night": 1,
    "morning": 2,
    "afternoon": 3,
    "evening": 4,
}


class NextPOITorchDataset(Dataset):
    """为序列推荐模型提供定长历史序列和下一 POI 目标。

    历史序列保留最近的 ``max_history`` 个 POI，并在左侧使用 PAD 编号
    补齐。左侧补齐可以保证当前 GRU 取到的最终隐藏状态对应最后一个真实
    POI，而不是补齐位置。

    Args:
        data_bundle: 公共数据接口返回的数据集合。
        partition: 需要构建的分区，可选 ``train``、``validation`` 或
            ``test``。
        max_history: 每条历史序列的固定长度。
        include_unknown_targets: 是否保留目标 POI 为 UNK 的样本。
        include_unknown_users: 是否保留用户编号为 UNK 的样本。
        include_metadata: 是否在样本中附带用于测试集评价的 ``event_id``。
    """

    def __init__(
        self,
        data_bundle: POIDataBundle,
        partition: PartitionName,
        max_history: int,
        include_unknown_targets: bool = False,
        include_unknown_users: bool = False,
        include_metadata: bool = False,
    ) -> None:
        if max_history <= 0:
            raise ValueError("max_history 必须为正整数")

        self.partition = partition
        self.max_history = max_history
        self.padding_idx = data_bundle.pad_id
        self.num_pois = data_bundle.vocabulary_size("poi_id")
        self.include_metadata = include_metadata

        samples = list(
            data_bundle.iter_next_poi_samples(
                partition,
                max_history=max_history,
                include_unknown_targets=include_unknown_targets,
                include_unknown_users=include_unknown_users,
            )
        )
        self.histories = tensor(
            [
                [self.padding_idx] * (max_history - len(sample.history))
                + list(sample.history)
                for sample in samples
            ],
            dtype=long,
        )
        self.targets = tensor(
            [sample.target_poi_idx for sample in samples],
            dtype=long,
        )
        self.event_ids = tensor(
            [sample.event_id for sample in samples],
            dtype=long,
        )

        if not samples:
            self.histories = self.histories.reshape(0, max_history)

    def __len__(self) -> int:
        """返回当前数据分区中的可用下一 POI 样本数。"""
        return int(self.targets.size(0))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | dict[str, Tensor]:
        """返回训练二元组，或附带事件编号的评价字典。"""
        if self.include_metadata:
            return {
                "history": self.histories[index],
                "target": self.targets[index],
                "event_id": self.event_ids[index],
            }
        return self.histories[index], self.targets[index]


class TimeAwareNextPOIDataset(Dataset):
    """提供历史 POI 和已知推荐请求时间的下一 POI 数据集。

    ``hour``、``weekday`` 和 ``time_slot`` 都来自目标事件的请求时间，
    不包含目标 POI 或目标类别信息。索引 ``0`` 保留给未启用或补齐状态，
    因此小时和星期分别在原始编号基础上加一。
    """

    def __init__(
        self,
        data_bundle: POIDataBundle,
        partition: PartitionName,
        max_history: int,
        include_unknown_targets: bool = False,
        include_unknown_users: bool = False,
    ) -> None:
        if max_history <= 0:
            raise ValueError("max_history 必须为正整数")

        self.partition = partition
        self.max_history = max_history
        self.padding_idx = data_bundle.pad_id
        self.num_pois = data_bundle.vocabulary_size("poi_id")

        samples = list(
            data_bundle.iter_next_poi_samples(
                partition,
                max_history=max_history,
                include_unknown_targets=include_unknown_targets,
                include_unknown_users=include_unknown_users,
            )
        )
        self.histories = tensor(
            [
                [self.padding_idx] * (max_history - len(sample.history))
                + list(sample.history)
                for sample in samples
            ],
            dtype=long,
        )
        self.hours = tensor([sample.hour + 1 for sample in samples], dtype=long)
        self.weekdays = tensor(
            [sample.weekday + 1 for sample in samples],
            dtype=long,
        )
        try:
            time_slot_indices = [
                TIME_SLOT_TO_INDEX[sample.time_slot] for sample in samples
            ]
        except KeyError as exc:
            raise ValueError(f"未知时间段：{exc.args[0]}") from exc
        self.time_slots = tensor(time_slot_indices, dtype=long)
        self.targets = tensor(
            [sample.target_poi_idx for sample in samples],
            dtype=long,
        )
        self.event_ids = tensor(
            [sample.event_id for sample in samples],
            dtype=long,
        )

        if not samples:
            self.histories = self.histories.reshape(0, max_history)

    def __len__(self) -> int:
        """返回当前数据分区中的可用下一 POI 样本数。"""
        return int(self.targets.size(0))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """返回 Time-GRU 训练和评价所需的一个样本。"""
        return {
            "history": self.histories[index],
            "hour": self.hours[index],
            "weekday": self.weekdays[index],
            "time_slot": self.time_slots[index],
            "target": self.targets[index],
            "event_id": self.event_ids[index],
        }
