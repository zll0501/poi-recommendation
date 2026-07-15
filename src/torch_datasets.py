"""将公共下一 POI 数据接口适配为 PyTorch Dataset。"""

from __future__ import annotations

from torch import Tensor, long, tensor
from torch.utils.data import Dataset

from src.datasets import POIDataBundle, PartitionName


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
        self.targets = tensor(
            [sample.target_poi_idx for sample in samples],
            dtype=long,
        )

        if not samples:
            self.histories = self.histories.reshape(0, max_history)

    def __len__(self) -> int:
        """返回当前数据分区中的可用下一 POI 样本数。"""
        return int(self.targets.size(0))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """返回 ``(history, target)`` 张量，供 DataLoader 自动组批。"""
        return self.histories[index], self.targets[index]
