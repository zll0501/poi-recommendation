"""为 SASRec 构造严格对齐的 POI、时间与类别序列。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.datasets import POIDataBundle


PartitionName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SASRecSample:
    """一条下一 POI 样本，以及预测时刻之前可见的对齐历史。"""

    user_idx: int
    poi_sequence: tuple[int, ...]
    time_sequence: tuple[int, ...]
    category_sequence: tuple[int, ...]
    target_poi_idx: int
    target_category_idx: int
    target_time_idx: int
    event_id: int


class SASRecDataset(Dataset[SASRecSample]):
    """按照公共时间协议生成无未来信息泄漏的滚动历史。"""

    def __init__(
        self,
        data: POIDataBundle,
        partition: PartitionName,
        max_seq_len: int,
        *,
        include_unknown_targets: bool = False,
        include_unknown_users: bool = False,
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = int(max_seq_len)
        self.samples = self._build_samples(
            data,
            partition,
            include_unknown_targets=include_unknown_targets,
            include_unknown_users=include_unknown_users,
        )

    def _build_samples(
        self,
        data: POIDataBundle,
        partition: PartitionName,
        *,
        include_unknown_targets: bool,
        include_unknown_users: bool,
    ) -> list[SASRecSample]:
        if partition == "train":
            prior_frames: list[pd.DataFrame] = []
            target_frame = data.train
        elif partition == "validation":
            prior_frames = [data.train]
            target_frame = data.validation
        elif partition == "test":
            prior_frames = [data.train, data.validation]
            target_frame = data.test
        else:
            raise ValueError(f"unknown partition: {partition}")

        histories: dict[str, dict[str, list[int]]] = {}
        if prior_frames:
            # 验证集继承训练历史；测试集继承训练集和验证集历史。
            prior = pd.concat(prior_frames, ignore_index=True).sort_values(
                ["user_id", "utc_time"], kind="stable"
            )
            for row in prior.itertuples(index=False):
                history = histories.setdefault(
                    str(row.user_id), {"pois": [], "times": [], "categories": []}
                )
                history["pois"].append(int(row.poi_idx))
                history["times"].append(int(row.hour) + 1)
                history["categories"].append(int(row.category_idx))

        samples: list[SASRecSample] = []
        for row in target_frame.itertuples(index=False):
            # 使用原始 user_id 作为键，避免不同冷启动用户因共享 UNK 而串联历史。
            history = histories.setdefault(
                str(row.user_id), {"pois": [], "times": [], "categories": []}
            )
            user_idx = int(row.user_idx)
            target_poi_idx = int(row.poi_idx)
            if (
                history["pois"]
                and (include_unknown_users or user_idx != data.unknown_id)
                and (include_unknown_targets or target_poi_idx != data.unknown_id)
            ):
                # 三类特征必须采用完全相同的截断起点，确保位置一一对应。
                start = max(0, len(history["pois"]) - self.max_seq_len)
                poi_sequence = tuple(history["pois"][start:])
                time_sequence = tuple(history["times"][start:])
                category_sequence = tuple(history["categories"][start:])
                if not (
                    len(poi_sequence)
                    == len(time_sequence)
                    == len(category_sequence)
                ):
                    raise RuntimeError("SASRec history features are not aligned")
                samples.append(
                    SASRecSample(
                        user_idx=user_idx,
                        poi_sequence=poi_sequence,
                        time_sequence=time_sequence,
                        category_sequence=category_sequence,
                        target_poi_idx=target_poi_idx,
                        target_category_idx=int(row.category_idx),
                        target_time_idx=int(row.hour) + 1,
                        event_id=int(row.event_id),
                    )
                )

            # 先产出当前预测样本，再加入真实目标，防止目标泄漏到自身输入。
            history["pois"].append(target_poi_idx)
            history["times"].append(int(row.hour) + 1)
            history["categories"].append(int(row.category_idx))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SASRecSample:
        return self.samples[index]


class SASRecCollator:
    """右侧补齐样本，并转换成独立 Trainer 使用的多输入批次。"""

    def __init__(
        self,
        max_seq_len: int,
        pad_id: int = 0,
        *,
        use_time: bool = True,
        use_category: bool = True,
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = int(max_seq_len)
        self.pad_id = int(pad_id)
        self.use_time = bool(use_time)
        self.use_category = bool(use_category)

    def _right_pad(self, values: tuple[int, ...]) -> list[int]:
        # 右侧补齐确保 PAD query 在因果约束下仍可看到之前的真实事件。
        values = values[-self.max_seq_len :]
        return list(values) + [self.pad_id] * (self.max_seq_len - len(values))

    def __call__(self, samples: list[SASRecSample]) -> dict[str, object]:
        if not samples:
            raise ValueError("cannot collate an empty sample list")

        poi_sequence = torch.tensor(
            [self._right_pad(sample.poi_sequence) for sample in samples],
            dtype=torch.long,
        )
        inputs: dict[str, Tensor] = {
            "poi_sequence": poi_sequence,
            # True 表示真实事件，False 表示 PAD；模型内部会转成屏蔽掩码。
            "attention_mask": poi_sequence.ne(self.pad_id),
        }
        if self.use_time:
            inputs["time_sequence"] = torch.tensor(
                [self._right_pad(sample.time_sequence) for sample in samples],
                dtype=torch.long,
            )
        if self.use_category:
            inputs["category_sequence"] = torch.tensor(
                [self._right_pad(sample.category_sequence) for sample in samples],
                dtype=torch.long,
            )

        return {
            "inputs": inputs,
            "target": torch.tensor(
                [sample.target_poi_idx for sample in samples], dtype=torch.long
            ),
            "event_id": torch.tensor(
                [sample.event_id for sample in samples], dtype=torch.long
            ),
        }
