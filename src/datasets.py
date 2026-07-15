"""Unified, model-agnostic data interface for every project member."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import pandas as pd

from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PartitionName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class NextPOISample:
    """One chronological next-POI target with only its available history."""

    user_idx: int
    history: tuple[int, ...]
    target_poi_idx: int
    target_category_idx: int
    event_id: int
    timestamp: pd.Timestamp
    hour: int
    weekday: int
    time_slot: str


@dataclass
class POIDataBundle:
    """All shared model inputs produced by the data owner."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    mappings: dict[str, Any]
    poi_metadata: pd.DataFrame

    def __post_init__(self) -> None:
        required = {
            "event_id",
            "user_id",
            "user_idx",
            "poi_idx",
            "category_idx",
            "utc_time",
            "hour",
            "weekday",
            "time_slot",
        }
        for name in ("train", "validation", "test"):
            frame = getattr(self, name)
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"{name} partition is missing columns: {missing}")
            frame["utc_time"] = pd.to_datetime(frame["utc_time"], utc=True)
            setattr(
                self,
                name,
                frame.sort_values(["user_idx", "utc_time"], kind="stable").reset_index(
                    drop=True
                ),
            )

    @property
    def pad_id(self) -> int:
        return int(self.mappings["special_tokens"]["PAD"])

    @property
    def unknown_id(self) -> int:
        return int(self.mappings["special_tokens"]["UNK"])

    @property
    def candidate_poi_ids(self) -> tuple[int, ...]:
        """Return the frozen train-POI candidate universe."""
        return tuple(self.poi_metadata["poi_idx"].astype(int).tolist())

    def vocabulary_size(self, raw_column: str) -> int:
        """Return embedding table size, including PAD and UNK."""
        return int(self.mappings["vocabularies"][raw_column]["embedding_size"])

    def partition(self, name: PartitionName) -> pd.DataFrame:
        """Access a partition without inventing model-specific copies."""
        return getattr(self, name)

    def user_sequences(
        self,
        name: PartitionName = "train",
        include_previous_partitions: bool = False,
    ) -> dict[int, list[int]]:
        """Return ordered POI-index sequences keyed by encoded user ID."""
        frames = [self.partition(name)]
        if include_previous_partitions and name == "validation":
            frames = [self.train, self.validation]
        elif include_previous_partitions and name == "test":
            frames = [self.train, self.validation, self.test]
        data = pd.concat(frames, ignore_index=True).sort_values(
            ["user_idx", "utc_time"], kind="stable"
        )
        data = data.loc[data["user_idx"].ne(self.unknown_id)]
        return {
            int(user_idx): group["poi_idx"].astype(int).tolist()
            for user_idx, group in data.groupby("user_idx", sort=False, observed=True)
        }

    def iter_next_poi_samples(
        self,
        name: PartitionName,
        max_history: int | None = None,
        include_unknown_targets: bool = False,
        include_unknown_users: bool = False,
    ) -> Iterator[NextPOISample]:
        """Yield rolling targets without copying all histories into memory."""
        if max_history is not None and max_history < 1:
            raise ValueError("max_history must be positive or None")

        prior_frames: list[pd.DataFrame]
        target_frame: pd.DataFrame
        if name == "train":
            prior_frames = []
            target_frame = self.train
        elif name == "validation":
            prior_frames = [self.train]
            target_frame = self.validation
        elif name == "test":
            prior_frames = [self.train, self.validation]
            target_frame = self.test
        else:
            raise ValueError(f"unknown partition: {name}")

        # Distinct cold-start users all have user_idx=UNK, so raw user_id must
        # remain the history key to prevent their trajectories being merged.
        histories: dict[str, list[int]] = {}
        if prior_frames:
            prior = pd.concat(prior_frames, ignore_index=True).sort_values(
                ["user_idx", "utc_time"], kind="stable"
            )
            histories = {
                str(user_id): group["poi_idx"].astype(int).tolist()
                for user_id, group in prior.groupby(
                    "user_id", sort=False, observed=True
                )
            }

        for row in target_frame.itertuples(index=False):
            user_idx = int(row.user_idx)
            history = histories.setdefault(str(row.user_id), [])
            target = int(row.poi_idx)
            if (
                history
                and (include_unknown_users or user_idx != self.unknown_id)
                and (include_unknown_targets or target != self.unknown_id)
            ):
                visible_history = history[-max_history:] if max_history else history
                yield NextPOISample(
                    user_idx=user_idx,
                    history=tuple(visible_history),
                    target_poi_idx=target,
                    target_category_idx=int(row.category_idx),
                    event_id=int(row.event_id),
                    timestamp=row.utc_time,
                    hour=int(row.hour),
                    weekday=int(row.weekday),
                    time_slot=str(row.time_slot),
                )
            history.append(target)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_data_bundle(
    config_path: str | Path = "configs/data.yaml",
) -> POIDataBundle:
    """Load the one canonical dataset interface used by all models."""
    config = load_yaml(config_path)
    output = config["output"]
    output_dir = _resolve_project_path(output["directory"])
    partition_keys = {
        "train": "encoded_train_file",
        "validation": "encoded_validation_file",
        "test": "encoded_test_file",
    }
    paths = {
        "train": output_dir / output[partition_keys["train"]],
        "validation": output_dir / output[partition_keys["validation"]],
        "test": output_dir / output[partition_keys["test"]],
        "mappings": output_dir / output["mappings_file"],
        "poi_metadata": output_dir / output["poi_metadata_file"],
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model-ready data is missing. Run src.encode_data first: "
            + ", ".join(missing)
        )
    read_types = {
        "user_id": "string",
        "poi_id": "string",
        "category_id": "string",
        "user_idx": "int64",
        "poi_idx": "int64",
        "category_idx": "int64",
    }
    return POIDataBundle(
        train=pd.read_csv(paths["train"], dtype=read_types),
        validation=pd.read_csv(paths["validation"], dtype=read_types),
        test=pd.read_csv(paths["test"], dtype=read_types),
        mappings=json.loads(paths["mappings"].read_text(encoding="utf-8")),
        poi_metadata=pd.read_csv(
            paths["poi_metadata"],
            dtype={"poi_id": "string", "category_id": "string"},
        ),
    )
