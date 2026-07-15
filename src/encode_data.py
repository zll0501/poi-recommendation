"""Fit train-only ID mappings and encode all dataset partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def validate_encoding_config(config: dict[str, Any]) -> None:
    """Require stable, non-overlapping special-token IDs."""
    if config.get("fit_on") != "train":
        raise ValueError("encoding.fit_on must be 'train' to prevent leakage")
    pad_id = int(config.get("pad_id", 0))
    unknown_id = int(config.get("unknown_id", 1))
    normal_start = int(config.get("normal_id_start", 2))
    if len({pad_id, unknown_id}) != 2:
        raise ValueError("PAD and UNK IDs must be different")
    if normal_start <= max(pad_id, unknown_id):
        raise ValueError("normal_id_start must be greater than special-token IDs")
    vocabularies = config.get("vocabularies")
    if not isinstance(vocabularies, dict) or not vocabularies:
        raise ValueError("encoding.vocabularies must be a non-empty mapping")


def fit_id_mappings(
    train: pd.DataFrame,
    encoding_config: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic vocabularies using training tokens only."""
    validate_encoding_config(encoding_config)
    normal_start = int(encoding_config["normal_id_start"])
    mappings: dict[str, Any] = {
        "version": 1,
        "fitted_on": "train",
        "special_tokens": {
            "PAD": int(encoding_config["pad_id"]),
            "UNK": int(encoding_config["unknown_id"]),
        },
        "normal_id_start": normal_start,
        "vocabularies": {},
    }
    for raw_column, encoded_column in encoding_config["vocabularies"].items():
        if raw_column not in train.columns:
            raise ValueError(f"training data is missing mapping column: {raw_column}")
        tokens = sorted(train[raw_column].dropna().astype("string").unique().tolist())
        token_to_id = {
            token: index for index, token in enumerate(tokens, start=normal_start)
        }
        mappings["vocabularies"][raw_column] = {
            "encoded_column": str(encoded_column),
            "token_count": len(token_to_id),
            "embedding_size": normal_start + len(token_to_id),
            "token_to_id": token_to_id,
        }
    return mappings


def encode_partition(
    frame: pd.DataFrame,
    mappings: dict[str, Any],
) -> pd.DataFrame:
    """Apply frozen mappings and encode unseen or missing tokens as UNK."""
    encoded = frame.copy()
    unknown_id = int(mappings["special_tokens"]["UNK"])
    for raw_column, vocabulary in mappings["vocabularies"].items():
        if raw_column not in encoded.columns:
            raise ValueError(f"partition is missing mapping column: {raw_column}")
        output_column = vocabulary["encoded_column"]
        values = encoded[raw_column].astype("string")
        encoded[output_column] = (
            values.map(vocabulary["token_to_id"])
            .fillna(unknown_id)
            .astype("int64")
        )
    return encoded


def build_poi_metadata(train_encoded: pd.DataFrame) -> pd.DataFrame:
    """Fit one canonical category and location per POI from training only."""
    rows = []
    for poi_idx, group in train_encoded.groupby("poi_idx", sort=True, observed=True):
        category_counts = (
            group.groupby(["category_idx", "category_id", "category_name"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "category_idx"], ascending=[False, True], kind="stable")
        )
        category = category_counts.iloc[0]
        rows.append({
            "poi_id": str(group["poi_id"].iloc[0]),
            "poi_idx": int(poi_idx),
            "category_id": category["category_id"],
            "category_idx": int(category["category_idx"]),
            "category_name": category["category_name"],
            "latitude": float(group["latitude"].median()),
            "longitude": float(group["longitude"].median()),
            "training_visit_count": int(len(group)),
            "category_confidence": float(category["count"] / len(group)),
            "coordinate_observations": int(group[["latitude", "longitude"]].drop_duplicates().shape[0]),
        })
    return pd.DataFrame(rows).sort_values("poi_idx", kind="stable").reset_index(drop=True)


def apply_canonical_poi_metadata(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    unknown_id: int,
) -> pd.DataFrame:
    """Use frozen training metadata for known POIs; never learn from later splits."""
    canonical = metadata.set_index("poi_idx")
    result = frame.copy()
    known = result["poi_idx"].ne(unknown_id)
    for column in (
        "category_id",
        "category_idx",
        "category_name",
        "latitude",
        "longitude",
    ):
        mapped = result.loc[known, "poi_idx"].map(canonical[column])
        result.loc[known, column] = mapped.to_numpy()
    result["category_idx"] = result["category_idx"].astype("int64")
    return result


def build_encoding_report(
    partitions: dict[str, pd.DataFrame],
    mappings: dict[str, Any],
    poi_metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize vocabulary sizes and unseen-token handling by partition."""
    unknown_id = int(mappings["special_tokens"]["UNK"])
    partition_stats: dict[str, Any] = {}
    for name, frame in partitions.items():
        unknown_rows = {}
        for raw_column, vocabulary in mappings["vocabularies"].items():
            encoded_column = vocabulary["encoded_column"]
            unknown_rows[raw_column] = int(frame[encoded_column].eq(unknown_id).sum())
        partition_stats[name] = {
            "rows": int(len(frame)),
            "unknown_rows": unknown_rows,
        }

    vocab_stats = {
        raw_column: {
            "encoded_column": vocabulary["encoded_column"],
            "normal_tokens": int(vocabulary["token_count"]),
            "embedding_size": int(vocabulary["embedding_size"]),
        }
        for raw_column, vocabulary in mappings["vocabularies"].items()
    }
    train_has_unknown = any(
        count > 0
        for count in partition_stats["train"]["unknown_rows"].values()
    )
    return {
        "fitted_on": "train",
        "special_tokens": mappings["special_tokens"],
        "normal_id_start": mappings["normal_id_start"],
        "vocabularies": vocab_stats,
        "partitions": partition_stats,
        "candidate_pois": int(len(poi_metadata)),
        "quality_checks": {
            "training_contains_unknown_ids": train_has_unknown,
            "poi_metadata_covers_all_train_pois": int(len(poi_metadata))
            == vocab_stats["poi_id"]["normal_tokens"],
        },
    }


def render_encoding_markdown(report: dict[str, Any]) -> str:
    """Render the mapping and encoding audit as Markdown."""
    rows = [
        "# Foursquare NYC ID编码报告",
        "",
        "## 编码规则",
        "",
        "- 映射表拟合数据：仅训练集",
        f"- PAD：{report['special_tokens']['PAD']}",
        f"- UNK：{report['special_tokens']['UNK']}",
        f"- 正常ID起点：{report['normal_id_start']}",
        "",
        "## 词表规模",
        "",
        "| 原始字段 | 编码字段 | 正常标记数 | Embedding大小 |",
        "|---|---|---:|---:|",
    ]
    for raw_column, item in report["vocabularies"].items():
        rows.append(
            f"| {raw_column} | {item['encoded_column']} | "
            f"{item['normal_tokens']:,} | {item['embedding_size']:,} |"
        )
    rows.extend(
        [
            "",
            "## 未知标记统计",
            "",
            "| 集合 | 记录数 | 未知用户 | 未知POI | 未知类别 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, display_name in (
        ("train", "训练集"),
        ("validation", "验证集"),
        ("test", "测试集"),
    ):
        item = report["partitions"][name]
        unknown = item["unknown_rows"]
        rows.append(
            f"| {display_name} | {item['rows']:,} | {unknown['user_id']:,} | "
            f"{unknown['poi_id']:,} | {unknown['category_id']:,} |"
        )
    quality = report["quality_checks"]
    rows.extend(
        [
            "",
            "## 质量检查",
            "",
            f"- 候选POI数量：{report['candidate_pois']:,}",
            f"- 训练集是否含UNK：{'是（失败）' if quality['training_contains_unknown_ids'] else '否（通过）'}",
            f"- POI元数据是否覆盖全部训练POI：{'通过' if quality['poi_metadata_covers_all_train_pois'] else '失败'}",
            "",
        ]
    )
    return "\n".join(rows)


def prepare_encoded_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    encoding_config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Fit on train and return encoded partitions, mappings, metadata, and report."""
    mappings = fit_id_mappings(train, encoding_config)
    encoded = {
        "train": encode_partition(train, mappings),
        "validation": encode_partition(validation, mappings),
        "test": encode_partition(test, mappings),
    }
    metadata = build_poi_metadata(encoded["train"])
    unknown_id = int(mappings["special_tokens"]["UNK"])
    partitions = {
        name: apply_canonical_poi_metadata(frame, metadata, unknown_id)
        for name, frame in encoded.items()
    }
    report = build_encoding_report(partitions, mappings, metadata)
    return partitions, mappings, metadata, report


def save_outputs(
    partitions: dict[str, pd.DataFrame],
    mappings: dict[str, Any],
    metadata: pd.DataFrame,
    report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Path]:
    """Persist model-ready data and its reproducibility artifacts."""
    output = config["output"]
    output_dir = resolve_project_path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / output["encoded_train_file"],
        "validation": output_dir / output["encoded_validation_file"],
        "test": output_dir / output["encoded_test_file"],
        "mappings": output_dir / output["mappings_file"],
        "poi_metadata": output_dir / output["poi_metadata_file"],
        "encoding_json": output_dir / output["encoding_json"],
        "encoding_markdown": output_dir / output["encoding_markdown"],
    }
    partitions["train"].to_csv(paths["train"], index=False, encoding="utf-8")
    partitions["validation"].to_csv(
        paths["validation"], index=False, encoding="utf-8"
    )
    partitions["test"].to_csv(paths["test"], index=False, encoding="utf-8")
    paths["mappings"].write_text(
        json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata.to_csv(paths["poi_metadata"], index=False, encoding="utf-8")
    paths["encoding_json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["encoding_markdown"].write_text(
        render_encoding_markdown(report), encoding="utf-8"
    )
    return paths


def run(config_path: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Load chronological splits, fit train-only mappings, encode, and save."""
    config = load_yaml(config_path)
    output = config["output"]
    output_dir = resolve_project_path(output["directory"])
    input_names = {
        "train": output["train_file"],
        "validation": output["validation_file"],
        "test": output["test_file"],
    }
    raw_partitions = {}
    for name, filename in input_names.items():
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Split data not found: {path}. Run src.split_data first."
            )
        raw_partitions[name] = pd.read_csv(
            path,
            dtype={"user_id": "string", "poi_id": "string", "category_id": "string"},
        )
    partitions, mappings, metadata, report = prepare_encoded_data(
        raw_partitions["train"],
        raw_partitions["validation"],
        raw_partitions["test"],
        config["encoding"],
    )
    paths = save_outputs(partitions, mappings, metadata, report, config)
    return partitions, mappings, metadata, report, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train-only ID mappings")
    parser.add_argument("--config", default="configs/data.yaml")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    partitions, mappings, metadata, report, paths = run(args.config)
    print(
        f"Encoded train={len(partitions['train']):,}, "
        f"validation={len(partitions['validation']):,}, "
        f"test={len(partitions['test']):,}"
    )
    print(
        f"users={mappings['vocabularies']['user_id']['token_count']:,}, "
        f"candidate_pois={len(metadata):,}, "
        f"categories={mappings['vocabularies']['category_id']['token_count']:,}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
