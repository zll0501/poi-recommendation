"""Global chronological splitting for reproducible next-POI evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"user_id", "poi_id", "utc_time"}


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def validate_split_config(split_config: dict[str, Any]) -> tuple[float, float, float]:
    if split_config.get("method") != "global_chronological":
        raise ValueError("split.method must be 'global_chronological'")
    if split_config.get("random_shuffle", False):
        raise ValueError("random_shuffle must be false for chronological splitting")
    ratios = tuple(
        float(split_config[name])
        for name in ("train_ratio", "validation_ratio", "test_ratio")
    )
    if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
        raise ValueError("all split ratios must be between 0 and 1")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/validation/test ratios must sum to 1")
    return ratios


def _cutoff_at_complete_timestamp(data: pd.DataFrame, target_rows: int, column: str) -> pd.Timestamp:
    """Choose a cutoff without placing the same timestamp in two partitions."""
    target_rows = min(max(target_rows, 1), len(data) - 1)
    return data.iloc[target_rows - 1][column]


def _partition_stats(frame: pd.DataFrame, total: int, timestamp: str) -> dict[str, Any]:
    return {
        "checkins": int(len(frame)),
        "ratio": float(len(frame) / total),
        "users": int(frame["user_id"].nunique()),
        "pois": int(frame["poi_id"].nunique()),
        "start_utc": frame[timestamp].min().isoformat(),
        "end_utc": frame[timestamp].max().isoformat(),
    }


def split_checkins(
    cleaned: pd.DataFrame,
    split_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split the whole event stream by time, preserving complete timestamp groups."""
    train_ratio, validation_ratio, test_ratio = validate_split_config(split_config)
    timestamp = str(split_config.get("timestamp_column", "utc_time"))
    missing = sorted((REQUIRED_COLUMNS | {timestamp}).difference(cleaned.columns))
    if missing:
        raise ValueError(f"cleaned data is missing required columns: {missing}")
    if len(cleaned) < 3:
        raise ValueError("cleaned data must contain at least three rows")

    data = cleaned.copy()
    data[timestamp] = pd.to_datetime(data[timestamp], utc=True, errors="coerce")
    if data[timestamp].isna().any():
        raise ValueError(f"{int(data[timestamp].isna().sum())} rows have invalid timestamps")
    data = data.sort_values([timestamp, "user_id", "poi_id"], kind="stable").reset_index(drop=True)
    data.insert(0, "event_id", range(len(data)))

    train_cutoff = _cutoff_at_complete_timestamp(data, int(len(data) * train_ratio), timestamp)
    validation_cutoff = _cutoff_at_complete_timestamp(
        data, int(len(data) * (train_ratio + validation_ratio)), timestamp
    )
    train = data.loc[data[timestamp].le(train_cutoff)].reset_index(drop=True)
    validation = data.loc[data[timestamp].gt(train_cutoff) & data[timestamp].le(validation_cutoff)].reset_index(drop=True)
    test = data.loc[data[timestamp].gt(validation_cutoff)].reset_index(drop=True)
    if any(frame.empty for frame in (train, validation, test)):
        raise ValueError("timestamp cutoffs produced an empty partition")

    train_users, train_pois = set(train["user_id"]), set(train["poi_id"])
    split_frames = {"train": train, "validation": validation, "test": test}
    cold_start: dict[str, int] = {}
    for name, frame in (("validation", validation), ("test", test)):
        known_user = frame["user_id"].isin(train_users)
        known_poi = frame["poi_id"].isin(train_pois)
        cold_start.update({
            f"{name}_unseen_user_rows": int((~known_user).sum()),
            f"{name}_unseen_users": int(frame.loc[~known_user, "user_id"].nunique()),
            f"{name}_unseen_poi_rows": int((~known_poi).sum()),
            f"{name}_unseen_pois": int(frame.loc[~known_poi, "poi_id"].nunique()),
            f"{name}_evaluable_rows": int((known_user & known_poi).sum()),
            f"{name}_evaluable_ratio": float((known_user & known_poi).mean()),
        })

    event_sets = {name: set(frame["event_id"]) for name, frame in split_frames.items()}
    overlap = sum(
        len(event_sets[a] & event_sets[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    report = {
        "method": "global_chronological",
        "rationale": "All models simulate one system timeline; no model may train on events later than an evaluated event.",
        "configured_ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        "timestamp_column": timestamp,
        "query_time_known": bool(split_config.get("query_time_known", True)),
        "candidate_scope": str(split_config.get("candidate_scope", "train_pois")),
        "total_checkins": int(len(data)),
        "partitions": {name: _partition_stats(frame, len(data), timestamp) for name, frame in split_frames.items()},
        "cutoffs": {"train_end_utc": train_cutoff.isoformat(), "validation_end_utc": validation_cutoff.isoformat()},
        "quality_checks": {
            "event_count_preserved": sum(map(len, split_frames.values())) == len(data),
            "overlapping_event_ids": int(overlap),
            "strict_global_time_order": bool(
                train[timestamp].max() < validation[timestamp].min()
                and validation[timestamp].max() < test[timestamp].min()
            ),
        },
        "cold_start": cold_start,
    }
    return train, validation, test, report


def render_split_markdown(report: dict[str, Any]) -> str:
    rows = [
        "# Foursquare NYC 数据集划分报告", "", "## 协议", "",
        "- 主协议：全局时间顺序 80% / 10% / 10%",
        "- 相同时间戳的记录不会跨集合，实际比例可能有极小偏差",
        "- 候选集：仅训练集出现过的 POI",
        f"- 推荐请求时间是否已知：{'是' if report['query_time_known'] else '否'}",
        "- 验证集用于调参与早停，测试集只用于最终报告", "",
        "| 集合 | 签到数 | 比例 | 用户 | POI | 起始UTC | 结束UTC |", "|---|---:|---:|---:|---:|---|---|",
    ]
    labels = {"train": "训练集", "validation": "验证集", "test": "测试集"}
    for name, item in report["partitions"].items():
        rows.append(f"| {labels[name]} | {item['checkins']:,} | {item['ratio']:.2%} | {item['users']:,} | {item['pois']:,} | {item['start_utc']} | {item['end_utc']} |")
    q, c = report["quality_checks"], report["cold_start"]
    rows += [
        "", "## 一致性与覆盖率", "",
        f"- 全局时间严格有序：{'通过' if q['strict_global_time_order'] else '失败'}",
        f"- 集合间重复事件：{q['overlapping_event_ids']}",
        f"- 验证集已知用户且已知POI：{c['validation_evaluable_rows']:,}（{c['validation_evaluable_ratio']:.2%}）",
        f"- 测试集已知用户且已知POI：{c['test_evaluable_rows']:,}（{c['test_evaluable_ratio']:.2%}）",
        "- 冷启动记录保留在数据文件中，但不纳入闭集 Top-K 主指标；覆盖率单独报告。", "",
    ]
    return "\n".join(rows)


def save_split_outputs(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, report: dict[str, Any], config: dict[str, Any]) -> dict[str, Path]:
    output = config["output"]
    output_dir = resolve_project_path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / output["train_file"], "validation": output_dir / output["validation_file"],
        "test": output_dir / output["test_file"], "split_json": output_dir / output["split_json"],
        "split_markdown": output_dir / output["split_markdown"],
    }
    train.to_csv(paths["train"], index=False, encoding="utf-8")
    validation.to_csv(paths["validation"], index=False, encoding="utf-8")
    test.to_csv(paths["test"], index=False, encoding="utf-8")
    paths["split_json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["split_markdown"].write_text(render_split_markdown(report), encoding="utf-8")
    return paths


def run(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    config = load_yaml(config_path)
    output = config["output"]
    path = resolve_project_path(Path(output["directory"]) / output["cleaned_file"])
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}. Run src.preprocess first.")
    cleaned = pd.read_csv(path, dtype={"user_id": "string", "poi_id": "string"})
    train, validation, test, report = split_checkins(cleaned, config["split"])
    paths = save_split_outputs(train, validation, test, report, config)
    return train, validation, test, report, paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Globally split check-ins by time")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    train, validation, test, report, paths = run(args.config)
    print(f"Split {report['total_checkins']:,}: train={len(train):,}, validation={len(validation):,}, test={len(test):,}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
