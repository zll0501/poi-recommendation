"""Reproducible standardization, auditing, cleaning, and filtering for NYC.

The pipeline preserves the standardized pre-cleaning table, then creates a
separate cleaned table. Train/validation/test splitting remains a later stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STANDARD_COLUMNS = [
    "user_id",
    "poi_id",
    "category_id",
    "category_name",
    "latitude",
    "longitude",
    "timezone_offset_minutes",
    "utc_time_raw",
]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve configuration paths relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable checksum for raw-data version tracking."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_time_slots(slots: list[dict[str, Any]]) -> None:
    """Require non-overlapping time slots that cover every hour exactly once."""
    covered: list[int] = []
    for slot in slots:
        start = int(slot["start_hour"])
        end = int(slot["end_hour"])
        if not 0 <= start < end <= 24:
            raise ValueError(f"Invalid time slot: {slot}")
        covered.extend(range(start, end))
    if sorted(covered) != list(range(24)):
        raise ValueError("Time slots must cover hours 0-23 exactly once")


def load_raw_checkins(config: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    """Load raw check-ins and verify the declared input schema."""
    dataset_config = config["dataset"]
    input_path = resolve_project_path(dataset_config["input_path"])
    if not input_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {input_path}")

    frame = pd.read_csv(
        input_path,
        sep=dataset_config.get("delimiter", ","),
        encoding=dataset_config.get("encoding", "utf-8"),
        dtype={
            "userId": "string",
            "venueId": "string",
            "venueCategoryId": "string",
            "venueCategory": "string",
            "utcTimestamp": "string",
        },
        low_memory=False,
    )

    column_map = config["columns"]
    missing = sorted(set(column_map) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(column_map))
    if missing or unexpected:
        raise ValueError(
            "Raw schema mismatch. "
            f"Missing columns={missing}; unexpected columns={unexpected}"
        )

    frame = frame.rename(columns=column_map)
    if list(frame.columns) != STANDARD_COLUMNS:
        frame = frame[STANDARD_COLUMNS]
    return frame, input_path


def assign_time_slots(hours: pd.Series, slots: list[dict[str, Any]]) -> pd.Series:
    """Map local hours to configured semantic time slots."""
    validate_time_slots(slots)
    conditions = [
        (
            hours.ge(int(slot["start_hour"]))
            & hours.lt(int(slot["end_hour"]))
        )
        .fillna(False)
        .to_numpy(dtype=bool)
        for slot in slots
    ]
    names = [str(slot["name"]) for slot in slots]
    values = np.select(conditions, names, default=None)
    return pd.Series(values, index=hours.index, dtype="string")


def standardize_checkins(
    frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Normalize numeric fields and derive leakage-safe calendar features."""
    data = frame.copy()
    for column in ["latitude", "longitude", "timezone_offset_minutes"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    utc_format = config["time"]["utc_format"]
    data["utc_time"] = pd.to_datetime(
        data["utc_time_raw"], format=utc_format, errors="coerce", utc=True
    )

    offset = pd.to_timedelta(data["timezone_offset_minutes"], unit="m")
    # Keep local wall-clock time timezone-naive because rows span DST offsets.
    data["local_time"] = (data["utc_time"] + offset).dt.tz_localize(None)
    data["timestamp"] = data["utc_time"].map(
        lambda value: int(value.timestamp()) if pd.notna(value) else pd.NA
    ).astype("Int64")
    data["hour"] = data["local_time"].dt.hour.astype("Int64")
    data["weekday"] = data["local_time"].dt.weekday.astype("Int64")
    data["is_weekend"] = data["weekday"].isin([5, 6]).astype("boolean")
    data["time_slot"] = assign_time_slots(data["hour"], config["time"]["slots"])

    hour_float = data["hour"].astype("Float64")
    data["hour_sin"] = np.sin(2 * np.pi * hour_float / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour_float / 24)
    return data


def _describe_counts(counts: pd.Series) -> dict[str, int | float]:
    """Return JSON-safe distribution statistics."""
    if counts.empty:
        return {key: 0 for key in ["min", "q25", "median", "mean", "q75", "max"]}
    return {
        "min": int(counts.min()),
        "q25": float(counts.quantile(0.25)),
        "median": float(counts.median()),
        "mean": float(counts.mean()),
        "q75": float(counts.quantile(0.75)),
        "max": int(counts.max()),
    }


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def build_audit_report(
    data: pd.DataFrame, raw_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Compute structural, temporal, geographic, and behavioral audit metrics."""
    audit_config = config["audit"]
    numeric_missing = data[["latitude", "longitude", "timezone_offset_minutes"]].isna()
    invalid_latitude = data["latitude"].notna() & ~data["latitude"].between(-90, 90)
    invalid_longitude = data["longitude"].notna() & ~data["longitude"].between(-180, 180)
    offset_min = int(audit_config["valid_timezone_offset_min"])
    offset_max = int(audit_config["valid_timezone_offset_max"])
    invalid_offset = data["timezone_offset_minutes"].notna() & ~data[
        "timezone_offset_minutes"
    ].between(offset_min, offset_max)

    sorted_data = data.sort_values(
        ["user_id", "utc_time", "poi_id"], kind="stable", na_position="last"
    )
    consecutive_same_poi = (
        sorted_data["user_id"].eq(sorted_data["user_id"].shift())
        & sorted_data["poi_id"].eq(sorted_data["poi_id"].shift())
    )
    revisits = data.duplicated(["user_id", "poi_id"], keep="first")

    decimals = int(audit_config["coordinate_round_decimals"])
    rounded_coordinates = (
        data.assign(
            latitude_rounded=data["latitude"].round(decimals),
            longitude_rounded=data["longitude"].round(decimals),
        )
        .dropna(subset=["poi_id", "latitude_rounded", "longitude_rounded"])
        .groupby("poi_id", observed=True)[["latitude_rounded", "longitude_rounded"]]
        .nunique()
    )
    inconsistent_coordinate_pois = int(
        ((rounded_coordinates["latitude_rounded"] > 1) | (rounded_coordinates["longitude_rounded"] > 1)).sum()
    )

    user_counts = data.groupby("user_id", observed=True).size()
    poi_counts = data.groupby("poi_id", observed=True).size()
    valid_times = data["utc_time"].dropna()
    duplicate_user_time_rows = data.duplicated(
        ["user_id", "utc_time_raw"], keep=False
    )

    return {
        "dataset": {
            "name": config["dataset"]["name"],
            "raw_file": raw_path.name,
            "raw_file_bytes": int(raw_path.stat().st_size),
            "raw_sha256": sha256_file(raw_path),
        },
        "scale": {
            "checkins": int(len(data)),
            "users": int(data["user_id"].nunique(dropna=True)),
            "pois": int(data["poi_id"].nunique(dropna=True)),
            "categories": int(data["category_id"].nunique(dropna=True)),
            "utc_start": valid_times.min().isoformat() if not valid_times.empty else None,
            "utc_end": valid_times.max().isoformat() if not valid_times.empty else None,
        },
        "missing_values": {
            column: int(count) for column, count in data.isna().sum().items()
        },
        "quality": {
            "exact_duplicate_rows": int(data.duplicated().sum()),
            "duplicate_user_poi_time_rows": int(
                data.duplicated(["user_id", "poi_id", "utc_time_raw"]).sum()
            ),
            "same_user_same_time_rows": int(duplicate_user_time_rows.sum()),
            "same_user_same_time_groups": int(
                data.loc[duplicate_user_time_rows]
                .groupby(["user_id", "utc_time_raw"], observed=True)
                .ngroups
            ),
            "unparsed_utc_rows": int(data["utc_time"].isna().sum()),
            "invalid_latitude_rows": int(invalid_latitude.sum()),
            "invalid_longitude_rows": int(invalid_longitude.sum()),
            "invalid_timezone_offset_rows": int(invalid_offset.sum()),
            "missing_numeric_rows": int(numeric_missing.any(axis=1).sum()),
            "pois_with_inconsistent_rounded_coordinates": inconsistent_coordinate_pois,
        },
        "behavior": {
            "revisit_rows": int(revisits.sum()),
            "revisit_rate": float(revisits.mean()),
            "consecutive_same_poi_rows": int(consecutive_same_poi.sum()),
            "consecutive_same_poi_rate": float(consecutive_same_poi.mean()),
        },
        "distributions": {
            "checkins_per_user": _describe_counts(user_counts),
            "visits_per_poi": _describe_counts(poi_counts),
            "checkins_by_hour": _value_counts(data["hour"].sort_values()),
            "checkins_by_weekday": _value_counts(data["weekday"].sort_values()),
            "checkins_by_time_slot": _value_counts(data["time_slot"]),
        },
        "geography": {
            "latitude_min": float(data["latitude"].min()),
            "latitude_max": float(data["latitude"].max()),
            "longitude_min": float(data["longitude"].min()),
            "longitude_max": float(data["longitude"].max()),
        },
    }


def _stage_statistics(stage: str, data: pd.DataFrame) -> dict[str, int | str]:
    """Return compact, JSON-safe statistics for one cleaning stage."""
    return {
        "stage": stage,
        "checkins": int(len(data)),
        "users": int(data["user_id"].nunique(dropna=True)),
        "pois": int(data["poi_id"].nunique(dropna=True)),
        "categories": int(data["category_id"].nunique(dropna=True)),
    }


def iterative_frequency_filter(
    data: pd.DataFrame,
    min_user_checkins: int,
    min_poi_visits: int,
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    """Apply user and POI frequency constraints until reaching a fixed point."""
    if min_user_checkins < 1 or min_poi_visits < 1:
        raise ValueError("Frequency thresholds must be positive integers")

    current = data.copy()
    history: list[dict[str, int]] = []
    iteration = 0
    while True:
        iteration += 1
        before_rows = len(current)
        before_users = current["user_id"].nunique(dropna=True)
        before_pois = current["poi_id"].nunique(dropna=True)

        user_counts = current.groupby("user_id", observed=True).size()
        valid_users = user_counts[user_counts >= min_user_checkins].index
        current = current[current["user_id"].isin(valid_users)].copy()

        poi_counts = current.groupby("poi_id", observed=True).size()
        valid_pois = poi_counts[poi_counts >= min_poi_visits].index
        current = current[current["poi_id"].isin(valid_pois)].copy()

        item = {
            "iteration": iteration,
            "before_checkins": int(before_rows),
            "after_checkins": int(len(current)),
            "removed_checkins": int(before_rows - len(current)),
            "before_users": int(before_users),
            "after_users": int(current["user_id"].nunique(dropna=True)),
            "before_pois": int(before_pois),
            "after_pois": int(current["poi_id"].nunique(dropna=True)),
        }
        history.append(item)

        if (
            len(current) == before_rows
            and item["after_users"] == before_users
            and item["after_pois"] == before_pois
        ):
            break

    return current, history


def clean_checkins(
    data: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean sequence-breaking noise and apply iterative frequency filtering."""
    cleaning_config = config["cleaning"]
    filtering_config = config["filtering"]
    stages = [_stage_statistics("standardized_input", data)]

    current = data.copy()
    if cleaning_config.get("remove_exact_duplicates", True):
        before = len(current)
        current = current.drop_duplicates().copy()
        exact_duplicates_removed = before - len(current)
    else:
        exact_duplicates_removed = 0
    stages.append(_stage_statistics("after_exact_deduplication", current))

    conflict_strategy = cleaning_config.get(
        "simultaneous_conflict_strategy", "drop_group"
    )
    if conflict_strategy != "drop_group":
        raise ValueError(
            "Only simultaneous_conflict_strategy='drop_group' is currently supported"
        )
    conflict_sizes = current.groupby(
        ["user_id", "utc_time"], observed=True, dropna=False
    )["poi_id"].transform("size")
    conflict_mask = conflict_sizes.gt(1)
    conflict_rows_removed = int(conflict_mask.sum())
    conflict_groups_removed = int(
        current.loc[conflict_mask]
        .groupby(["user_id", "utc_time"], observed=True, dropna=False)
        .ngroups
    )
    current = current.loc[~conflict_mask].copy()
    stages.append(_stage_statistics("after_simultaneous_conflicts", current))

    current = current.sort_values(
        ["user_id", "utc_time", "poi_id"], kind="stable", na_position="last"
    )
    threshold_minutes = float(
        cleaning_config.get("consecutive_same_poi_minutes", 10)
    )
    if threshold_minutes < 0:
        raise ValueError("consecutive_same_poi_minutes cannot be negative")

    if cleaning_config.get("merge_consecutive_same_poi", True):
        same_user = current["user_id"].eq(current["user_id"].shift())
        same_poi = current["poi_id"].eq(current["poi_id"].shift())
        gap_minutes = (
            current["utc_time"] - current["utc_time"].shift()
        ).dt.total_seconds().div(60)
        short_repeat_mask = (
            same_user
            & same_poi
            & gap_minutes.ge(0)
            & gap_minutes.le(threshold_minutes)
        )
        short_repeats_removed = int(short_repeat_mask.sum())
        current = current.loc[~short_repeat_mask].copy()
    else:
        short_repeats_removed = 0
    stages.append(_stage_statistics("after_short_consecutive_repeats", current))

    min_user_checkins = int(filtering_config["min_user_checkins"])
    min_poi_visits = int(filtering_config["min_poi_visits"])
    if filtering_config.get("iterative", True):
        current, filter_history = iterative_frequency_filter(
            current,
            min_user_checkins=min_user_checkins,
            min_poi_visits=min_poi_visits,
        )
    else:
        filter_history = []
    stages.append(_stage_statistics("after_iterative_filtering", current))

    current = current.sort_values(
        ["user_id", "utc_time", "poi_id"], kind="stable", na_position="last"
    ).reset_index(drop=True)

    final_user_counts = current.groupby("user_id", observed=True).size()
    final_poi_counts = current.groupby("poi_id", observed=True).size()
    report: dict[str, Any] = {
        "rules": {
            "remove_exact_duplicates": bool(
                cleaning_config.get("remove_exact_duplicates", True)
            ),
            "simultaneous_conflict_strategy": conflict_strategy,
            "consecutive_same_poi_minutes": threshold_minutes,
            "min_user_checkins": min_user_checkins,
            "min_poi_visits": min_poi_visits,
            "iterative_filtering": bool(filtering_config.get("iterative", True)),
        },
        "removed": {
            "exact_duplicate_rows": int(exact_duplicates_removed),
            "simultaneous_conflict_groups": conflict_groups_removed,
            "simultaneous_conflict_rows": conflict_rows_removed,
            "short_consecutive_same_poi_rows": short_repeats_removed,
            "frequency_filter_rows": int(
                stages[-2]["checkins"] - stages[-1]["checkins"]
            ),
            "total_rows": int(len(data) - len(current)),
        },
        "stages": stages,
        "filter_iterations": filter_history,
        "final_constraints": {
            "minimum_user_checkins": int(final_user_counts.min())
            if not final_user_counts.empty
            else 0,
            "minimum_poi_visits": int(final_poi_counts.min())
            if not final_poi_counts.empty
            else 0,
            "exact_duplicate_rows": int(current.duplicated().sum()),
            "simultaneous_conflict_groups": int(
                (
                    current.groupby(
                        ["user_id", "utc_time"], observed=True, dropna=False
                    ).size()
                    > 1
                ).sum()
            ),
        },
    }
    return current, report


def render_audit_markdown(report: dict[str, Any]) -> str:
    """Create a concise human-readable companion to the JSON audit."""
    scale = report["scale"]
    quality = report["quality"]
    behavior = report["behavior"]
    user_stats = report["distributions"]["checkins_per_user"]
    poi_stats = report["distributions"]["visits_per_poi"]
    return "\n".join(
        [
            "# Foursquare NYC 原始数据审计报告",
            "",
            "## 数据版本",
            "",
            f"- 文件：`{report['dataset']['raw_file']}`",
            f"- 大小：{report['dataset']['raw_file_bytes']:,} bytes",
            f"- SHA256：`{report['dataset']['raw_sha256']}`",
            "",
            "## 数据规模",
            "",
            f"- 签到记录：{scale['checkins']:,}",
            f"- 用户：{scale['users']:,}",
            f"- POI：{scale['pois']:,}",
            f"- 类别：{scale['categories']:,}",
            f"- UTC时间范围：{scale['utc_start']} 至 {scale['utc_end']}",
            "",
            "## 质量检查",
            "",
            f"- 完全重复记录：{quality['exact_duplicate_rows']:,}",
            f"- 用户、POI、时间重复记录：{quality['duplicate_user_poi_time_rows']:,}",
            f"- 同一用户同一时间记录：{quality['same_user_same_time_rows']:,}",
            f"- 无法解析UTC时间：{quality['unparsed_utc_rows']:,}",
            f"- 非法纬度：{quality['invalid_latitude_rows']:,}",
            f"- 非法经度：{quality['invalid_longitude_rows']:,}",
            f"- 非法时区偏移：{quality['invalid_timezone_offset_rows']:,}",
            f"- 坐标不一致POI：{quality['pois_with_inconsistent_rounded_coordinates']:,}",
            "",
            "## 行为特征",
            "",
            f"- 重复访问记录比例：{behavior['revisit_rate']:.2%}",
            f"- 连续相同POI比例：{behavior['consecutive_same_poi_rate']:.2%}",
            f"- 用户签到数：中位数 {user_stats['median']:.1f}，平均数 {user_stats['mean']:.2f}，最大值 {user_stats['max']}",
            f"- POI访问数：中位数 {poi_stats['median']:.1f}，平均数 {poi_stats['mean']:.2f}，最大值 {poi_stats['max']}",
            "",
            "## 当前阶段说明",
            "",
            "本阶段仅完成字段标准化、时间解析和质量审计，没有删除任何记录，也没有使用验证集或测试集信息计算模型统计量。",
            "",
        ]
    )


def render_cleaning_markdown(report: dict[str, Any]) -> str:
    """Render stage-by-stage cleaning statistics as a Markdown report."""
    removed = report["removed"]
    rules = report["rules"]
    rows = [
        "# Foursquare NYC 数据清洗报告",
        "",
        "## 清洗规则",
        "",
        f"- 完全重复记录：{'删除' if rules['remove_exact_duplicates'] else '保留'}",
        f"- 同用户同时间冲突：`{rules['simultaneous_conflict_strategy']}`",
        f"- 连续相同POI合并阈值：{rules['consecutive_same_poi_minutes']:g}分钟",
        f"- 用户最少签到数：{rules['min_user_checkins']}",
        f"- POI最少访问数：{rules['min_poi_visits']}",
        f"- 迭代过滤：{'是' if rules['iterative_filtering'] else '否'}",
        "",
        "## 分阶段统计",
        "",
        "| 阶段 | 签到数 | 用户数 | POI数 | 类别数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in report["stages"]:
        rows.append(
            f"| {stage['stage']} | {stage['checkins']:,} | "
            f"{stage['users']:,} | {stage['pois']:,} | {stage['categories']:,} |"
        )

    rows.extend(
        [
            "",
            "## 删除记录明细",
            "",
            f"- 完全重复记录：{removed['exact_duplicate_rows']:,}",
            f"- 时间冲突组：{removed['simultaneous_conflict_groups']:,}组，共{removed['simultaneous_conflict_rows']:,}条",
            f"- {rules['consecutive_same_poi_minutes']:g}分钟内连续相同POI：{removed['short_consecutive_same_poi_rows']:,}",
            f"- 频率过滤：{removed['frequency_filter_rows']:,}",
            f"- 总删除记录：{removed['total_rows']:,}",
            "",
            "## 迭代过滤过程",
            "",
            "| 轮次 | 处理前 | 处理后 | 删除 | 用户数变化 | POI数变化 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["filter_iterations"]:
        rows.append(
            f"| {item['iteration']} | {item['before_checkins']:,} | "
            f"{item['after_checkins']:,} | {item['removed_checkins']:,} | "
            f"{item['before_users']:,}→{item['after_users']:,} | "
            f"{item['before_pois']:,}→{item['after_pois']:,} |"
        )
    rows.extend(
        [
            "",
            "最终数据已按用户和UTC时间排序。当前阶段尚未执行训练、验证、测试划分，也未建立连续整数映射。",
            "",
        ]
    )
    return "\n".join(rows)


def save_outputs(
    standardized: pd.DataFrame,
    audit_report: dict[str, Any],
    cleaned: pd.DataFrame,
    cleaning_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Path]:
    """Persist standardized/cleaned data and their audit reports."""
    output_config = config["output"]
    output_dir = resolve_project_path(output_config["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    standardized_path = output_dir / output_config["standardized_file"]
    audit_json_path = output_dir / output_config["audit_json"]
    audit_markdown_path = output_dir / output_config["audit_markdown"]
    cleaned_path = output_dir / output_config["cleaned_file"]
    cleaning_json_path = output_dir / output_config["cleaning_json"]
    cleaning_markdown_path = output_dir / output_config["cleaning_markdown"]

    standardized.to_csv(standardized_path, index=False, encoding="utf-8")
    with audit_json_path.open("w", encoding="utf-8") as file:
        json.dump(audit_report, file, ensure_ascii=False, indent=2)
    audit_markdown_path.write_text(
        render_audit_markdown(audit_report), encoding="utf-8"
    )
    cleaned.to_csv(cleaned_path, index=False, encoding="utf-8")
    with cleaning_json_path.open("w", encoding="utf-8") as file:
        json.dump(cleaning_report, file, ensure_ascii=False, indent=2)
    cleaning_markdown_path.write_text(
        render_cleaning_markdown(cleaning_report), encoding="utf-8"
    )
    return {
        "standardized": standardized_path,
        "audit_json": audit_json_path,
        "audit_markdown": audit_markdown_path,
        "cleaned": cleaned_path,
        "cleaning_json": cleaning_json_path,
        "cleaning_markdown": cleaning_markdown_path,
    }


def run(
    config_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Execute standardization, audit, cleaning, and iterative filtering."""
    config = load_yaml(config_path)
    raw, raw_path = load_raw_checkins(config)
    standardized = standardize_checkins(raw, config)
    audit_report = build_audit_report(standardized, raw_path, config)
    cleaned, cleaning_report = clean_checkins(standardized, config)
    paths = save_outputs(
        standardized,
        audit_report,
        cleaned,
        cleaning_report,
        config,
    )
    return standardized, cleaned, audit_report, cleaning_report, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardize and audit the Foursquare NYC dataset"
    )
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to the data YAML configuration",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    standardized, cleaned, audit_report, cleaning_report, paths = run(args.config)
    print(f"Standardized {len(standardized):,} check-ins")
    print(f"Cleaned {len(cleaned):,} check-ins")
    print(
        f"Users={cleaning_report['stages'][-1]['users']:,}, "
        f"POIs={cleaning_report['stages'][-1]['pois']:,}, "
        f"categories={cleaning_report['stages'][-1]['categories']:,}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
