"""Domain-aware profiling for cleaned POI check-in data.

This module audits and describes the full cleaned dataset. Its outputs are for
data understanding and reporting only; model features must still be fitted on
the training partition by the feature-engineering stage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {
    "user_id",
    "poi_id",
    "category_id",
    "latitude",
    "longitude",
    "timezone_offset_minutes",
    "utc_time",
    "local_time",
    "hour",
    "weekday",
    "is_weekend",
    "time_slot",
}


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def haversine_km(
    latitude_a: pd.Series,
    longitude_a: pd.Series,
    latitude_b: pd.Series,
    longitude_b: pd.Series,
) -> pd.Series:
    """Calculate great-circle distance between paired coordinates."""
    lat_a = np.radians(latitude_a.astype(float))
    lon_a = np.radians(longitude_a.astype(float))
    lat_b = np.radians(latitude_b.astype(float))
    lon_b = np.radians(longitude_b.astype(float))
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    value = np.clip(value, 0.0, 1.0)
    return pd.Series(6371.0088 * 2.0 * np.arcsin(np.sqrt(value)), index=latitude_a.index)


def _numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {key: None for key in ("count", "minimum", "mean", "median", "p95", "maximum")}
    return {
        "count": int(len(clean)),
        "minimum": float(clean.min()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "maximum": float(clean.max()),
    }


def build_transition_table(
    data: pd.DataFrame,
    suspicious_speed_kmh: float,
) -> pd.DataFrame:
    """Construct consecutive-user transitions without removing any records."""
    ordered = data.sort_values(["user_id", "utc_time", "poi_id"], kind="stable").copy()
    ordered.insert(0, "profile_event_id", range(len(ordered)))
    grouped = ordered.groupby("user_id", sort=False, observed=True)
    ordered["previous_poi_id"] = grouped["poi_id"].shift()
    ordered["previous_utc_time"] = grouped["utc_time"].shift()
    ordered["previous_latitude"] = grouped["latitude"].shift()
    ordered["previous_longitude"] = grouped["longitude"].shift()
    ordered["time_gap_hours"] = (
        ordered["utc_time"] - ordered["previous_utc_time"]
    ).dt.total_seconds() / 3600.0
    valid = ordered["previous_poi_id"].notna() & ordered["time_gap_hours"].gt(0)
    ordered["distance_km"] = np.nan
    ordered.loc[valid, "distance_km"] = haversine_km(
        ordered.loc[valid, "previous_latitude"],
        ordered.loc[valid, "previous_longitude"],
        ordered.loc[valid, "latitude"],
        ordered.loc[valid, "longitude"],
    )
    ordered["speed_kmh"] = ordered["distance_km"] / ordered["time_gap_hours"]
    ordered["is_suspicious_speed"] = ordered["speed_kmh"].gt(suspicious_speed_kmh)
    columns = [
        "profile_event_id",
        "user_id",
        "previous_poi_id",
        "poi_id",
        "previous_utc_time",
        "utc_time",
        "time_gap_hours",
        "distance_km",
        "speed_kmh",
        "is_suspicious_speed",
    ]
    return ordered.loc[valid, columns].reset_index(drop=True)


def build_user_profile(data: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    """Create descriptive per-user behaviour statistics."""
    grouped = data.groupby("user_id", observed=True)
    profile = grouped.agg(
        checkins=("poi_id", "size"),
        unique_pois=("poi_id", "nunique"),
        unique_categories=("category_id", "nunique"),
        first_checkin=("utc_time", "min"),
        last_checkin=("utc_time", "max"),
        weekend_ratio=("is_weekend", "mean"),
    )
    profile["active_days"] = grouped["local_time"].apply(
        lambda values: values.dt.date.nunique()
    )
    profile["revisit_rate"] = 1.0 - profile["unique_pois"] / profile["checkins"]
    if not transitions.empty:
        transition_stats = transitions.groupby("user_id", observed=True).agg(
            mean_time_gap_hours=("time_gap_hours", "mean"),
            median_time_gap_hours=("time_gap_hours", "median"),
            mean_move_distance_km=("distance_km", "mean"),
            median_move_distance_km=("distance_km", "median"),
            suspicious_speed_count=("is_suspicious_speed", "sum"),
        )
        profile = profile.join(transition_stats)
    profile["suspicious_speed_count"] = (
        profile["suspicious_speed_count"].fillna(0).astype(int)
    )
    return profile.reset_index().sort_values("checkins", ascending=False, kind="stable")


def build_poi_profile(
    data: pd.DataFrame,
    coordinate_decimals: int,
    coordinate_tolerance_m: float,
) -> pd.DataFrame:
    """Create POI popularity and metadata-consistency statistics."""
    working = data.copy()
    working["coordinate_key"] = (
        working["latitude"].round(coordinate_decimals).astype("string")
        + ","
        + working["longitude"].round(coordinate_decimals).astype("string")
    )
    grouped = working.groupby("poi_id", observed=True)
    profile = grouped.agg(
        checkins=("user_id", "size"),
        unique_users=("user_id", "nunique"),
        category_count=("category_id", "nunique"),
        category_name_count=("category_name", "nunique"),
        coordinate_count=("coordinate_key", "nunique"),
        timezone_count=("timezone_offset_minutes", "nunique"),
        first_checkin=("utc_time", "min"),
        last_checkin=("utc_time", "max"),
        weekend_ratio=("is_weekend", "mean"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        latitude_min=("latitude", "min"),
        latitude_max=("latitude", "max"),
        longitude_min=("longitude", "min"),
        longitude_max=("longitude", "max"),
        category_id=("category_id", "first"),
        category_name=("category_name", "first"),
    ).reset_index()
    profile["coordinate_span_m"] = 1000.0 * haversine_km(
        profile["latitude_min"],
        profile["longitude_min"],
        profile["latitude_max"],
        profile["longitude_max"],
    )
    profile["visit_share"] = profile["checkins"] / len(data)
    profile["popularity_rank"] = profile["checkins"].rank(
        method="min", ascending=False
    ).astype(int)
    profile["is_metadata_consistent"] = (
        profile["category_count"].eq(1)
        & profile["category_name_count"].eq(1)
        & profile["coordinate_span_m"].le(coordinate_tolerance_m)
    )
    return profile.sort_values(
        ["checkins", "poi_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def build_category_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Create category-level scale and temporal summaries."""
    profile = (
        data.groupby(["category_id", "category_name"], observed=True)
        .agg(
            checkins=("user_id", "size"),
            unique_users=("user_id", "nunique"),
            unique_pois=("poi_id", "nunique"),
            weekend_ratio=("is_weekend", "mean"),
            mean_hour=("hour", "mean"),
        )
        .reset_index()
    )
    profile["visit_share"] = profile["checkins"] / len(data)
    return profile.sort_values(
        ["checkins", "category_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def _top_popularity_shares(
    poi_profile: pd.DataFrame,
    fractions: list[float],
) -> dict[str, dict[str, float | int]]:
    total = float(poi_profile["checkins"].sum())
    result = {}
    for fraction in fractions:
        if fraction <= 0 or fraction > 1:
            raise ValueError("popularity_top_fractions must be in (0, 1]")
        count = max(1, math.ceil(len(poi_profile) * fraction))
        result[f"top_{fraction:.0%}"] = {
            "poi_count": count,
            "checkin_share": float(poi_profile.head(count)["checkins"].sum() / total),
        }
    return result


def _value_counts(values: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in values.value_counts(dropna=False).sort_index().items()
    }


def build_data_profile(
    cleaned: pd.DataFrame,
    partitions: dict[str, pd.DataFrame],
    profile_config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the full descriptive audit and its detailed profile tables."""
    missing = sorted(REQUIRED_COLUMNS.difference(cleaned.columns))
    if missing:
        raise ValueError(f"cleaned data is missing profile columns: {missing}")
    data = cleaned.copy()
    data["utc_time"] = pd.to_datetime(data["utc_time"], utc=True, errors="coerce")
    data["local_time"] = pd.to_datetime(data["local_time"], errors="coerce")
    if data["utc_time"].isna().any() or data["local_time"].isna().any():
        raise ValueError("profile input contains invalid timestamps")

    speed_threshold = float(profile_config.get("suspicious_speed_kmh", 300))
    coordinate_decimals = int(profile_config.get("coordinate_round_decimals", 5))
    coordinate_tolerance_m = float(profile_config.get("coordinate_tolerance_m", 100))
    expected_offsets = set(
        int(value)
        for value in profile_config.get(
            "expected_timezone_offsets_minutes", [-300, -240]
        )
    )
    fractions = [
        float(value)
        for value in profile_config.get("popularity_top_fractions", [0.01, 0.05, 0.1])
    ]
    transitions = build_transition_table(data, speed_threshold)
    users = build_user_profile(data, transitions)
    pois = build_poi_profile(data, coordinate_decimals, coordinate_tolerance_m)
    categories = build_category_profile(data)
    suspicious = transitions.loc[transitions["is_suspicious_speed"]].copy()

    user_count = int(data["user_id"].nunique())
    poi_count = int(data["poi_id"].nunique())
    unique_user_poi = int(data[["user_id", "poi_id"]].drop_duplicates().shape[0])
    invalid_coordinate = ~data["latitude"].between(-90, 90) | ~data["longitude"].between(-180, 180)
    unexpected_timezone = ~data["timezone_offset_minutes"].isin(expected_offsets)
    unexpected_timezone_pois = data.loc[unexpected_timezone, "poi_id"].nunique()
    train_pois = set(partitions["train"]["poi_id"].astype("string"))
    split_stats = {}
    for name, frame in partitions.items():
        poi_values = frame["poi_id"].astype("string")
        unseen = ~poi_values.isin(train_pois)
        split_stats[name] = {
            "checkins": int(len(frame)),
            "users": int(frame["user_id"].nunique()),
            "pois": int(frame["poi_id"].nunique()),
            "unseen_poi_rows_against_train": int(unseen.sum()) if name != "train" else 0,
            "unseen_pois_against_train": int(poi_values[unseen].nunique()) if name != "train" else 0,
        }

    report = {
        "purpose": "descriptive_profile_not_model_features",
        "scale": {
            "checkins": int(len(data)),
            "users": user_count,
            "pois": poi_count,
            "categories": int(data["category_id"].nunique()),
            "time_start": data["utc_time"].min().isoformat(),
            "time_end": data["utc_time"].max().isoformat(),
        },
        "quality": {
            "missing_values": {
                column: int(value) for column, value in data.isna().sum().items()
            },
            "exact_duplicate_rows": int(data.duplicated().sum()),
            "invalid_coordinate_rows": int(invalid_coordinate.sum()),
            "inconsistent_metadata_pois": int((~pois["is_metadata_consistent"]).sum()),
            "inconsistent_category_pois": int(pois["category_count"].gt(1).sum()),
            "coordinate_tolerance_m": coordinate_tolerance_m,
            "inconsistent_coordinate_pois": int(
                pois["coordinate_span_m"].gt(coordinate_tolerance_m).sum()
            ),
            "unexpected_timezone_offset_rows": int(unexpected_timezone.sum()),
            "pois_with_unexpected_timezone_offsets": int(unexpected_timezone_pois),
        },
        "sparsity_and_long_tail": {
            "unique_user_poi_pairs": unique_user_poi,
            "user_poi_matrix_density": float(unique_user_poi / (user_count * poi_count)),
            "user_poi_matrix_sparsity": float(1.0 - unique_user_poi / (user_count * poi_count)),
            "poi_popularity": _numeric_summary(pois["checkins"]),
            "user_activity": _numeric_summary(users["checkins"]),
            "top_popularity_shares": _top_popularity_shares(pois, fractions),
        },
        "behaviour": {
            "overall_revisit_rate": float(
                1.0 - unique_user_poi / len(data)
            ),
            "transition_count": int(len(transitions)),
            "time_gap_hours": _numeric_summary(transitions["time_gap_hours"]),
            "move_distance_km": _numeric_summary(transitions["distance_km"]),
            "speed_kmh": _numeric_summary(transitions["speed_kmh"]),
            "suspicious_speed_threshold_kmh": speed_threshold,
            "suspicious_speed_transitions": int(len(suspicious)),
        },
        "temporal": {
            "hour_counts": _value_counts(data["hour"]),
            "weekday_counts": _value_counts(data["weekday"]),
            "time_slot_counts": _value_counts(data["time_slot"]),
            "weekend_counts": _value_counts(data["is_weekend"]),
            "month_counts": _value_counts(data["local_time"].dt.to_period("M").astype(str)),
        },
        "splits": split_stats,
    }
    return report, users, pois, categories, suspicious


def render_profile_markdown(report: dict[str, Any]) -> str:
    """Render key findings in a report-ready Markdown document."""
    scale = report["scale"]
    quality = report["quality"]
    long_tail = report["sparsity_and_long_tail"]
    behaviour = report["behaviour"]
    rows = [
        "# Foursquare NYC 数据画像与领域质量报告",
        "",
        "> 本报告使用完整清洗数据进行描述性分析，不作为模型特征统计来源。模型特征必须仅从训练集拟合。",
        "",
        "## 数据规模",
        "",
        f"- 签到：{scale['checkins']:,}",
        f"- 用户：{scale['users']:,}",
        f"- POI：{scale['pois']:,}",
        f"- 类别：{scale['categories']:,}",
        f"- 时间范围：{scale['time_start']} 至 {scale['time_end']}",
        "",
        "## 质量与一致性",
        "",
        f"- 完全重复记录：{quality['exact_duplicate_rows']:,}",
        f"- 非法经纬度记录：{quality['invalid_coordinate_rows']:,}",
        f"- 元数据不一致POI：{quality['inconsistent_metadata_pois']:,}",
        f"- 类别不一致POI：{quality['inconsistent_category_pois']:,}",
        f"- 坐标跨度超过{quality['coordinate_tolerance_m']:g}米的POI：{quality['inconsistent_coordinate_pois']:,}",
        f"- 非纽约标准时区偏移记录：{quality['unexpected_timezone_offset_rows']:,}",
        f"- 涉及非标准时区偏移的POI：{quality['pois_with_unexpected_timezone_offsets']:,}",
        "",
        "## 稀疏性与长尾",
        "",
        f"- 唯一用户—POI对：{long_tail['unique_user_poi_pairs']:,}",
        f"- 用户—POI矩阵密度：{long_tail['user_poi_matrix_density']:.4%}",
        f"- 用户—POI矩阵稀疏度：{long_tail['user_poi_matrix_sparsity']:.4%}",
    ]
    for name, item in long_tail["top_popularity_shares"].items():
        rows.append(
            f"- {name.replace('_', ' ')} POI（{item['poi_count']:,}个）贡献签到：{item['checkin_share']:.2%}"
        )
    rows.extend(
        [
            "",
            "## 用户行为与移动审计",
            "",
            f"- 整体重复访问率：{behaviour['overall_revisit_rate']:.2%}",
            f"- 连续移动记录：{behaviour['transition_count']:,}",
            f"- 可疑速度阈值：{behaviour['suspicious_speed_threshold_kmh']:g} km/h",
            f"- 可疑高速移动：{behaviour['suspicious_speed_transitions']:,}",
            "",
            "可疑移动仅被标记，不会自动删除；是否处理需结合距离、时间间隔和敏感性实验判断。",
            "",
            "## 数据划分覆盖",
            "",
            "| 集合 | 签到 | 用户 | POI | 新POI签到 | 新POI数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["splits"].items():
        rows.append(
            f"| {name} | {item['checkins']:,} | {item['users']:,} | {item['pois']:,} | "
            f"{item['unseen_poi_rows_against_train']:,} | {item['unseen_pois_against_train']:,} |"
        )
    rows.append("")
    return "\n".join(rows)


def save_outputs(
    report: dict[str, Any],
    users: pd.DataFrame,
    pois: pd.DataFrame,
    categories: pd.DataFrame,
    suspicious: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Path]:
    output = config["output"]
    output_dir = resolve_project_path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "profile_json": output_dir / output["data_profile_json"],
        "profile_markdown": output_dir / output["data_profile_markdown"],
    }
    paths["profile_json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["profile_markdown"].write_text(
        render_profile_markdown(report), encoding="utf-8"
    )
    return paths


def run(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = load_yaml(config_path)
    output = config["output"]
    output_dir = resolve_project_path(output["directory"])
    cleaned_path = output_dir / output["cleaned_file"]
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}")
    dtype = {"user_id": "string", "poi_id": "string", "category_id": "string"}
    cleaned = pd.read_csv(cleaned_path, dtype=dtype)
    partition_files = {
        "train": output["train_file"],
        "validation": output["validation_file"],
        "test": output["test_file"],
    }
    partitions = {}
    for name, filename in partition_files.items():
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Split data not found: {path}")
        partitions[name] = pd.read_csv(path, dtype=dtype)
    report, users, pois, categories, suspicious = build_data_profile(
        cleaned, partitions, config["profiling"]
    )
    paths = save_outputs(report, users, pois, categories, suspicious, config)
    return report, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile cleaned POI check-ins")
    parser.add_argument("--config", default="configs/data.yaml")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, paths = run(args.config)
    scale = report["scale"]
    print(
        f"Profiled {scale['checkins']:,} check-ins, {scale['users']:,} users, "
        f"{scale['pois']:,} POIs"
    )
    print(
        f"sparsity={report['sparsity_and_long_tail']['user_poi_matrix_sparsity']:.4%}, "
        f"suspicious_transitions={report['behaviour']['suspicious_speed_transitions']:,}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
