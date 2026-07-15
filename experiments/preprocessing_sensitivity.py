"""Sensitivity experiments for POI filtering and repeat-merge thresholds."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocess import clean_checkins, load_raw_checkins, standardize_checkins
from src.split_data import split_checkins
from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _rank_pois(values: pd.Series) -> dict[str, int]:
    """Rank POIs by descending count with a deterministic ID tie-break."""
    counts = values.astype("string").value_counts().rename_axis("poi_id").reset_index(name="count")
    counts = counts.sort_values(
        ["count", "poi_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    return {
        str(poi_id): rank
        for rank, poi_id in enumerate(counts["poi_id"], start=1)
    }


def evaluate_popularity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: list[int],
    time_column: str = "time_slot",
) -> dict[str, float | int]:
    """Evaluate global and time-slot popularity on full train candidates."""
    if not top_k or any(k < 1 for k in top_k):
        raise ValueError("top_k must contain positive integers")
    global_ranks = _rank_pois(train["poi_id"])
    slot_ranks = {
        str(slot): _rank_pois(group["poi_id"])
        for slot, group in train.groupby(time_column, observed=True)
    }
    rows = test[["user_id", "poi_id", time_column]].copy()
    rows["poi_id"] = rows["poi_id"].astype("string")
    train_users = set(train["user_id"].astype("string"))
    rows["is_evaluable"] = (
        rows["poi_id"].isin(global_ranks)
        & rows["user_id"].astype("string").isin(train_users)
    )
    evaluable = rows.loc[rows["is_evaluable"]].copy()
    if evaluable.empty:
        raise ValueError("test partition has no targets seen in training")
    evaluable["global_rank"] = evaluable["poi_id"].map(global_ranks).astype(int)
    evaluable["time_rank"] = [
        slot_ranks.get(str(slot), global_ranks).get(str(poi_id), global_ranks[str(poi_id)])
        for slot, poi_id in zip(evaluable[time_column], evaluable["poi_id"])
    ]
    result: dict[str, float | int] = {
        "test_rows": int(len(rows)),
        "evaluable_rows": int(len(evaluable)),
        "candidate_coverage": float(len(evaluable) / len(rows)),
        "candidate_pois": int(len(global_ranks)),
    }
    primary_k = max(top_k)
    for model, rank_column in (("global", "global_rank"), ("time", "time_rank")):
        ranks = evaluable[rank_column].astype(float)
        result[f"{model}_mrr"] = float((1.0 / ranks).mean())
        for k in sorted(set(top_k)):
            result[f"{model}_acc@{k}"] = float(ranks.le(k).mean())
        per_user_acc = ranks.le(primary_k).groupby(evaluable["user_id"]).mean()
        per_user_mrr = (1.0 / ranks).groupby(evaluable["user_id"]).mean()
        result[f"{model}_macro_user_acc@{primary_k}"] = float(per_user_acc.mean())
        result[f"{model}_macro_user_mrr"] = float(per_user_mrr.mean())
    return result


def _dataset_statistics(cleaned: pd.DataFrame, raw_rows: int) -> dict[str, float | int]:
    users = int(cleaned["user_id"].nunique())
    pois = int(cleaned["poi_id"].nunique())
    pairs = int(cleaned[["user_id", "poi_id"]].drop_duplicates().shape[0])
    denominator = users * pois
    user_counts = cleaned.groupby("user_id", observed=True).size()
    poi_counts = cleaned.groupby("poi_id", observed=True).size()
    return {
        "checkins": int(len(cleaned)),
        "retention_rate": float(len(cleaned) / raw_rows),
        "users": users,
        "pois": pois,
        "categories": int(cleaned["category_id"].nunique()),
        "user_poi_pairs": pairs,
        "matrix_sparsity": float(1.0 - pairs / denominator),
        "revisit_rate": float(1.0 - pairs / len(cleaned)),
        "minimum_user_checkins": int(user_counts.min()),
        "median_user_checkins": float(user_counts.median()),
        "minimum_poi_visits": int(poi_counts.min()),
        "median_poi_visits": float(poi_counts.median()),
    }


def evaluate_variant(
    standardized: pd.DataFrame,
    base_config: dict[str, Any],
    experiment_group: str,
    variant: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    """Re-run cleaning, chronological split, and popularity diagnostics."""
    config = deepcopy(base_config)
    if experiment_group == "frequency":
        config["filtering"]["min_user_checkins"] = int(variant["min_user_checkins"])
        config["filtering"]["min_poi_visits"] = int(variant["min_poi_visits"])
    elif experiment_group == "merge":
        config["cleaning"]["merge_consecutive_same_poi"] = bool(variant["enabled"])
        config["cleaning"]["consecutive_same_poi_minutes"] = float(variant["minutes"])
    else:
        raise ValueError(f"unknown experiment group: {experiment_group}")

    cleaned, cleaning_report = clean_checkins(standardized, config)
    train, validation, test, split_report = split_checkins(cleaned, config["split"])
    metrics = evaluate_popularity(
        train,
        test,
        [int(value) for value in evaluation_config["top_k"]],
        str(evaluation_config.get("query_time_column", "time_slot")),
    )
    result: dict[str, Any] = {
        "experiment_group": experiment_group,
        "variant": str(variant["name"]),
        "min_user_checkins": int(config["filtering"]["min_user_checkins"]),
        "min_poi_visits": int(config["filtering"]["min_poi_visits"]),
        "merge_enabled": bool(config["cleaning"]["merge_consecutive_same_poi"]),
        "merge_minutes": float(config["cleaning"]["consecutive_same_poi_minutes"]),
        **_dataset_statistics(cleaned, len(standardized)),
        "removed_exact_duplicates": int(cleaning_report["removed"]["exact_duplicate_rows"]),
        "removed_simultaneous_conflicts": int(cleaning_report["removed"]["simultaneous_conflict_rows"]),
        "removed_short_repeats": int(cleaning_report["removed"]["short_consecutive_same_poi_rows"]),
        "removed_frequency_filter": int(cleaning_report["removed"]["frequency_filter_rows"]),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_unseen_poi_rows": int(split_report["cold_start"]["test_unseen_poi_rows"]),
        "test_unseen_pois": int(split_report["cold_start"]["test_unseen_pois"]),
        **metrics,
    }
    return result


def run_sensitivity(
    standardized: pd.DataFrame,
    data_config: dict[str, Any],
    sensitivity_config: dict[str, Any],
) -> pd.DataFrame:
    """Run every configured variant in a stable order."""
    rows = []
    for group, key in (("frequency", "frequency_variants"), ("merge", "merge_variants")):
        for variant in sensitivity_config[key]:
            rows.append(
                evaluate_variant(
                    standardized,
                    data_config,
                    group,
                    variant,
                    sensitivity_config["evaluation"],
                )
            )
    return pd.DataFrame(rows)


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def render_markdown(results: pd.DataFrame, primary_k: int) -> str:
    """Render a compact report containing scale and recommendation effects."""
    rows = [
        "# POI预处理参数敏感性实验",
        "",
        "所有方案均重新执行清洗和全局时间顺序8:1:1划分，并使用训练集候选POI进行完整排序评价。指标仅在训练阶段出现过的测试目标上计算，同时单独报告候选覆盖率。",
        "",
        "## 频率过滤阈值",
        "",
        f"| 方案 | 用户阈值 | POI阈值 | 签到 | 用户 | POI | 稀疏度 | 覆盖率 | Global Acc@{primary_k} | Time Acc@{primary_k} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results.loc[results["experiment_group"].eq("frequency")].iterrows():
        rows.append(
            f"| {row['variant']} | {int(row['min_user_checkins'])} | {int(row['min_poi_visits'])} | "
            f"{int(row['checkins']):,} | {int(row['users']):,} | {int(row['pois']):,} | "
            f"{_format_percent(row['matrix_sparsity'])} | {_format_percent(row['candidate_coverage'])} | "
            f"{row[f'global_acc@{primary_k}']:.4f} | "
            f"{row[f'time_acc@{primary_k}']:.4f} |"
        )
    rows.extend(
        [
            "",
            "## 连续相同POI合并阈值",
            "",
            f"| 方案 | 合并 | 分钟 | 合并记录 | 签到 | 重访率 | 覆盖率 | Global Acc@{primary_k} | Time Acc@{primary_k} |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in results.loc[results["experiment_group"].eq("merge")].iterrows():
        rows.append(
            f"| {row['variant']} | {'是' if row['merge_enabled'] else '否'} | {row['merge_minutes']:g} | "
            f"{int(row['removed_short_repeats']):,} | {int(row['checkins']):,} | "
            f"{_format_percent(row['revisit_rate'])} | {_format_percent(row['candidate_coverage'])} | "
            f"{row[f'global_acc@{primary_k}']:.4f} | {row[f'time_acc@{primary_k}']:.4f} |"
        )
    frequency = results.loc[results["experiment_group"].eq("frequency")]
    merge = results.loc[results["experiment_group"].eq("merge")]
    default_frequency = frequency.loc[
        frequency["variant"].eq("freq_u10_p5_default")
    ].iloc[0]
    lower_poi = frequency.loc[frequency["variant"].eq("freq_u10_p3")].iloc[0]
    higher_user = frequency.loc[frequency["variant"].eq("freq_u20_p5")].iloc[0]
    higher_poi = frequency.loc[frequency["variant"].eq("freq_u10_p10")].iloc[0]
    merge_off = merge.loc[merge["variant"].eq("merge_off")].iloc[0]
    merge_default = merge.loc[merge["variant"].eq("merge_10m_default")].iloc[0]
    removed_by_default_merge = int(merge_off["checkins"] - merge_default["checkins"])
    merge_time_spread = float(
        merge[f"time_acc@{primary_k}"].max() - merge[f"time_acc@{primary_k}"].min()
    )
    rows.extend(
        [
            "",
            "## 实验结论",
            "",
            f"- `POI≥3` 比默认方案多保留 {int(lower_poi['checkins'] - default_frequency['checkins']):,} 条签到和 {int(lower_poi['pois'] - default_frequency['pois']):,} 个POI，但测试候选覆盖率下降 {(default_frequency['candidate_coverage'] - lower_poi['candidate_coverage']):.2%}，Time Acc@{primary_k}下降 {(default_frequency[f'time_acc@{primary_k}'] - lower_poi[f'time_acc@{primary_k}']):.2%}。",
            f"- `POI≥10` 相比默认方案删除 {int(default_frequency['checkins'] - higher_poi['checkins']):,} 条签到和 {int(default_frequency['pois'] - higher_poi['pois']):,} 个POI。指标上升主要伴随候选集合大幅缩小，任务变得更容易，不能单独据此选择严格阈值。",
            f"- 在 `POI≥5` 时，用户阈值10与20结果完全{'一致' if int(default_frequency['checkins']) == int(higher_user['checkins']) else '接近'}；迭代过滤后的最短用户序列实际为 {int(default_frequency['minimum_user_checkins'])}。继续提高用户阈值没有实质收益。",
            f"- 10分钟合并最终减少 {removed_by_default_merge:,} 条签到，占不合并方案的 {removed_by_default_merge / merge_off['checkins']:.2%}；5至30分钟方案的Time Acc@{primary_k}最大差异仅 {merge_time_spread:.4f}。",
            "- 综合记录保留、长尾POI、候选覆盖和指标稳定性，保留 `用户≥10、POI≥5、10分钟合并` 作为主方案是合理的。10分钟并非唯一最优点，而是处于稳定区间内、具有明确去重含义的折中选择。",
            "",
            "## 解释原则",
            "",
            "- 不以最高准确率作为唯一依据，需同时考虑真实记录保留、长尾POI、冷启动覆盖和评估稳定性。",
            "- 合并阈值用于抑制短时间重复上报，不能为了提高指标大量删除真实重访。",
            "- 用户阈值从10开始用于保证序列模型拥有基本历史长度，不是全局时间划分的数学要求。",
            "",
        ]
    )
    return "\n".join(rows)


def save_figures(results: pd.DataFrame, scale_path: Path, performance_path: Path, primary_k: int) -> None:
    """Save report-ready scale and baseline-performance comparisons."""
    scale_path.parent.mkdir(parents=True, exist_ok=True)
    performance_path.parent.mkdir(parents=True, exist_ok=True)
    labels = results["variant"].tolist()
    positions = np.arange(len(results))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].bar(positions, results["checkins"], color="#4C78A8")
    axes[0].set_ylabel("Check-ins")
    axes[0].set_xticks(positions, labels, rotation=35, ha="right")
    axes[0].set_title("Dataset scale under preprocessing variants")
    axes[1].plot(positions, results["candidate_coverage"], marker="o", label="Coverage")
    axes[1].plot(positions, 1.0 - results["matrix_sparsity"], marker="s", label="Density")
    axes[1].set_ylabel("Ratio")
    axes[1].set_xticks(positions, labels, rotation=35, ha="right")
    axes[1].legend()
    fig.savefig(scale_path, dpi=180)
    plt.close(fig)

    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(
        positions - width / 2,
        results[f"global_acc@{primary_k}"],
        width,
        label=f"Global Acc@{primary_k}",
    )
    ax.bar(
        positions + width / 2,
        results[f"time_acc@{primary_k}"],
        width,
        label=f"Time Acc@{primary_k}",
    )
    ax.set_ylabel("Accuracy")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_title("Popularity baselines under preprocessing variants")
    ax.legend()
    fig.savefig(performance_path, dpi=180)
    plt.close(fig)


def save_outputs(
    results: pd.DataFrame,
    sensitivity_config: dict[str, Any],
) -> dict[str, Path]:
    output = sensitivity_config["output"]
    paths = {name: resolve_project_path(value) for name, value in output.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(paths["metrics_csv"], index=False, encoding="utf-8")
    paths["metrics_json"].write_text(
        json.dumps(results.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    primary_k = int(sensitivity_config["evaluation"]["primary_k"])
    paths["report_markdown"].write_text(
        render_markdown(results, primary_k), encoding="utf-8"
    )
    save_figures(
        results,
        paths["scale_figure"],
        paths["performance_figure"],
        primary_k,
    )
    return paths


def run(
    data_config_path: str | Path,
    sensitivity_config_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    data_config = load_yaml(data_config_path)
    sensitivity_config = load_yaml(sensitivity_config_path)
    raw, _ = load_raw_checkins(data_config)
    standardized = standardize_checkins(raw, data_config)
    results = run_sensitivity(standardized, data_config, sensitivity_config)
    paths = save_outputs(results, sensitivity_config)
    return results, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preprocessing sensitivity experiments")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--experiment-config", default="configs/preprocessing_sensitivity.yaml"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results, paths = run(args.data_config, args.experiment_config)
    print(f"Completed {len(results)} preprocessing variants")
    for _, row in results.iterrows():
        print(
            f"{row['variant']}: checkins={int(row['checkins']):,}, "
            f"pois={int(row['pois']):,}, coverage={row['candidate_coverage']:.2%}, "
            f"global_acc10={row['global_acc@10']:.4f}, "
            f"time_acc10={row['time_acc@10']:.4f}"
        )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
