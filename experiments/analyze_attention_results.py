"""Post-process saved SASRec attention weights for reports and presentations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    "results/attention_analysis/sasrec_query_time_category/attention_examples.csv"
)
DEFAULT_OUTPUT = "results/attention_analysis/sasrec_query_time_category"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_special_tokens(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as file:
        mappings = json.load(file)
    tokens = mappings["special_tokens"]
    return int(tokens["PAD"]), int(tokens["UNK"])


def _load_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    required = {"poi_idx", "poi_id", "category_idx", "category_name"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"POI metadata is missing columns: {missing}")
    metadata = metadata.drop_duplicates("poi_idx", keep="first").copy()
    # 当前数据没有场所名称，用原始 Foursquare POI ID 作为稳定可识别名称。
    if "poi_name" not in metadata.columns:
        metadata["poi_name"] = metadata["poi_id"].astype(str)
    return metadata


def _enrich_attention(attention: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id",
        "user_idx",
        "target_poi_idx",
        "layer",
        "head",
        "query_index",
        "history_position",
        "history_poi_idx",
        "attention_weight",
    }
    missing = sorted(required.difference(attention.columns))
    if missing:
        raise ValueError(f"attention CSV is missing columns: {missing}")

    history_columns = [
        "poi_idx",
        "poi_id",
        "poi_name",
        "category_idx",
        "category_name",
    ]
    for optional in ("latitude", "longitude"):
        if optional in metadata.columns:
            history_columns.append(optional)
    history_metadata = metadata[history_columns].rename(
        columns={
            "poi_idx": "history_poi_idx",
            "poi_id": "mapped_history_poi_id",
            "category_idx": "mapped_history_category_idx",
            "category_name": "category",
        }
    )
    target_metadata = metadata[
        ["poi_idx", "poi_id", "poi_name", "category_idx", "category_name"]
    ].rename(
        columns={
            "poi_idx": "target_poi_idx",
            "poi_id": "target_poi",
            "poi_name": "target_poi_name",
            "category_idx": "target_category_idx",
            "category_name": "target_category",
        }
    )
    enriched = attention.merge(
        history_metadata, on="history_poi_idx", how="left", validate="many_to_one"
    ).merge(
        target_metadata, on="target_poi_idx", how="left", validate="many_to_one"
    )
    enriched["poi_idx"] = enriched["history_poi_idx"].astype(int)
    enriched["poi_name"] = enriched["poi_name"].fillna(
        enriched.get("history_poi_id", enriched["history_poi_idx"].astype(str))
    )
    enriched["category"] = enriched["category"].fillna("Unknown category")
    enriched["history_distance"] = (
        enriched["query_index"].astype(int)
        - enriched["history_position"].astype(int)
    )
    return enriched


def _topk_rows(enriched: pd.DataFrame, top_k: int) -> pd.DataFrame:
    group_columns = ["event_id", "layer", "head"]
    ranked = enriched.sort_values(
        group_columns + ["attention_weight", "history_position"],
        ascending=[True, True, True, False, False],
        kind="stable",
    )
    topk = ranked.groupby(group_columns, sort=False, observed=True).head(top_k).copy()
    topk["attention_rank"] = (
        topk.groupby(group_columns, sort=False, observed=True).cumcount() + 1
    )
    columns = [
        "event_id",
        "user_idx",
        "layer",
        "head",
        "attention_rank",
        "history_position",
        "history_distance",
        "poi_idx",
        "poi_name",
        "category",
        "attention_weight",
        "target_poi",
        "target_category",
    ]
    for optional in ("latitude", "longitude"):
        if optional in topk.columns:
            columns.append(optional)
    return topk[columns]


def _safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _statistics(enriched: pd.DataFrame, topk: pd.DataFrame, top_k: int) -> dict[str, Any]:
    group_columns = ["event_id", "layer", "head"]
    group_stats: list[dict[str, Any]] = []
    for (event_id, layer, head), group in enriched.groupby(
        group_columns, sort=True, observed=True
    ):
        kept_mass = float(group["attention_weight"].sum())
        weighted_distance = (
            float(
                (group["attention_weight"] * group["history_distance"]).sum()
                / kept_mass
            )
            if kept_mass > 0
            else None
        )
        earliest = group.loc[group["history_distance"].idxmax()]
        recent = group.loc[group["history_distance"].idxmin()]
        group_stats.append(
            {
                "event_id": int(event_id),
                "layer": int(layer),
                "head": int(head),
                "valid_history_count": int(len(group)),
                "attention_mass_after_token_filter": kept_mass,
                "attention_weighted_mean_distance": weighted_distance,
                "most_recent_valid_weight": float(recent["attention_weight"]),
                "earliest_valid_weight": float(earliest["attention_weight"]),
            }
        )

    position_distribution = (
        enriched.groupby(["layer", "history_distance"], observed=True)[
            "attention_weight"
        ]
        .mean()
        .reset_index(name="mean_attention_weight")
        .sort_values(["layer", "history_distance"])
    )
    same_category = topk["category"].eq(topk["target_category"])
    category_by_layer = []
    for layer, group in topk.assign(same_target_category=same_category).groupby(
        "layer", observed=True
    ):
        category_by_layer.append(
            {
                "layer": int(layer),
                "same_target_category_count": int(
                    group["same_target_category"].sum()
                ),
                "topk_count": int(len(group)),
                "same_target_category_rate": float(
                    group["same_target_category"].mean()
                ),
            }
        )

    stats_frame = pd.DataFrame(group_stats)
    return {
        "definitions": {
            "history_distance": "query_index - history_position; 0 is most recent",
            "token_filter": "PAD and UNK history POIs are excluded",
            "category_consistency": f"share of Top{top_k} rows matching target category",
            "poi_name_fallback": "raw poi_id because current metadata has no venue name",
        },
        "sample_count": int(enriched["event_id"].nunique()),
        "layer_count": int(enriched["layer"].nunique()),
        "head_count_per_layer": int(enriched["head"].nunique()),
        "top_k": int(top_k),
        "overall": {
            "attention_weighted_mean_distance": _safe_float(
                stats_frame["attention_weighted_mean_distance"].mean()
            ),
            "most_recent_valid_mean_weight": _safe_float(
                stats_frame["most_recent_valid_weight"].mean()
            ),
            "earliest_valid_mean_weight": _safe_float(
                stats_frame["earliest_valid_weight"].mean()
            ),
            "mean_attention_mass_after_token_filter": _safe_float(
                stats_frame["attention_mass_after_token_filter"].mean()
            ),
            "topk_same_target_category_rate": _safe_float(same_category.mean()),
        },
        "by_layer_head": group_stats,
        "category_consistency_by_layer": category_by_layer,
        "position_distribution": position_distribution.to_dict(orient="records"),
    }


def _plot_top10(enriched: pd.DataFrame, path: Path) -> None:
    event_id = int(enriched["event_id"].min())
    event = enriched.loc[enriched["event_id"].eq(event_id)]
    last_layer = int(event["layer"].max())
    averaged = (
        event.loc[event["layer"].eq(last_layer)]
        .groupby(
            ["history_position", "poi_name", "category"],
            as_index=False,
            observed=True,
        )["attention_weight"]
        .mean()
        .nlargest(10, "attention_weight")
        .sort_values("attention_weight")
    )
    labels = [
        f"pos {int(row.history_position)} | {row.poi_name} | {row.category}"
        for row in averaged.itertuples(index=False)
    ]
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.barh(labels, averaged["attention_weight"], color="#4472C4")
    axis.set_xlabel("Mean attention weight across heads")
    axis.set_title(f"Event {event_id}: Top-10 history attention (layer {last_layer})")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_position_distribution(enriched: pd.DataFrame, path: Path) -> None:
    distribution = (
        enriched.groupby(["layer", "history_distance"], observed=True)[
            "attention_weight"
        ]
        .mean()
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(9, 5))
    for layer, group in distribution.groupby("layer", sort=True, observed=True):
        axis.plot(
            group["history_distance"],
            group["attention_weight"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"Layer {int(layer)}",
        )
    axis.set_xlabel("History distance (0 = most recent)")
    axis.set_ylabel("Mean attention weight")
    axis.set_title("Attention weight by historical distance")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_results(
    attention_csv: str | Path = DEFAULT_INPUT,
    *,
    metadata_csv: str | Path = "data/processed/poi_metadata.csv",
    mappings_json: str | Path = "data/processed/id_mappings.json",
    output_directory: str | Path = DEFAULT_OUTPUT,
    top_k: int = 5,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    attention_path = _project_path(attention_csv)
    metadata_path = _project_path(metadata_csv)
    mappings_path = _project_path(mappings_json)
    if not attention_path.exists():
        raise FileNotFoundError(f"attention input does not exist: {attention_path}")
    attention = pd.read_csv(attention_path)
    metadata = _load_metadata(metadata_path)
    pad_id, unknown_id = _load_special_tokens(mappings_path)
    enriched = _enrich_attention(attention, metadata)
    filtered = enriched.loc[
        ~enriched["history_poi_idx"].isin([pad_id, unknown_id])
    ].copy()
    if filtered.empty:
        raise ValueError("no valid POI remains after PAD/UNK filtering")

    topk = _topk_rows(filtered, top_k)
    statistics = _statistics(filtered, topk, top_k)
    statistics["input_rows"] = int(len(attention))
    statistics["filtered_rows"] = int(len(filtered))
    statistics["removed_special_token_rows"] = int(len(attention) - len(filtered))

    output_dir = _project_path(output_directory)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    topk.to_csv(
        output_dir / "attention_topk_examples.csv", index=False, encoding="utf-8"
    )
    (output_dir / "attention_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_top10(filtered, figures_dir / "attention_top10_bar.png")
    _plot_position_distribution(
        filtered, figures_dir / "attention_position_distribution.png"
    )
    return statistics


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process saved SASRec attention")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--metadata", default="data/processed/poi_metadata.csv")
    parser.add_argument("--mappings", default="data/processed/id_mappings.json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    statistics = analyze_results(
        args.input,
        metadata_csv=args.metadata,
        mappings_json=args.mappings,
        output_directory=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
