"""Tests for offline attention-result post-processing."""

import json

import pandas as pd

from experiments.analyze_attention_results import analyze_results


def test_postprocessing_filters_special_tokens_and_writes_outputs(tmp_path):
    attention = pd.DataFrame(
        {
            "event_id": [10, 10, 10, 10],
            "user_idx": [2, 2, 2, 2],
            "target_poi_idx": [3, 3, 3, 3],
            "layer": [0, 0, 0, 0],
            "head": [0, 0, 0, 0],
            "query_index": [3, 3, 3, 3],
            "history_position": [0, 1, 2, 3],
            "history_poi_idx": [0, 1, 2, 3],
            "history_poi_id": ["<PAD>", "<UNK>", "poi-a", "poi-b"],
            "attention_weight": [0.10, 0.20, 0.30, 0.40],
        }
    )
    metadata = pd.DataFrame(
        {
            "poi_idx": [2, 3],
            "poi_id": ["poi-a", "poi-b"],
            "category_idx": [5, 6],
            "category_name": ["Cafe", "Park"],
            "latitude": [40.0, 41.0],
            "longitude": [-73.0, -74.0],
        }
    )
    mappings = {"special_tokens": {"PAD": 0, "UNK": 1}}
    attention_path = tmp_path / "attention.csv"
    metadata_path = tmp_path / "metadata.csv"
    mappings_path = tmp_path / "mappings.json"
    output_path = tmp_path / "output"
    attention.to_csv(attention_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    mappings_path.write_text(json.dumps(mappings), encoding="utf-8")

    statistics = analyze_results(
        attention_path,
        metadata_csv=metadata_path,
        mappings_json=mappings_path,
        output_directory=output_path,
        top_k=2,
    )
    topk = pd.read_csv(output_path / "attention_topk_examples.csv")

    assert statistics["input_rows"] == 4
    assert statistics["filtered_rows"] == 2
    assert statistics["removed_special_token_rows"] == 2
    assert topk["poi_idx"].tolist() == [3, 2]
    assert topk["poi_name"].tolist() == ["poi-b", "poi-a"]
    assert topk["category"].tolist() == ["Park", "Cafe"]
    assert (output_path / "attention_statistics.json").exists()
    assert (output_path / "figures" / "attention_top10_bar.png").stat().st_size > 0
    assert (
        output_path / "figures" / "attention_position_distribution.png"
    ).stat().st_size > 0
