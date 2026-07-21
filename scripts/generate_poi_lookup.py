from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "poi_metadata.csv"
OUTPUT_PATH = ROOT / "app" / "src" / "data" / "poi_lookup.json"


def _emoji_for(category: str) -> str:
    normalized = (category or "").lower()
    if "home" in normalized:
        return "🏠"
    if "subway" in normalized or "station" in normalized:
        return "🚇"
    if "coffee" in normalized or "cafe" in normalized or "café" in normalized:
        return "☕"
    if "office" in normalized:
        return "💼"
    if "restaurant" in normalized or "diner" in normalized:
        return "🍽️"
    if "food truck" in normalized:
        return "🚚"
    if "gym" in normalized:
        return "🏋️"
    if "park" in normalized:
        return "🌳"
    if "bar" in normalized or "pub" in normalized:
        return "🍸"
    if "mall" in normalized or "shopping" in normalized:
        return "🛍️"
    if "bridge" in normalized:
        return "🌉"
    if "school" in normalized or "university" in normalized:
        return "🏫"
    if "grocery" in normalized or "supermarket" in normalized:
        return "🛒"
    if "museum" in normalized or "art" in normalized:
        return "🎨"
    if "cinema" in normalized or "movie" in normalized:
        return "🎬"
    return "📍"


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)
    frame = frame.sort_values(["training_visit_count", "category_confidence"], ascending=False)
    frame = frame.drop_duplicates(subset=["poi_idx"], keep="first")

    lookup = {}
    for _, row in frame.iterrows():
      poi_idx = int(row["poi_idx"])
      category = str(row.get("category_name", "Unknown"))
      lookup[str(poi_idx)] = {
          "poiIdx": poi_idx,
          "poiName": category,
          "category": category,
          "emoji": _emoji_for(category),
          "lat": float(row["latitude"]),
          "lon": float(row["longitude"]),
          "visitCount": int(row.get("training_visit_count", 0)),
      }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(lookup, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()