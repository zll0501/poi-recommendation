from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train_encoded.csv"
POI_META_PATH = ROOT / "data" / "processed" / "poi_metadata.csv"
OUTPUT_PATH = ROOT / "app" / "public" / "data" / "trajectories_view.json"


def _derive_borough(lat: float, lon: float) -> str:
    if lat >= 40.79:
        return "Bronx / Upper Manhattan" if lon < -73.95 else "Queens / Bronx"
    if lat >= 40.73:
        return "Manhattan · West Side" if lon < -74.0 else "Manhattan · East Side"
    if lat >= 40.66:
        return "Brooklyn" if lon < -73.95 else "Queens"
    return "NYC"


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


def _time_label(row: pd.Series) -> str:
    local_time = str(row.get("local_time", ""))
    if len(local_time) >= 16 and local_time[11:16].count(":") == 1:
        return local_time[11:16]
    hour = int(row.get("hour", 0))
    return f"{hour:02d}:00"


def build_payload() -> dict:
    train = pd.read_csv(TRAIN_PATH)
    poi_meta = pd.read_csv(POI_META_PATH)

    poi_meta = poi_meta.sort_values(["training_visit_count", "category_confidence"], ascending=False)
    poi_meta = poi_meta.drop_duplicates(subset=["poi_idx"], keep="first")
    poi_meta = poi_meta.set_index("poi_idx")

    grouped = []
    for user_idx, frame in train.groupby("user_idx"):
        frame = frame.sort_values("timestamp")
        if len(frame) < 5:
          continue
        grouped.append((int(user_idx), frame))

    grouped.sort(key=lambda item: len(item[1]), reverse=True)

    users = []
    colors = ["#e11d48", "#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777"]

    for index, (user_idx, frame) in enumerate(grouped[:6]):
        first = frame.iloc[0]
        weekday = str(first.get("weekday", ""))
        borough = _derive_borough(float(first["latitude"]), float(first["longitude"]))

        checkins = []
        for order, (_, row) in enumerate(frame.iterrows(), start=1):
            poi_idx = int(row["poi_idx"])
            meta = poi_meta.loc[poi_idx] if poi_idx in poi_meta.index else None
            category = str(meta["category_name"]) if meta is not None else str(row.get("category_name", "Unknown"))
            latitude = float(meta["latitude"]) if meta is not None else float(row["latitude"])
            longitude = float(meta["longitude"]) if meta is not None else float(row["longitude"])

            checkins.append(
                {
                    "order": order,
                    "poiIdx": poi_idx,
                    "poiName": category,
                    "category": category,
                    "emoji": _emoji_for(category),
                    "lat": latitude,
                    "lon": longitude,
                    "weekday": str(row.get("weekday", weekday)),
                    "hour": int(row.get("hour", 0)),
                    "timeLabel": _time_label(row),
                }
            )

        users.append(
            {
                "userIdx": user_idx,
                "userLabel": f"User #{user_idx}",
                "borough": f"{borough} · {weekday or 'Trajectory'}",
                "color": colors[index % len(colors)],
                "weekday": weekday,
                "checkins": checkins,
            }
        )

    return {
        "generatedAt": pd.Timestamp.utcnow().isoformat(),
        "source": "train_encoded.csv + poi_metadata.csv",
        "users": users,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()