from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "src" / "data" / "user_recommendations.csv"
OUTPUT_PATH = ROOT / "src" / "data" / "user_recommendations.json"


def build_payload() -> dict:
    frame = pd.read_csv(INPUT_PATH)
    users = []

    for user_id, user_frame in frame.groupby("user_id", sort=False):
        latest_event_id = int(user_frame["event_id"].max())
        latest_frame = user_frame[user_frame["event_id"] == latest_event_id].sort_values("rank")
        user_idx = int(user_frame["user_idx"].iloc[0])

        users.append(
            {
                "userId": int(user_id),
                "userIdx": user_idx,
                "latestEventId": latest_event_id,
                "eventCount": int(user_frame["event_id"].nunique()),
                "topK": [
                    {
                        "rank": int(row["rank"]),
                        "poiIdx": int(row["poi_idx"]),
                        "poiId": str(row["poi_id"]),
                        "categoryName": str(row["category_name"]),
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "trainingVisitCount": int(row["training_visit_count"]),
                    }
                    for _, row in latest_frame.iterrows()
                ],
            }
        )

    users.sort(key=lambda item: item["userId"])
    return {
        "generatedAt": pd.Timestamp.utcnow().isoformat(),
        "source": "user_recommendations.csv",
        "users": users,
    }


def main() -> None:
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()