"""生成与最新推荐事件严格对齐的单用户轨迹视图数据。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_USERS_PATH = PROJECT_ROOT / "app" / "src" / "data" / "sample_users.json"
RECOMMENDATIONS_PATH = (
    PROJECT_ROOT / "app" / "src" / "data" / "user_recommendations.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "app" / "public" / "data" / "trajectories_view.json"
HISTORY_LIMIT = 50
COLORS = ("#e11d48", "#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777")
WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _derive_borough(latitude: float, longitude: float) -> str:
    """根据坐标生成仅用于界面展示的区域标签。"""
    if latitude >= 40.79:
        return "Bronx / Upper Manhattan" if longitude < -73.95 else "Queens / Bronx"
    if latitude >= 40.73:
        return "Manhattan · West Side" if longitude < -74.0 else "Manhattan · East Side"
    if latitude >= 40.66:
        return "Brooklyn" if longitude < -73.95 else "Queens"
    return "NYC"


def _emoji_for(category: str) -> str:
    """按照POI类别返回展示图标。"""
    normalized = category.lower()
    rules = (
        (("home",), "🏠"),
        (("subway", "station"), "🚇"),
        (("coffee", "cafe", "café"), "☕"),
        (("office",), "💼"),
        (("restaurant", "diner"), "🍽️"),
        (("food truck",), "🚚"),
        (("gym",), "🏋️"),
        (("park",), "🌳"),
        (("bar", "pub"), "🍸"),
        (("mall", "shopping"), "🛍️"),
        (("bridge",), "🌉"),
        (("school", "university"), "🏫"),
        (("grocery", "supermarket"), "🛒"),
        (("museum", "art"), "🎨"),
        (("cinema", "movie"), "🎬"),
    )
    for keywords, emoji in rules:
        if any(keyword in normalized for keyword in keywords):
            return emoji
    return "📍"


def _weekday_label(value: Any) -> str:
    """将编码后的星期值转换为中文标签。"""
    weekday = int(value)
    return WEEKDAY_NAMES[weekday] if 0 <= weekday < len(WEEKDAY_NAMES) else "未知"


def _time_label(row: pd.Series) -> str:
    """从本地时间字段提取小时和分钟。"""
    local_time = str(row.get("local_time", ""))
    if len(local_time) >= 16 and local_time[11:16].count(":") == 1:
        return local_time[11:16]
    return f"{int(row.get('hour', 0)):02d}:00"


def _checkin_payload(row: pd.Series, order: int) -> dict[str, Any]:
    """将一条编码签到记录转换为前端字段。"""
    category = str(row.get("category_name", "Unknown"))
    return {
        "eventId": int(row["event_id"]),
        "order": order,
        "poiIdx": int(row["poi_idx"]),
        "poiName": category,
        "category": category,
        "emoji": _emoji_for(category),
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"]),
        "weekday": _weekday_label(row.get("weekday", -1)),
        "hour": int(row.get("hour", 0)),
        "timeLabel": _time_label(row),
    }


def _read_checkins() -> pd.DataFrame:
    """读取三个时间分区，并保留生成视图所需字段。"""
    columns = (
        "event_id",
        "user_id",
        "user_idx",
        "poi_idx",
        "category_name",
        "latitude",
        "longitude",
        "local_time",
        "timestamp",
        "hour",
        "weekday",
    )
    frames = []
    for filename in ("train_encoded.csv", "valid_encoded.csv", "test_encoded.csv"):
        path = PROCESSED_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少轨迹生成输入文件：{path}")
        frames.append(pd.read_csv(path, usecols=columns))
    return pd.concat(frames, ignore_index=True).sort_values("event_id", kind="stable")


def build_payload(history_limit: int = HISTORY_LIMIT) -> dict[str, Any]:
    """构建样本用户最近历史、预测事件和真实下一POI。"""
    if history_limit <= 0:
        raise ValueError("history_limit必须为正整数")

    sample_user_ids = json.loads(SAMPLE_USERS_PATH.read_text(encoding="utf-8"))
    if not isinstance(sample_user_ids, list):
        raise TypeError("sample_users.json顶层必须是列表")

    recommendations = pd.read_csv(RECOMMENDATIONS_PATH)
    checkins = _read_checkins()
    users: list[dict[str, Any]] = []
    unavailable_user_ids: list[int] = []

    for user_id_value in sample_user_ids:
        user_id = int(user_id_value)
        user_recommendations = recommendations[recommendations["user_id"] == user_id]
        if user_recommendations.empty:
            unavailable_user_ids.append(user_id)
            continue

        prediction_event_id = int(user_recommendations["event_id"].max())
        event_recommendations = user_recommendations[
            user_recommendations["event_id"] == prediction_event_id
        ]
        if sorted(event_recommendations["rank"].astype(int).tolist()) != list(range(1, 11)):
            raise ValueError(f"用户{user_id}的事件{prediction_event_id}不是完整Top-10")

        target_rows = checkins[
            (checkins["user_id"] == user_id)
            & (checkins["event_id"] == prediction_event_id)
        ]
        if len(target_rows) != 1:
            raise ValueError(
                f"用户{user_id}的预测事件{prediction_event_id}应唯一对应一条目标记录"
            )
        target_row = target_rows.iloc[0]

        history = checkins[
            (checkins["user_id"] == user_id)
            & (checkins["event_id"] < prediction_event_id)
        ].tail(history_limit)
        if history.empty:
            unavailable_user_ids.append(user_id)
            continue

        user_idx = int(target_row["user_idx"])
        recommendation_user_indices = set(event_recommendations["user_idx"].astype(int))
        if recommendation_user_indices != {user_idx}:
            raise ValueError(f"用户{user_id}的推荐user_idx与目标事件不一致")

        last_history = history.iloc[-1]
        target_weekday = _weekday_label(target_row.get("weekday", -1))
        users.append(
            {
                "userId": user_id,
                "userIdx": user_idx,
                "userLabel": f"User {user_id}",
                "borough": (
                    f"{_derive_borough(float(last_history['latitude']), float(last_history['longitude']))}"
                    f" · {target_weekday}预测"
                ),
                "color": COLORS[len(users) % len(COLORS)],
                "weekday": target_weekday,
                "predictionEventId": prediction_event_id,
                "checkins": [
                    _checkin_payload(row, order)
                    for order, (_, row) in enumerate(history.iterrows(), start=1)
                ],
                "targetCheckin": _checkin_payload(target_row, len(history) + 1),
            }
        )

    return {
        "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": (
            "train_encoded.csv + valid_encoded.csv + test_encoded.csv + "
            "user_recommendations.csv"
        ),
        "historyLimit": history_limit,
        "requestedSampleCount": len(sample_user_ids),
        "unavailableUserIds": unavailable_user_ids,
        "users": users,
    }


def main() -> None:
    """生成并写入Vite public目录。"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"已生成{OUTPUT_PATH}：可展示用户{len(payload['users'])}人，"
        f"不可用用户{payload['unavailableUserIds']}"
    )


if __name__ == "__main__":
    main()
