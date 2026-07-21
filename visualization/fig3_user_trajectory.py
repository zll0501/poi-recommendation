import pandas as pd
import folium
from folium.plugins import AntPath

# ==========================
# 1. 读取数据
# ==========================
df = pd.read_csv("data/processed/train_encoded.csv")

# ==========================
# 2. 选择一个用户
# （后面可以改）
# ==========================

# 找签到次数最多的用户
user = df["user_id"].value_counts().idxmax()

user_df = df[df["user_id"] == user].copy()

# 按时间排序
user_df = user_df.sort_values("timestamp")

print(f"User ID: {user}")
print(f"Check-ins: {len(user_df)}")

# ==========================
# 3. 创建地图
# ==========================

center = [
    user_df["latitude"].mean(),
    user_df["longitude"].mean()
]

m = folium.Map(
    location=center,
    zoom_start=13,
    tiles="CartoDB Positron"
)

# ==========================
# 4. 历史轨迹
# ==========================

trajectory = list(
    zip(
        user_df["latitude"],
        user_df["longitude"]
    )
)

AntPath(
    trajectory,
    color="blue",
    weight=4,
    opacity=0.8
).add_to(m)

# ==========================
# 5. 起点
# ==========================

folium.Marker(
    trajectory[0],
    popup="Start",
    icon=folium.Icon(color="green")
).add_to(m)

# ==========================
# 6. 终点
# ==========================

folium.Marker(
    trajectory[-1],
    popup="End",
    icon=folium.Icon(color="red")
).add_to(m)

# ==========================
# 7. 中间签到点
# ==========================

for _, row in user_df.iterrows():

    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['category_name']}"
    ).add_to(m)

# ==========================
# 8. 保存
# ==========================

m.save("data/figures/figure3_user_trajectory.html")

print("Saved: figure3_user_trajectory.html")