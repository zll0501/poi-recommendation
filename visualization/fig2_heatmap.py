import pandas as pd
import folium
from folium.plugins import HeatMap

# ==========================
# 1. 读取数据
# ==========================
df = pd.read_csv("data/processed/poi_metadata.csv")

# ==========================
# 2. 构造热力图数据
# [纬度，经度，权重]
# ==========================
heat_data = df[
    ["latitude", "longitude", "training_visit_count"]
].values.tolist()

# ==========================
# 3. 创建地图
# ==========================
m = folium.Map(
    location=[40.74, -73.98],   # NYC中心
    zoom_start=11,
    tiles="CartoDB Positron"
)

# ==========================
# 4. 添加热力图
# ==========================
HeatMap(
    heat_data,

    # ===== 推荐参数 =====
    radius=16,
    blur=22,
    max_zoom=13,
    min_opacity=0.35,

).add_to(m)

# ==========================
# 5. 保存
# ==========================
m.save("data/figures/figure2_heatmap.html")

print("Heatmap saved.")