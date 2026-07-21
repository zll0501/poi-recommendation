import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# ==========================
# 1. 读取数据
# ==========================
df = pd.read_csv("data/processed/poi_metadata.csv")

# ==========================
# 2. 转GeoDataFrame
# ==========================
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)

# 转成Web Mercator，方便加载地图
gdf = gdf.to_crs(epsg=3857)

# ==========================
# 3. 绘图
# ==========================
fig, ax = plt.subplots(figsize=(10,10))

gdf.plot(
    ax=ax,
    color="red",
    markersize=8,
    alpha=0.6
)

# 添加OpenStreetMap底图
ctx.add_basemap(ax)

ax.set_axis_off()
ax.set_title("Spatial Distribution of POIs in New York", fontsize=16)

plt.tight_layout()

# ==========================
# 4. 保存
# ==========================
plt.savefig(
    "data/figures/figure1_poi_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()