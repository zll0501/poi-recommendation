user_recommendations.csv 使用说明
文件说明

user_recommendations.csv 为推荐模型（SASRec + Query + Time + Category）最终输出的推荐结果文件。

文件中的每一行表示：

某个用户在某一次预测事件（event）下，第 rank 名推荐的 POI（兴趣点）。

前端可直接读取该文件进行推荐列表展示和地图可视化，无需再进行数据解码。

字段说明
字段	类型	说明
user_id	int	数据集中的真实用户 ID，用于用户选择或检索。
user_idx	int	模型内部编码后的用户编号，仅供调试使用，前端一般无需使用。
event_id	int	一次预测事件的唯一编号。同一用户可能存在多个 event。
rank	int	推荐排名，范围为 1~10，数字越小表示推荐优先级越高。
poi_idx	int	模型内部 POI 编号，仅供调试使用。
poi_id	string	POI 的真实唯一标识（Foursquare Venue ID）。
category_name	string	推荐地点所属类别，如 Coffee Shop、Park、Bar 等。
latitude	float	推荐地点纬度。
longitude	float	推荐地点经度。
training_visit_count	int	该 POI 在训练集中的签到次数，可作为地点热度参考。
数据组织方式

例如：

user_id	event_id	rank	category_name
2	160805	1	Gym / Fitness Center
2	160805	2	Department Store
...	...	...	...
2	160805	10	Neighborhood

表示：

用户 2 在预测事件 160805 中，模型生成了 Top-10 推荐结果。

随后：

user_id	event_id	rank
2	160843	1
2	160843	2
...		

说明：

同一用户进行了下一次预测，因此生成了另一组 Top-10 推荐。

因此：

一个用户（user）可以对应多个预测事件（event）。
每个 event 固定包含 10 条推荐记录。
前端推荐展示

建议前端按照以下流程展示：

选择用户
      │
      ▼
读取该用户所有 event_id
      │
      ▼
选择一个 event（通常选择最新一个）
      │
      ▼
按照 rank 升序排序
      │
      ▼
展示 Top-10 推荐地点
地图可视化

可直接使用：

latitude
longitude

作为 Marker 坐标。

Marker Popup 建议显示：

Rank：1

Category：Coffee Shop

POI ID：4ace6c89f964a52078d020e3

Training Visits：138
注意事项

同一用户可能对应多个 event。

推荐结果并不是固定一份，而是针对不同预测时刻生成的。

rank=1 表示 Top-1 推荐。

推荐优先级依次降低至 rank=10。

category_name 表示地点类别，而不是地点名称。

当前数据集中未提供 POI 的商业名称（如 Starbucks、Central Park），因此展示类别信息作为地点描述。

poi_id 为 Foursquare Venue ID。

如需进一步查询地点详情，可利用该 ID 与外部 POI 数据进行关联。

推荐前端调用示例（Python）
import pandas as pd

df = pd.read_csv("user_recommendations.csv")

# 获取用户2最新一次推荐
user_df = df[df["user_id"] == 2]

latest_event = user_df["event_id"].max()

top10 = (
    user_df[user_df["event_id"] == latest_event]
    .sort_values("rank")
)

print(top10)

推荐使用方式：
前端启动时读取 user_recommendations.csv，根据用户选择筛选对应 event_id 的 Top-10 数据，并利用 latitude、longitude 绘制地图 Marker，同时按 rank 顺序展示推荐列表。这样无需再次访问模型或进行额外的数据解码。