# 基于多源数据融合的下一 POI 推荐

本项目使用 Foursquare NYC 签到数据，在统一协议下比较 Global Popular、Time Popular、ItemCF、Markov、GRU、Time-GRU、SASRec 和 Category-SASRec。

## 任务定义

给定用户在推荐时刻之前的签到历史，以及已知的推荐请求时间，预测下一次访问的 POI。目标 POI 和推荐时刻之后的事件均不可见。

- 主协议：按全体签到的 UTC 时间执行全局 80% / 10% / 10% 划分。
- 训练集用于拟合模型、ID 映射、POI 元数据和全部统计量。
- 验证集用于调参与早停；测试集只用于最终报告。
- 所有模型使用训练集 POI 作为相同候选集。
- 主指标只计算“训练集中已知用户且已知 POI”的闭集样本，同时单独报告覆盖率。
- Time Popular 和 Time-GRU 可以使用已知请求时间；不能使用目标地点产生的距离等特征。

全局时间划分使 Global Popular、Time Popular 和序列模型遵守同一系统时间线，避免使用其他用户的未来签到预测较早事件。

## 数据流水线

```text
原始签到
  → 字段与时区标准化
  → 规则清洗与迭代过滤
  → 全局时间划分
  → 仅训练集拟合 ID 和权威 POI 元数据
  → 统一数据接口
  → 模型训练与统一评估
```

数据放置位置：

```text
data/raw/dataset_TSMC2014_NYC.csv
```

依次运行：

```bash
python -m src.preprocess --config configs/data.yaml
python -m src.split_data --config configs/data.yaml
python -m src.encode_data --config configs/data.yaml
python -m src.analysis.data_profile --config configs/data.yaml
pytest -q
```

### 下载纽约逐小时天气旁路数据

天气数据与现有推荐数据接口相互独立，不会改变签到划分或已有模型输入。运行：

```powershell
python -m src.download_weather --config configs/weather.yaml
```

脚本从 Open-Meteo Historical Weather API 下载 ERA5 逐小时数据，输出到
`data/external/weather/nyc_hourly_weather.csv`，并生成包含请求参数、质量检查和
SHA256 校验值的 `nyc_hourly_weather_metadata.json`。生成的数据文件默认不提交到 Git。

预处理包含：

- 统一字段类型、UTC 时间和 `America/New_York` 本地时间；
- 删除完全重复记录；
- 删除同一用户同一时刻对应多个 POI 的冲突组；
- 合并 10 分钟内连续相同 POI；
- 迭代保留签到不少于 10 次的用户和访问不少于 5 次的 POI；
- 输出清洗前后审计报告。

迭代过滤作用于完整语料，用于定义研究对象；这是一项公开的数据集纳入规则，不宣称为训练集拟合步骤。划分后的映射、元数据、模型统计量必须严格只用训练集拟合。

## 最小共享产物

`data/processed/` 中仅保留以下可解释产物：

```text
checkins_cleaned.csv             # 唯一清洗主表
audit_report.json / .md          # 原始质量审计
cleaning_report.json / .md       # 清洗过程与参数
train.csv / valid.csv / test.csv # 全局时间划分
split_report.json / .md          # 时间边界、覆盖率与一致性检查
train_encoded.csv
valid_encoded.csv
test_encoded.csv                 # 模型统一输入
id_mappings.json                 # 仅训练集拟合，0=PAD、1=UNK、2起为正常ID
poi_metadata.csv                 # 训练集拟合的权威类别与中位数坐标
encoding_report.json / .md
data_profile.json / .md          # 汇报用的紧凑数据画像
```

标准化全量表、重复的事件特征表、逐用户/POI画像表和可疑轨迹明细不持久化；需要时由原始数据和代码重新计算。

## 统一接口

```python
from src.datasets import load_data_bundle

data = load_data_bundle("configs/data.yaml")
train, valid, test = data.train, data.validation, data.test
candidate_pois = data.candidate_poi_ids

for sample in data.iter_next_poi_samples("test", max_history=50):
    user = sample.user_idx
    history = sample.history
    target = sample.target_poi_idx
    request_hour = sample.hour
```

- Popular、ItemCF、Markov 可直接读取三个 DataFrame。
- GRU、SASRec 使用 `user_sequences()` 或 `iter_next_poi_samples()`。
- 验证/测试采用滚动历史：每次预测只能看到当前事件之前的真实签到。
- 未知目标映射为 UNK，默认不进入闭集主指标，但必须进入覆盖率统计。

## 参数敏感性实验

```bash
python -m experiments.preprocessing_sensitivity \
  --data-config configs/data.yaml \
  --experiment-config configs/preprocessing_sensitivity.yaml
```

该实验复用同一个全局时间划分函数、训练候选集和已知请求时间假设，用于比较过滤阈值与连续签到合并阈值。结果是预处理选择依据，不替代最终模型实验。

## 小组分工

- 成员 1：数据处理、Global Popular、Time Popular。
- 成员 2：GRU、Time-GRU。
- 成员 3：SASRec、Category-SASRec。
- 成员 4：ItemCF、Markov、实验分析与平台展示。

所有成员必须使用同一划分、候选集、滚动历史和评价器，不在各自模型脚本中重新切分数据。

## 统一评价接口

每个模型输出相同的长表格式：

```text
event_id,rank,poi_idx
10001,1,25
10001,2,103
10001,3,8
```

每个可评价事件必须提交恰好10个互不重复、属于训练候选集的POI。统一评价规则为：

- `HitRate@5/10`：真实下一POI是否出现在前5/10名；
- `NDCG@5/10`：命中位置越靠前，得分越高；
- `MRR@10`：真实POI排名倒数的平均值；
- `Coverage`：全部目标中“训练已知用户且训练已知POI”的比例。

```python
from src.evaluator import evaluate_next_poi

metrics = evaluate_next_poi(
    targets=data.test,
    predictions=prediction_frame,
    candidate_poi_ids=data.candidate_poi_ids,
    unknown_id=data.unknown_id,
    ks=(5, 10),
    mrr_k=10,
)
```

评价器会自动排除冷启动目标，并拒绝缺失事件、重复POI、非法排名和候选集外POI，避免不同模型使用不一致的评价范围。

统一参数保存在 `configs/evaluation.yaml`。由于下一POI任务包含真实的重复访问行为，默认不排除用户历史访问过的POI。
