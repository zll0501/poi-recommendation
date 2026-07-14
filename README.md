# Next POI Recommendation

《人工智能实训II》下一POI推荐项目。

## 项目模型

- Global Popular
- Time Popular
- ItemCF
- Markov
- GRU
- Time-GRU
- SASRec
- Category-SASRec

## 小组分工

- 成员1：数据处理、Global Popular、Time Popular
- 成员2：GRU、Time-GRU
- 成员3：SASRec、Category-SASRec
- 成员4：ItemCF、Markov、实验分析与展示平台

## 项目结构

```text
poi-recommendation/
├── data/          # 原始数据和处理后的数据
├── src/           # 公共代码和模型实现
├── configs/       # 数据及模型配置
├── experiments/   # 实验与消融脚本
├── results/       # 预测、指标、图表和模型权重
├── app/           # Streamlit展示平台
└── tests/         # 自动化测试
```

## 数据准备

将Foursquare NYC原始数据放入：

```text
data/raw/dataset_TSMC2014_NYC.csv
```

原始数据、处理后数据、模型权重和自动生成的实验结果默认不会提交到GitHub。

## 环境安装

建议创建独立Python环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

## 数据预处理

```bash
python -m src.preprocess --config configs/data.yaml
```

当前预处理流程包括：

- 校验并标准化8个原始字段；
- 将UTC时间结合每条记录的时区偏移转换为当地时间；
- 生成小时、星期、周末、时间段及周期时间特征；
- 删除完全重复记录；
- 删除同一用户同一时间的冲突签到组；
- 合并10分钟内连续发生的相同POI签到；
- 迭代保留签到不少于10次的用户和访问不少于5次的POI；
- 保存标准化数据、清洗数据、审计报告和清洗前后统计。

生成文件位于 `data/processed/`：

```text
checkins_standardized.csv
checkins_cleaned.csv
audit_report.json
audit_report.md
cleaning_report.json
cleaning_report.md
```

这些文件由程序自动生成，因此不会提交到GitHub。其他成员可以使用相同配置在本地复现。

## 运行实验

```bash
python run_experiment.py --config configs/global_popular.yaml
```

## 协作规则

- `main`只保存已经检查并可以运行的代码。
- 每名成员在自己的功能分支开发。
- 所有模型使用相同的数据划分、候选集合和评价指标。
- 所有模型输出统一格式的Top-K推荐文件。
- 通过Pull Request审核后再合并到`main`。
