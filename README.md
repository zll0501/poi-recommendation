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
data/raw/dataset_TSMC2014_NYC.txt
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
