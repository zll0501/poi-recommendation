import pandas as pd
import plotly.graph_objects as go

# 1. 整理8个模型的实验结果数据（请替换为你们跑出的真实结果）
data = [
    {
        "Model": "Global Popular",
        "HitRate@5": 0.022579087177010383,
        "HitRate@10": 0.03127263945906786,
        "NDCG@5": 0.01441165656531247,
        "NDCG@10": 0.017226354740680655,
        "MRR@10": 0.012910960085555595
    },
    {
        "Model": "Time Popular",
        "HitRate@5": 0.021854624486838926,
        "HitRate@10": 0.03344602752958223,
        "NDCG@5": 0.014920983115752951,
        "NDCG@10": 0.01860152875265174,
        "MRR@10": 0.014114109581689875
    },
    {
        "Model": "ItemCF",
        "HitRate@5": 0.14585848828785317,
        "HitRate@10": 0.2853779280367061,
        "NDCG@5": 0.08311979525098066,
        "NDCG@10": 0.1279188309107116,
        "MRR@10": 0.0811281541917258
    },
    {
        "Model": "Markov",
        "HitRate@5": 0.2649118570393625,
        "HitRate@10": 0.31007003139338324,
        "NDCG@5": 0.2001560709743413,
        "NDCG@10": 0.21499830690208532,
        "MRR@10": 0.1848883647075509
    },
    {
        "Model": "GRU",
        "HitRate@5": 0.3935643564356436,
        "HitRate@10": 0.48092248249215164,
        "NDCG@5": 0.28936310316270003,
        "NDCG@10": 0.3176700737772493,
        "MRR@10": 0.2665628365972492
    },
    {
        "Model": "Time-GRU",
        "HitRate@5": 0.41873943491910165,
        "HitRate@10": 0.4864766964501328,
        "NDCG@5": 0.32991353758621444,
        "NDCG@10": 0.35196347570493947,
        "MRR@10": 0.3095885416067723
    },
    {
        "Model": "SASRec",
        "HitRate@5": 0.4392658778072929,
        "HitRate@10": 0.5428036706109636,
        "NDCG@5": 0.3136120253570754,
        "NDCG@10": 0.3473198128254253,
        "MRR@10": 0.2860713183687702
    },
    {
        "Model": "Category-SASRec",
        "HitRate@5": 0.46613136923448445,
        "HitRate@10": 0.5583796184496499,
        "NDCG@5": 0.35324987651478995,
        "NDCG@10": 0.3832846054289824,
        "MRR@10": 0.32823926338626563
    }
]
df = pd.DataFrame(data)

models = df['Model'].tolist()

# 定义两种展示状态的指标列表
metrics_all = ['HitRate@5', 'HitRate@10', 'NDCG@5', 'NDCG@10', 'MRR@10']
metrics_mrr = ['MRR@10']

# 保持同色系深浅对比的消融实验配色
color_map = {
    "Global Popular":  "#CCCCCC", # 浅灰
    "Time Popular":    "#999999", # 深灰
    "ItemCF":          "#9ECAE1", # 浅蓝
    "Markov":          "#3182BD", # 深蓝
    "GRU":             "#FDBE85", # 浅橙
    "Time-GRU":        "#E6550D", # 深橙
    "SASRec":          "#BCBDDC", # 浅紫
    "Category-SASRec": "#756BB1"  # 深紫
}

# 2. 初始化图表
fig = go.Figure()

# 默认绘制全指标的数据轨迹 (Traces)
for model in models:
    y_vals = df[df['Model'] == model][metrics_all].values[0]
    fig.add_trace(go.Bar(
        x=metrics_all,
        y=y_vals,
        name=model,
        marker_color=color_map[model],
        text=y_vals,
        texttemplate='%{text:.3f}',  # 柱状图上保留三位小数
        textposition='outside'
    ))

# 3. 准备下拉菜单的数据切换逻辑
# 预计算切换至 "全指标" 时的 Y 轴数据
y_all_list = [df[df['Model'] == m][metrics_all].values[0] for m in models]
# 预计算切换至 "MRR@10" 时的 Y 轴数据
y_mrr_list = [df[df['Model'] == m][metrics_mrr].values[0] for m in models]

# 4. 配置布局与下拉菜单
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(
                    label="🔍 总体评价：全指标对比看板",
                    method="restyle",
                    # 切换回5个指标的 X 轴和对应数据
                    args=[{"x": [metrics_all] * 8, "y": y_all_list, "text": y_all_list}]
                ),
                dict(
                    label="🎯 首位质量：MRR@10 专项分析",
                    method="restyle",
                    # 切换为仅1个指标的 X 轴和对应数据
                    args=[{"x": [metrics_mrr] * 8, "y": y_mrr_list, "text": y_mrr_list}]
                )
            ]),
            direction="down",
            showactive=True,
            x=0.01,           # 菜单在左上角
            xanchor="left",
            y=1.2,            # 调整高度避免遮挡标题
            yanchor="top",
            bgcolor="white",
            bordercolor="#e5e5e5"
        )
    ],
    title=dict(
        text="<b>下一POI推荐：多模型性能对比与消融实验评估</b>", 
        y=0.9, x=0.5, xanchor='center', yanchor='top'
    ),
    xaxis_title="<b>评估指标 (Metrics)</b>",
    yaxis_title="<b>得分 (Score)</b>",
    legend_title="<b>模型层级与消融对比</b>",
    font=dict(size=14, family="Arial"),
    barmode="group",
    hovermode="x unified",    # 鼠标悬浮时展示同组所有信息
    bargap=0.15,
    bargroupgap=0.05,
    plot_bgcolor="#f9f9f9",   # 浅灰色背景让柱体更突出
    margin=dict(t=150)        # 给顶部的下拉菜单留出充足空间
)

# 给图表增加轻微的网格线方便数值对比
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='white')

# 5. 导出为交互式 HTML 页面
html_filename = "model_evaluation_dashboard.html"
# include_plotlyjs='cdn' 表示图表库通过网络加载，可以让生成的 HTML 文件体积保持在几百KB以内
fig.write_html(html_filename, include_plotlyjs='cdn')

print(f"✅ 图表已成功导出为: {html_filename}")
print("直接双击该文件即可在浏览器中进行展示和交互操作。")