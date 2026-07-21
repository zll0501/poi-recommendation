import pandas as pd
import plotly.graph_objects as go

# 1. 读取训练集数据 (请替换为你的实际文件路径)
# 假设文件名为 train_encoded.csv
df_train = pd.read_csv('data/processed/train_encoded.csv')
target_user_id = 470 
user_data = df_train[df_train['user_id'] == target_user_id]

# 2. 统计频次与百分比
category_counts = user_data['category_name'].value_counts().reset_index()
category_counts.columns = ['category_name', 'visit_count']

# 计算百分比
total_visits = category_counts['visit_count'].sum()
category_counts['percentage'] = (category_counts['visit_count'] / total_visits * 100).round(1)

# 3. 提取 Top-6
top_k = 6
top_categories = category_counts.head(top_k)

categories = top_categories['category_name'].tolist()
counts = top_categories['visit_count'].tolist()
percentages = top_categories['percentage'].tolist()

# 4. 闭合多边形
categories.append(categories[0])
counts.append(counts[0])
percentages.append(percentages[0])

# 5. 绘制优化后的雷达图
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=counts,
    theta=categories,
    fill='toself',
    fillcolor='rgba(31, 119, 180, 0.4)', # 半透明蓝色，质感更好
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=6, color='#1f77b4'),
    # 自定义 Hover 悬停提示文字：显示类别、次数和百分比
    hovertemplate='<b>%{theta}</b><br>签到次数: %{r}次<br>占比: %{text}%<extra></extra>',
    text=percentages,
    name=f'User {target_user_id}'
))

# 6. 视觉细节调优
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, max(counts) * 1.15], # 留出一点外围空间，避免文字贴边
            angle=45,                      # 刻度线旋转45度，避开轴线文字
            tickfont=dict(size=10, color='gray'),
            gridcolor='rgba(0, 0, 0, 0.1)'
        ),
        angularaxis=dict(
            tickfont=dict(size=11, color='#333333'),
            rotation=90 # 旋转角度，让图形分布更协调
        )
    ),
    title=dict(
        text=f"用户 {target_user_id} 的历史访问偏好分布 (Top-{top_k})",
        x=0.5,
        font=dict(size=16, color='#333333')
    ),
    showlegend=False,
    margin=dict(l=80, r=80, t=60, b=60) # 增加边距，防止长文本被截断
)

# 展示图表
fig.show(renderer="browser")