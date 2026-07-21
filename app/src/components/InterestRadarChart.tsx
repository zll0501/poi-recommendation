import React from 'react';
import Plot from 'react-plotly.js';

// 定义传入组件的数据接口
interface InterestRadarChartProps {
  userId?: number;
  categories?: string[];
  counts?: number[];
  percentages?: number[];
}

const InterestRadarChart: React.FC<InterestRadarChartProps> = ({
  userId = 470,
  // 默认数据：注意雷达图的数组首尾必须相同，以闭合多边形
  categories = [
    'Gym / Fitness Center',
    'Arts & Crafts Store',
    'Library',
    'Athletic & Sport',
    'Vegetarian / Vegan Restaurant',
    'Mexican Restaurant',
    'Gym / Fitness Center' // 闭合点
  ],
  counts = [29, 18, 5, 2, 1, 1, 29], // 闭合点
  percentages = [51.8, 32.1, 8.9, 3.6, 1.8, 1.8, 51.8] // 闭合点
}) => {
  
  // 动态计算最大值，留出 1.15 倍的外围空间，避免文字贴边
  const maxCount = Math.max(...counts);

  return (
    <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
      <Plot
        data={[
          {
            type: 'scatterpolar',
            r: counts,
            theta: categories,
            fill: 'toself',
            fillcolor: 'rgba(31, 119, 180, 0.4)', // 半透明蓝色
            line: {
              color: '#1f77b4',
              width: 2
            },
            marker: {
              size: 6,
              color: '#1f77b4'
            },
            // 自定义 Hover 悬停提示文字
            hovertemplate: '<b>%{theta}</b><br>签到次数: %{r}次<br>占比: %{text}%<extra></extra>',
            text: percentages,
            name: `User ${userId}`,
          }
        ]}
        layout={{
          polar: {
            radialaxis: {
              visible: true,
              range: [0, maxCount * 1.15],
              angle: 45, // 刻度线旋转45度，避开轴线文字
              tickfont: { size: 10, color: 'gray' },
              gridcolor: 'rgba(0, 0, 0, 0.1)'
            },
            angularaxis: {
              tickfont: { size: 11, color: '#333333' },
              rotation: 90 // 旋转角度，让图形分布更协调
            }
          },
          title: {
            text: `用户 ${userId} 的历史访问偏好分布 (Top-6)`,
            x: 0.5,
            font: { size: 16, color: '#333333' }
          },
          showlegend: false,
          // 可以通过外部容器控制大小，这里设置为自适应
          autosize: true, 
          margin: { l: 80, r: 80, t: 60, b: 60 } // 增加边距，防止长文本被截断
        }}
        useResizeHandler={true} // 允许图表跟随容器缩放
        style={{ width: '100%', height: '100%', minHeight: '400px' }}
        config={{ displayModeBar: false }} // 隐藏右上角多余的工具栏
      />
    </div>
  );
};

export default InterestRadarChart;