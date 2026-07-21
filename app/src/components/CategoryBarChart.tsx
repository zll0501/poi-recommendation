import Plot from "react-plotly.js";

interface Props {
  categories: string[];
  counts: number[];
  title?: string;
}

export default function CategoryBarChart({ categories, counts, title }: Props) {
  const maxCount = Math.max(1, ...counts);

  return (
    <Plot
      data={[
        {
          type: "bar",
          orientation: "h",
          x: counts,
          y: categories,
          marker: {
            color: counts.map((count, index) => `rgba(37, 99, 235, ${0.28 + (index / Math.max(1, counts.length)) * 0.5})`),
            line: { color: "rgba(37, 99, 235, 0.9)", width: 1 },
          },
          hovertemplate: "<b>%{y}</b><br>签到次数: %{x}<extra></extra>",
        },
      ]}
      layout={{
        title: {
          text: title ?? "POI 类别统计",
          x: 0,
          xanchor: "left",
          font: { size: 16, color: "#0f172a" },
        },
        margin: { l: 150, r: 24, t: 48, b: 36 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: {
          range: [0, maxCount * 1.15],
          gridcolor: "rgba(148, 163, 184, 0.18)",
          zeroline: false,
          tickfont: { color: "#64748b", size: 11 },
        },
        yaxis: {
          tickfont: { color: "#334155", size: 11 },
          automargin: true,
        },
        showlegend: false,
        autosize: true,
      }}
      useResizeHandler
      style={{ width: "100%", height: "100%", minHeight: 320 }}
      config={{ displayModeBar: false }}
    />
  );
}