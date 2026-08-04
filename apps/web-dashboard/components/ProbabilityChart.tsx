"use client";

import ReactECharts from "echarts-for-react";

type Point = {
  horizon_days?: number;
  squeeze_probability?: number;
  expected_magnitude_pct?: number;
};

export function ProbabilityChart({ series }: { series: Point[] }) {
  const option = {
    backgroundColor: "transparent",
    textStyle: { color: "#d7dee8" },
    tooltip: { trigger: "axis" },
    legend: { data: ["P(squeeze)", "E[mag]%"], textStyle: { color: "#9aa7b8" } },
    xAxis: {
      type: "category",
      data: series.map((s) => `${s.horizon_days ?? ""}d`),
      axisLabel: { color: "#9aa7b8" },
    },
    yAxis: [
      { type: "value", min: 0, max: 1, axisLabel: { color: "#9aa7b8" }, splitLine: { lineStyle: { color: "#243041" } } },
      { type: "value", axisLabel: { color: "#9aa7b8" }, splitLine: { show: false } },
    ],
    series: [
      {
        name: "P(squeeze)",
        type: "line",
        smooth: true,
        data: series.map((s) => s.squeeze_probability ?? 0),
        itemStyle: { color: "#5eead4" },
      },
      {
        name: "E[mag]%",
        type: "bar",
        yAxisIndex: 1,
        data: series.map((s) => s.expected_magnitude_pct ?? 0),
        itemStyle: { color: "#38bdf8" },
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 320, width: "100%" }} />;
}
