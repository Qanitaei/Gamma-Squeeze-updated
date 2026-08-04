"use client";

import ReactECharts from "echarts-for-react";

type Props = {
  title: string;
  x?: (string | number)[];
  y?: (string | number)[];
  z?: ((number | null)[] | undefined)[];
  kind?: "heatmap" | "bar";
};

export function HeatmapPanel({ title, x = [], y = [], z = [], kind = "heatmap" }: Props) {
  if (kind === "bar") {
    const option = {
      backgroundColor: "transparent",
      title: { text: title, left: 0, textStyle: { color: "#c9d4e5", fontSize: 13 } },
      grid: { left: 48, right: 16, top: 36, bottom: 40 },
      xAxis: { type: "category", data: x, axisLabel: { color: "#8b9bb4" } },
      yAxis: { type: "value", axisLabel: { color: "#8b9bb4" }, splitLine: { lineStyle: { color: "#243049" } } },
      series: [{ type: "bar", data: y, itemStyle: { color: "#3dd6c6" } }],
    };
    return <ReactECharts option={option} style={{ height: 320, width: "100%" }} />;
  }

  const flat = (z || []).flat().filter((v) => v != null) as number[];
  const min = flat.length ? Math.min(...flat) : 0;
  const max = flat.length ? Math.max(...flat) : 1;
  const data: [number, number, number][] = [];
  (z || []).forEach((row, yi) => {
    (row || []).forEach((val, xi) => {
      if (val != null) data.push([xi, yi, val]);
    });
  });

  const option = {
    backgroundColor: "transparent",
    title: { text: title, left: 0, textStyle: { color: "#c9d4e5", fontSize: 13 } },
    tooltip: { position: "top" },
    grid: { left: 48, right: 24, top: 36, bottom: 40 },
    xAxis: { type: "category", data: x, axisLabel: { color: "#8b9bb4" }, name: "Strike", nameTextStyle: { color: "#8b9bb4" } },
    yAxis: { type: "category", data: y, axisLabel: { color: "#8b9bb4" }, name: "DTE", nameTextStyle: { color: "#8b9bb4" } },
    visualMap: {
      min,
      max,
      calculable: true,
      orient: "vertical",
      right: 0,
      top: "middle",
      textStyle: { color: "#8b9bb4" },
      inRange: { color: ["#0b1220", "#1d4e89", "#3dd6c6", "#f0b429"] },
    },
    series: [
      {
        type: "heatmap",
        data,
        emphasis: { itemStyle: { shadowBlur: 8, shadowColor: "rgba(0,0,0,.4)" } },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: 320, width: "100%" }} />;
}
