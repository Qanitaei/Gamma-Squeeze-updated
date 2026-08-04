"use client";

import { useEffect, useRef } from "react";
import { createChart, type IChartApi } from "lightweight-charts";

/** TradingView Lightweight Charts placeholder — feed real OHLCV from API later. */
export function PriceChart({
  bars,
}: {
  bars: { time: string; open: number; high: number; low: number; close: number }[];
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = createChart(ref.current, {
      width: ref.current.clientWidth,
      height: 280,
      layout: { background: { color: "#0f1419" }, textColor: "#d7dee8" },
      grid: { vertLines: { color: "#1c2530" }, horzLines: { color: "#1c2530" } },
    });
    const series = chart.addCandlestickSeries({
      upColor: "#5eead4",
      downColor: "#f87171",
      borderVisible: false,
      wickUpColor: "#5eead4",
      wickDownColor: "#f87171",
    });
    if (bars.length) series.setData(bars as any);
    chartRef.current = chart;
    const onResize = () => {
      if (ref.current) chart.applyOptions({ width: ref.current.clientWidth });
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
    };
  }, [bars]);

  return <div ref={ref} style={{ width: "100%" }} />;
}
