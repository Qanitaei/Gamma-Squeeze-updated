"""In-process metrics collection (counters, timers, gauges)."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class MetricsCollector:
    namespace: str = "gamma_squeeze"
    enabled: bool = True
    counters: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    timings_ms: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _key(self, name: str) -> str:
        return f"{self.namespace}.{name}" if self.namespace else name

    def incr(self, name: str, value: float = 1.0) -> None:
        if not self.enabled:
            return
        k = self._key(name)
        with self._lock:
            self.counters[k] = self.counters.get(k, 0.0) + value

    def gauge(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        k = self._key(name)
        with self._lock:
            self.gauges[k] = float(value)

    def timing(self, name: str, milliseconds: float) -> None:
        if not self.enabled:
            return
        k = self._key(name)
        with self._lock:
            self.timings_ms.setdefault(k, []).append(float(milliseconds))

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timing(name, (time.perf_counter() - start) * 1000.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            timing_summary = {
                k: {
                    "count": len(v),
                    "mean_ms": (sum(v) / len(v)) if v else 0.0,
                    "p95_ms": sorted(v)[int(0.95 * (len(v) - 1))] if v else 0.0,
                }
                for k, v in self.timings_ms.items()
            }
            return {
                "namespace": self.namespace,
                "enabled": self.enabled,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timings": timing_summary,
            }

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.timings_ms.clear()


_METRICS: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    global _METRICS
    if _METRICS is None:
        from gamma_squeeze.core.settings import load_platform_settings

        s = load_platform_settings()
        _METRICS = MetricsCollector(namespace=s.metrics_namespace, enabled=s.metrics_enabled)
    return _METRICS
