#!/usr/bin/env python3
"""Launch all microservices as local uvicorn processes."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.common.registry import SERVICE_REGISTRY  # noqa: E402

MODULE_BY_NAME = {
    "orchestrator": "services.orchestrator.app:app",
    "data_collection": "services.data_collection.app:app",
    "feature_engineering": "services.feature_engineering.app:app",
    "pattern_recognition": "services.pattern_recognition.app:app",
    "regime_hmm": "services.regime_hmm.app:app",
    "xgboost_direction": "services.xgboost_direction.app:app",
    "tft_forecasting": "services.tft_forecasting.app:app",
    "dealer_hedging": "services.dealer_hedging.app:app",
    "gamma_squeeze_engine": "services.gamma_squeeze_engine.app:app",
    "ppo_agent": "services.ppo_agent.app:app",
    "portfolio_hedging": "services.portfolio_hedging.app:app",
    "dashboard": "services.dashboard.app:app",
    "trade_alerts": "services.trade_alerts.app:app",
    "explainability": "services.explainability.app:app",
    "model_training": "services.model_training.app:app",
    "performance_metrics": "services.performance_metrics.app:app",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default="", help="Comma-separated service names")
    args = p.parse_args()

    names = list(SERVICE_REGISTRY.keys())
    if args.only.strip():
        names = [n.strip() for n in args.only.split(",") if n.strip()]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")]
    )

    procs: list[subprocess.Popen] = []
    try:
        for name in names:
            info = SERVICE_REGISTRY[name]
            module = MODULE_BY_NAME[name]
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                module,
                "--host",
                "127.0.0.1",
                "--port",
                str(info["port"]),
            ]
            print(f"Starting {name} on :{info['port']}")
            procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env))
        print("All requested services launched. Ctrl+C to stop.")
        while True:
            time.sleep(1)
            for proc in procs:
                if proc.poll() is not None:
                    print(f"Process exited with {proc.returncode}")
                    raise SystemExit(proc.returncode or 1)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        for proc in procs:
            proc.send_signal(signal.SIGTERM)
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
