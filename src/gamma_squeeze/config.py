"""Platform configuration: paths, worker URLs, horizons."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

HORIZONS = tuple(range(1, 11))
DEFAULT_LOOKBACK_TRADING_DAYS = 252
DEFAULT_SSD_SUBDIR = "Gamma Squeeze Matrix"

PLATFORM_ROOT = Path(__file__).resolve().parents[2]
FALLBACK_ROOT = PLATFORM_ROOT / "data" / "exports" / DEFAULT_SSD_SUBDIR

ALPACA_BACKUP_WORKER_URL_DEFAULT = (
    "https://alpaca-options-matrix-backup-empty-cherry-67dc.2s6m8rz8fc.workers.dev"
)
ALPACA_BACKUP_NAMESPACE = "alpaca-options-matrix-backup"
ALPACA_BACKUP_NAMESPACE_ID = "e290dbe341d3496aac5e47d17042b3e5"

OHLCV_KV_BASE_URL_DEFAULT = "https://ohcl-nasdaq-nameless-flower.2s6m8rz8fc.workers.dev"
SKEW_IV_WORKER_URL_DEFAULT = (
    "https://skew-3d-implied-volatility-shiny-darkness-9ebb.2s6m8rz8fc.workers.dev"
)
FRED_WORKER_URL_DEFAULT = "https://fred-economic-data-icy-cloud-8332.2s6m8rz8fc.workers.dev"
CALENDAR_WORKER_URL_DEFAULT = "https://icy-poetry-6b0c.2s6m8rz8fc.workers.dev"


def _pick_writable_root(candidates: list[Path], *, mkdir: bool = True) -> Path:
    for path in candidates:
        anchor = path
        while anchor.parent != anchor and not anchor.exists():
            anchor = anchor.parent
        check = anchor if anchor.exists() else path.parent
        if check.exists() and os.access(check, os.W_OK):
            if mkdir:
                path.mkdir(parents=True, exist_ok=True)
            return path
    root = candidates[-1]
    if mkdir:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _data_plane_root() -> Path | None:
    """Optional legacy external data-plane root (no longer required at runtime)."""
    raw = (
        os.getenv("DATA_PLANE_ROOT")
        or os.getenv("ALPACA_DATA_PLANE_ROOT")
        or os.getenv("SCHWAB_OPTIONS_EXPORT_ROOT")  # legacy alias
        or ""
    ).strip()
    return Path(raw).expanduser() if raw else None


def load_env() -> None:
    load_dotenv(PLATFORM_ROOT / ".env")
    sibling = _data_plane_root()
    if sibling is not None:
        sibling_env = sibling / ".env"
        if sibling_env.is_file():
            load_dotenv(sibling_env, override=False)


def ensure_data_plane_on_path() -> Path | None:
    """Legacy no-op helper: GEX/Alpaca clients are vendored in gamma_squeeze."""
    load_env()
    root = _data_plane_root()
    if root is None or not root.is_dir():
        return None
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_matrix_root(explicit: str | None = None) -> Path:
    """Portable SSD root for Gamma Squeeze Matrix annual options history.

    Prefers an existing root that already has ``tickers/`` (Alpaca matrices),
    even when that volume is temporarily non-writable, so feature/infer paths
    keep reading the data plane. Falls back to a writable root for bootstrap.
    """
    load_env()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.getenv("EXPORT_GAMMA_SQUEEZE_MATRIX_SSD_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path(f"/Volumes/PortableSSD/{DEFAULT_SSD_SUBDIR}"))
    candidates.append(FALLBACK_ROOT)

    for path in candidates:
        if (path / "tickers").is_dir():
            return path
    for path in candidates:
        if path.is_dir():
            return path
    return _pick_writable_root(candidates)


def resolve_features_root(explicit: str | None = None) -> Path:
    """Feature/label store under the matrix root ``features/`` when writable."""
    load_env()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.getenv("EXPORT_GAMMA_SQUEEZE_FEATURES_SSD_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    matrix = resolve_matrix_root()
    candidates.append(matrix / "features")
    candidates.append(FALLBACK_ROOT / "features")
    # Prefer matrix/features when the matrix volume itself is writable
    preferred = matrix / "features"
    anchor = matrix if matrix.exists() else preferred.parent
    if anchor.exists() and os.access(anchor, os.W_OK):
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    return _pick_writable_root(candidates)


def resolve_models_root(explicit: str | None = None) -> Path:
    load_env()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.getenv("EXPORT_GAMMA_SQUEEZE_MODELS_SSD_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(resolve_matrix_root() / "models")
    candidates.append(FALLBACK_ROOT / "models")
    return _pick_writable_root(candidates)


def resolve_forecasts_root(explicit: str | None = None) -> Path:
    load_env()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.getenv("EXPORT_GAMMA_SQUEEZE_FORECASTS_SSD_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(resolve_matrix_root() / "forecasts")
    candidates.append(FALLBACK_ROOT / "forecasts")
    return _pick_writable_root(candidates)


def alpaca_backup_worker_url() -> str:
    load_env()
    return os.getenv("ALPACA_BACKUP_WORKER_URL", ALPACA_BACKUP_WORKER_URL_DEFAULT).rstrip("/")


def ohlcv_worker_url() -> str:
    load_env()
    return os.getenv("OHLCV_KV_BASE_URL", OHLCV_KV_BASE_URL_DEFAULT).rstrip("/")


def skew_worker_url() -> str:
    load_env()
    return os.getenv("SKEW_IV_WORKER_URL", SKEW_IV_WORKER_URL_DEFAULT).rstrip("/")


def fred_worker_url() -> str:
    load_env()
    return os.getenv("FRED_WORKER_URL", FRED_WORKER_URL_DEFAULT).rstrip("/")


def calendar_worker_url() -> str:
    load_env()
    return os.getenv("CALENDAR_WORKER_URL", CALENDAR_WORKER_URL_DEFAULT).rstrip("/")


ALPACA_TRADING_API_PAPER_DEFAULT = "https://paper-api.alpaca.markets/v2"
ALPACA_TRADING_API_LIVE_DEFAULT = "https://api.alpaca.markets/v2"
ALPACA_DATA_API_DEFAULT = "https://data.alpaca.markets"


def alpaca_credentials() -> dict[str, str]:
    """Alpaca API credentials + bases for tickers / options / OHLCV."""
    load_env()
    paper = os.getenv("ALPACA_PAPER", "true").strip().lower() in ("1", "true", "yes")
    trading_default = ALPACA_TRADING_API_PAPER_DEFAULT if paper else ALPACA_TRADING_API_LIVE_DEFAULT
    return {
        "key_id": os.getenv("ALPACA_API_KEY_ID", "").strip() or os.getenv("APCA_API_KEY_ID", "").strip(),
        "secret": os.getenv("ALPACA_API_SECRET_KEY", "").strip() or os.getenv("APCA_API_SECRET_KEY", "").strip(),
        "paper": "true" if paper else "false",
        "trading_base": os.getenv("ALPACA_TRADING_API_BASE", trading_default).rstrip("/"),
        "data_base": os.getenv("ALPACA_DATA_API_BASE", ALPACA_DATA_API_DEFAULT).rstrip("/"),
        "option_feed": os.getenv("ALPACA_OPTION_FEED", "indicative").strip() or "indicative",
    }


def alpaca_configured() -> bool:
    creds = alpaca_credentials()
    return bool(creds["key_id"] and creds["secret"])
