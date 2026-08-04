"""Deterministic preprocessing and reproducible experiments."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int | None = None) -> int:
    """
    Seed Python, NumPy, and common ML libs for reproducible runs.
    Returns the seed actually applied.
    """
    if seed is None:
        from gamma_squeeze.core.settings import load_platform_settings

        seed = load_platform_settings().global_seed
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001
        pass
    return seed


def stable_hash(payload: Any, *, digest_size: int = 16) -> str:
    """Stable content hash for feature/matrix fingerprints (deterministic pipelines)."""
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    elif isinstance(payload, np.ndarray):
        raw = payload.tobytes() + str(payload.shape).encode() + str(payload.dtype).encode()
    else:
        raw = repr(payload).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=digest_size).hexdigest()


def sorted_feature_matrix(columns: list[str], values: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Ensure column order is lexicographic for deterministic model inputs."""
    order = sorted(range(len(columns)), key=lambda i: columns[i])
    cols = [columns[i] for i in order]
    if values.ndim != 2 or values.shape[1] != len(columns):
        return cols, values
    return cols, values[:, order]
