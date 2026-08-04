"""Shared pytest fixtures — reproducibility and DI isolation."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_container():
    """Reset the process DI container around each test for isolation."""
    from gamma_squeeze.core.container import reset_container

    reset_container()
    yield
    reset_container()


@pytest.fixture
def seeded():
    """Apply the platform global seed for deterministic unit tests."""
    from gamma_squeeze.core.reproducibility import seed_everything

    return seed_everything()
