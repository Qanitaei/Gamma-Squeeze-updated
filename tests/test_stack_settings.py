from gamma_squeeze.stack.boosters import available_boosters, make_classifier
from gamma_squeeze.stack.inventory import PYTHON_REQUIRES, locked_stack
from gamma_squeeze.stack.settings import get_stack_settings


def test_stack_settings_python_requires():
    get_stack_settings.cache_clear()
    s = get_stack_settings()
    assert s.python_requires == PYTHON_REQUIRES
    assert s.phase == 2
    assert "postgresql" in s.database_url or "postgres" in s.database_url
    assert "redis" in s.redis_url


def test_booster_factory_fallback():
    clf = make_classifier("xgboost")
    assert clf is not None
    avail = available_boosters()
    assert avail["sklearn"] is True


def test_locked_stack_helper_exports():
    inv = locked_stack()
    assert inv["phase"] == 2
    assert "FastAPI" in [c["name"] for c in inv["categories"]["backend"]]
