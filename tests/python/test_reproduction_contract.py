from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_reproduction_module():
    path = REPOSITORY_ROOT / "tools" / "reproduce.py"
    spec = importlib.util.spec_from_file_location("reproduce", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproduction_has_required_stages() -> None:
    plan = load_reproduction_module().load_reproduction_plan()
    assert plan.stages == [
        "preflight", "build", "unit", "data-validate", "cache-validate",
        "smoke", "metrics", "map-validate",
    ]


def test_reproduction_plan_requires_an_explicit_asset_root() -> None:
    module = load_reproduction_module()
    try:
        module.resolve_asset_root(None, REPOSITORY_ROOT)
    except ValueError as error:
        assert "asset root" in str(error).lower()
    else:
        raise AssertionError("missing asset root must fail closed")
