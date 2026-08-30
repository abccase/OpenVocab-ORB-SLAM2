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


def test_reproduction_uses_declared_asset_environment_when_available(tmp_path: Path) -> None:
    module = load_reproduction_module()
    python = tmp_path / "venv/semantic-gpu/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    assert module.resolve_python(tmp_path) == python


def test_reproduction_uses_the_project_fresh_build_entrypoint() -> None:
    module = load_reproduction_module()

    assert module.fresh_build_command(REPOSITORY_ROOT, REPOSITORY_ROOT / "unused") == [
        "bash", str(REPOSITORY_ROOT / "build.sh")
    ]


def test_clean_reproduction_attaches_external_assets_by_symlink(tmp_path: Path) -> None:
    module = load_reproduction_module()
    checkout, assets = tmp_path / "checkout", tmp_path / "assets"
    checkout.mkdir()
    (assets / "external").mkdir(parents=True)

    module.attach_asset_links(checkout, assets)

    assert (checkout / "external").is_symlink()
    assert (checkout / "external").resolve() == assets / "external"


def test_primary_checkout_keeps_its_existing_external_assets(tmp_path: Path) -> None:
    module = load_reproduction_module()
    (tmp_path / "external").mkdir()

    module.attach_asset_links(tmp_path, tmp_path)

    assert (tmp_path / "external").is_dir()
    assert not (tmp_path / "external").is_symlink()
