from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


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


def test_payload_index_rejects_missing_and_corrupt_payloads(tmp_path: Path) -> None:
    module = load_reproduction_module()
    index = tmp_path / "index.jsonl"
    index.write_text('{"path":"score_maps/000000.npy","sha256":"' + "0" * 64 + '"}\n', encoding="utf-8")

    try:
        module.validate_index_payloads(tmp_path, index, "score map")
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing indexed payload must fail closed")


def test_dynamic_completion_rejects_tampered_tracks_hash(tmp_path: Path) -> None:
    module = load_reproduction_module()
    for name in ("cache_manifest.json", "cache_index.jsonl", "dynamic_tracks.jsonl", "semantic_identity.jsonl", "diagnostics_index.jsonl"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    completion = {key: "0" * 64 for key in ("manifest_sha256", "index_sha256", "tracks_sha256", "semantic_identity_sha256", "diagnostics_index_sha256")}
    completion["frame_count"] = 1
    try:
        module.validate_dynamic_completion(tmp_path, completion, 1)
    except ValueError as error:
        assert "completion" in str(error)
    else:
        raise AssertionError("tampered dynamic completion must fail closed")


def test_primary_checkout_keeps_its_existing_external_assets(tmp_path: Path) -> None:
    module = load_reproduction_module()
    (tmp_path / "external").mkdir()

    module.attach_asset_links(tmp_path, tmp_path)

    assert (tmp_path / "external").is_dir()
    assert not (tmp_path / "external").is_symlink()


def _write_final_delivery_fixture(module, root: Path, commit: str, *, executed: bool) -> None:
    final = root / "reports/final"
    reproduction_root = final / "reproduction-approved"
    logs = reproduction_root / "logs"
    logs.mkdir(parents=True)
    required_stages = module.load_reproduction_plan().stages
    stage_rows = [{"name": name, "ok": True} for name in required_stages]
    if executed:
        build_log, unit_log = logs / "build.log", logs / "unit.log"
        build_log.write_text("build passed\n", encoding="utf-8")
        unit_log.write_text("260 passed\n", encoding="utf-8")
        stage_rows[1].update({"log": str(build_log), "log_sha256": module.sha256_file(build_log)})
        stage_rows[2].update({"log": str(unit_log), "log_sha256": module.sha256_file(unit_log)})
        smoke_facts = {}
        for mode in ("baseline", "semantic-feedback"):
            manifest = reproduction_root / "smoke" / mode / "run_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"valid": True, "mode": mode, "sequence_id": "fr3_walking_xyz"}), encoding="utf-8")
            smoke_log = logs / f"smoke-{mode}.log"
            smoke_log.write_text(f"{mode} passed\n", encoding="utf-8")
            smoke_facts[mode] = {
                "manifest": str(manifest),
                "manifest_sha256": module.sha256_file(manifest),
                "stage": {"ok": True, "log": str(smoke_log), "log_sha256": module.sha256_file(smoke_log)},
            }
        stage_rows[5]["facts"] = smoke_facts
    else:
        stage_rows[1]["status"] = "not rerun by --validate-existing"
        stage_rows[2]["status"] = "not rerun by --validate-existing"
        stage_rows[5]["facts"] = {"status": "not requested"}
    reproduction = reproduction_root / "reproduction_manifest.json"
    reproduction.write_text(json.dumps({
        "valid": True,
        "repository_commit": commit,
        "contract_stages": required_stages,
        "contract_observed": required_stages,
        "stages": stage_rows,
    }), encoding="utf-8")

    report = final / "FINAL_REPORT.md"
    sheet = final / "visual_acceptance_sheet.html"
    source = root / "runs/source.json"
    source.parent.mkdir(parents=True)
    report.write_text("final report\n", encoding="utf-8")
    sheet.write_text("<html>accepted</html>\n", encoding="utf-8")
    source.write_text("source evidence\n", encoding="utf-8")
    panels = [
        "paired-formal-trajectories", "fixed-diagnostics-25-50-75",
        "static-tsdf-and-boxes", "queries-chair-monitor-person",
        "success-and-limitation", "p06-online-timing",
    ]
    sources = final / "visual_acceptance_sources.json"
    sources.write_text(json.dumps({
        "self_contained": True,
        "panels": panels,
        "sheet": {"path": str(sheet), "sha256": module.sha256_file(sheet)},
        "sources": [{"path": str(source), "sha256": module.sha256_file(source)}],
    }), encoding="utf-8")
    delivery = {
        "final_commit": commit,
        "test_build_identity": {
            "status": "valid",
            "repository_commit": commit,
            "manifest_path": str(reproduction),
            "manifest_sha256": module.sha256_file(reproduction),
            "build_log_sha256": module.sha256_file(logs / "build.log") if executed else "0" * 64,
            "unit_log_sha256": module.sha256_file(logs / "unit.log") if executed else "0" * 64,
        },
        "report": {
            "final_report_sha256": module.sha256_file(report),
            "visual_acceptance_sheet_sha256": module.sha256_file(sheet),
            "source_manifest_sha256": module.sha256_file(sources),
            "p08_artifacts": {},
        },
    }
    (final / "delivery_manifest.json").write_text(json.dumps(delivery), encoding="utf-8")


def _patch_expensive_reproduction_checks(monkeypatch, module, commit: str) -> None:
    monkeypatch.setattr(module, "attach_asset_links", lambda *_: None)
    monkeypatch.setattr(module, "_validate_data", lambda *_: {})
    monkeypatch.setattr(module, "_validate_caches", lambda *_: {})
    monkeypatch.setattr(module, "_validate_metrics", lambda *_: {})
    monkeypatch.setattr(module, "_validate_formal_runs", lambda *_: {})
    monkeypatch.setattr(module, "_validate_maps", lambda *_: {})
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_args, **_kwargs: commit)


def test_validate_existing_rejects_unexecuted_build_unit_and_smoke(tmp_path: Path, monkeypatch) -> None:
    module = load_reproduction_module()
    commit = "a" * 40
    _write_final_delivery_fixture(module, tmp_path, commit, executed=False)
    _patch_expensive_reproduction_checks(monkeypatch, module, commit)

    with pytest.raises(ValueError, match="executed reproduction evidence"):
        module.run_reproduction(
            tmp_path, tmp_path, tmp_path / "output", validate_existing=True,
            smoke=False, build_dir=tmp_path / "build",
        )


def test_validate_existing_rejects_tampered_visual_sheet(tmp_path: Path, monkeypatch) -> None:
    module = load_reproduction_module()
    commit = "b" * 40
    _write_final_delivery_fixture(module, tmp_path, commit, executed=True)
    _patch_expensive_reproduction_checks(monkeypatch, module, commit)
    (tmp_path / "reports/final/visual_acceptance_sheet.html").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="visual acceptance sheet hash mismatch"):
        module.run_reproduction(
            tmp_path, tmp_path, tmp_path / "output", validate_existing=True,
            smoke=False, build_dir=tmp_path / "build",
        )


def test_validate_existing_rejects_tampered_visual_source(tmp_path: Path, monkeypatch) -> None:
    module = load_reproduction_module()
    commit = "c" * 40
    _write_final_delivery_fixture(module, tmp_path, commit, executed=True)
    _patch_expensive_reproduction_checks(monkeypatch, module, commit)
    (tmp_path / "runs/source.json").write_text("tampered source\n", encoding="utf-8")

    with pytest.raises(ValueError, match="visual source 0 hash mismatch"):
        module.run_reproduction(
            tmp_path, tmp_path, tmp_path / "output", validate_existing=True,
            smoke=False, build_dir=tmp_path / "build",
        )
