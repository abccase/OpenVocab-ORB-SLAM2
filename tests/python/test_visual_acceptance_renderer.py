from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_renderer_module():
    path = ROOT / "tools" / "render_visual_acceptance.py"
    spec = importlib.util.spec_from_file_location("render_visual_acceptance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_online_plot_is_bounded_and_consolidates_degradation_intervals() -> None:
    module = load_renderer_module()
    report = {
        "run_id": "online-test",
        "request_rate_cap_hz": 5.0,
        "request_attempt_rate_hz": 2.5,
        "accepted_semantic_rate_hz": 0.0,
        "mean_service_inference_ms": 975.0,
        "p95_service_inference_ms": 1144.0,
        "request_drop_fraction": 0.75,
        "result_unusable_fraction": 1.0,
        "degraded_frame_fraction": 0.6,
    }
    rows = [
        {"frame_index": str(index), "timestamp": str(10.0 + index), "semantic_state": state}
        for index, state in enumerate([
            "ONLINE_VALID", "DEGRADED_TO_BASELINE", "DEGRADED_TO_BASELINE",
            "ONLINE_VALID", "DEGRADED_TO_BASELINE",
        ])
    ]

    root = ET.fromstring(module.online_plot_svg(report, rows))
    width, height = (float(value) for value in root.attrib["viewBox"].split()[2:])
    rectangles = root.findall(".//{http://www.w3.org/2000/svg}rect")
    for rectangle in rectangles:
        if "%" in rectangle.attrib.get("width", ""):
            continue
        x = float(rectangle.attrib.get("x", "0"))
        y = float(rectangle.attrib.get("y", "0"))
        rect_width = float(rectangle.attrib["width"])
        rect_height = float(rectangle.attrib["height"])
        assert 0 <= x <= width and 0 <= y <= height
        assert x + rect_width <= width and y + rect_height <= height
    degraded = [item for item in rectangles if item.attrib.get("data-state") == "DEGRADED_TO_BASELINE"]
    assert [(item.attrib["data-start-frame"], item.attrib["data-end-frame"]) for item in degraded] == [("1", "2"), ("4", "4")]
    text = " ".join(node.text or "" for node in root.iter("{http://www.w3.org/2000/svg}text"))
    assert "rate (Hz)" in text and "latency (ms)" in text and "degradation intervals" in text


def test_manifest_artifact_binding_rejects_corruption(tmp_path: Path) -> None:
    module = load_renderer_module()
    artifact = tmp_path / "trajectory.txt"
    artifact.write_text("approved\n", encoding="utf-8")
    descriptor = {"path": artifact.name, "sha256": module.sha256(artifact), "size_bytes": artifact.stat().st_size}

    assert module.require_manifest_artifact(tmp_path, descriptor, "trajectory") == artifact
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory.*hash"):
        module.require_manifest_artifact(tmp_path, descriptor, "trajectory")


def test_manifest_output_key_supplies_relative_path(tmp_path: Path) -> None:
    module = load_renderer_module()
    artifact = tmp_path / "screenshots/front.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"png")
    outputs = {"screenshots/front.png": {"sha256": module.sha256(artifact), "size_bytes": 3}}

    assert module.require_output_artifact(tmp_path, outputs, "screenshots/front.png", "front") == artifact


def test_delivery_reproduction_identity_requires_matching_commit(tmp_path: Path) -> None:
    module = load_renderer_module()
    reproduction = tmp_path / "reports/final/reproduction-old/reproduction_manifest.json"
    reproduction.parent.mkdir(parents=True)
    reproduction.write_text(json.dumps({"valid": True, "repository_commit": "old", "stages": []}), encoding="utf-8")

    pending = module.find_reproduction_identity(tmp_path, "current")
    assert pending == {"status": "pending_pre_h02_clean_reproduction", "repository_commit": "current"}

    reproduction.write_text(json.dumps({"valid": True, "repository_commit": "current", "stages": [
        {"name": "build", "ok": True, "log_sha256": "a" * 64},
        {"name": "unit", "ok": True, "log_sha256": "b" * 64},
    ]}), encoding="utf-8")
    assert module.find_reproduction_identity(tmp_path, "current") == {
        "status": "pending_pre_h02_clean_reproduction", "repository_commit": "current"
    }

    logs = reproduction.parent / "logs"
    logs.mkdir()
    build_log, unit_log = logs / "build.log", logs / "unit.log"
    build_log.write_text("build passed\n", encoding="utf-8")
    unit_log.write_text("unit passed\n", encoding="utf-8")
    stages = ["preflight", "build", "unit", "data-validate", "cache-validate", "smoke", "metrics", "map-validate"]
    stage_rows = [{"name": name, "ok": True} for name in stages]
    stage_rows[1].update({"log": str(build_log), "log_sha256": module.sha256(build_log)})
    stage_rows[2].update({"log": str(unit_log), "log_sha256": module.sha256(unit_log)})
    reproduction.write_text(json.dumps({
        "valid": True, "repository_commit": "current", "contract_stages": stages,
        "contract_observed": stages, "stages": stage_rows,
    }), encoding="utf-8")
    identity = module.find_reproduction_identity(tmp_path, "current")
    assert identity["status"] == "valid"
    assert identity["manifest_sha256"] == module.sha256(reproduction)
    assert identity["build_log_sha256"] == module.sha256(build_log)
    assert identity["unit_log_sha256"] == module.sha256(unit_log)
