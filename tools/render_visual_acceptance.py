#!/usr/bin/env python3
"""Render the self-contained H02 evidence sheet from approved ignored artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_py.openvocab_slam.cache import read_cache_frame
from semantic_py.openvocab_slam.schemas import decode_binary_mask_rle



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected object")
    return value


def require_manifest_artifact(root: Path, descriptor: object, label: str) -> Path:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{label} descriptor is absent")
    relative = Path(str(descriptor.get("path", "")))
    expected = descriptor.get("sha256")
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{label} path is unsafe")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"{label} is missing: {relative}")
    if not isinstance(expected, str) or sha256(path) != expected:
        raise ValueError(f"{label} hash mismatch: {relative}")
    size = descriptor.get("size_bytes")
    if size is not None and int(size) != path.stat().st_size:
        raise ValueError(f"{label} size mismatch: {relative}")
    return path


def require_output_artifact(root: Path, outputs: object, relative: str, label: str) -> Path:
    if not isinstance(outputs, dict) or not isinstance(outputs.get(relative), dict):
        raise ValueError(f"{label} output descriptor is absent")
    descriptor = {"path": relative, **outputs[relative]}
    return require_manifest_artifact(root, descriptor, label)


def find_reproduction_identity(asset_root: Path, commit: str) -> dict[str, Any]:
    for path in sorted((asset_root / "reports/final").glob("reproduction-*/reproduction_manifest.json"), reverse=True):
        try:
            value = read_json_object(path, "reproduction manifest")
        except ValueError:
            continue
        if value.get("valid") is not True or value.get("repository_commit") != commit:
            continue
        stages = {item.get("name"): item for item in value.get("stages", []) if isinstance(item, dict)}
        build_hash = stages.get("build", {}).get("log_sha256")
        unit_hash = stages.get("unit", {}).get("log_sha256")
        if not all(isinstance(item, str) and len(item) == 64 for item in (build_hash, unit_hash)):
            continue
        return {
            "status": "valid", "repository_commit": commit,
            "manifest_path": str(path), "manifest_sha256": sha256(path),
            "build_log_sha256": build_hash, "unit_log_sha256": unit_hash,
        }
    return {"status": "pending_pre_h02_clean_reproduction", "repository_commit": commit}


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/svg+xml"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def svg(title: str, body: str, width: int = 920, height: int = 420) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
            f'<text x="20" y="30" font-family="sans-serif" font-size="20" font-weight="bold">{html.escape(title)}</text>{body}</svg>')


def online_plot_svg(report: dict[str, Any], telemetry_rows: list[dict[str, Any]]) -> str:
    """Bounded, unit-separated P06 plot with an explicit degradation timeline."""
    width, height = 920, 520
    hz = [float(report["request_rate_cap_hz"]), float(report["request_attempt_rate_hz"]), float(report["accepted_semantic_rate_hz"])]
    ms = [float(report["mean_service_inference_ms"]), float(report["p95_service_inference_ms"])]
    frac = [float(report["request_drop_fraction"]), float(report["result_unusable_fraction"]), float(report["degraded_frame_fraction"])]
    if not all(math.isfinite(value) and value >= 0 for value in hz + ms) or not all(math.isfinite(value) and 0 <= value <= 1 for value in frac):
        raise ValueError("online metrics are outside their declared scales")

    def bars(values: list[float], maximum: float, x: int, title: str, labels: list[str], color: str) -> str:
        elements = [f'<text x="{x}" y="75" font-family="sans-serif">{html.escape(title)}</text>']
        for index, value in enumerate(values):
            bar_height = 100 * value / maximum
            bar_x = x + index * 62
            elements.append(f'<rect x="{bar_x}" y="{190-bar_height:.1f}" width="38" height="{bar_height:.1f}" fill="{color}"/>')
            elements.append(f'<text x="{bar_x}" y="207" font-family="sans-serif" font-size="10">{labels[index]}</text>')
            elements.append(f'<text x="{bar_x}" y="{184-bar_height:.1f}" font-family="sans-serif" font-size="10">{value:.3f}</text>')
        return "".join(elements)

    intervals: list[tuple[int, int, str, float, float]] = []
    for position, row in enumerate(telemetry_rows):
        frame = int(row["frame_index"])
        timestamp = float(row["timestamp"])
        state = str(row["semantic_state"])
        if frame != position or not math.isfinite(timestamp):
            raise ValueError("online telemetry is not contiguous and finite")
        if intervals and intervals[-1][2] == state:
            start, _, old_state, start_time, _ = intervals[-1]
            intervals[-1] = (start, frame, old_state, start_time, timestamp)
        else:
            intervals.append((frame, frame, state, timestamp, timestamp))
    frame_count = len(telemetry_rows)
    timeline_parts = ['<g id="degradation-intervals">']
    for start, end, state, start_time, end_time in intervals:
        x = 60 + 800 * start / max(1, frame_count)
        interval_width = 800 * (end - start + 1) / max(1, frame_count)
        color = "#d62728" if state == "DEGRADED_TO_BASELINE" else "#2ca02c"
        timeline_parts.append(
            f'<rect x="{x:.2f}" y="390" width="{interval_width:.2f}" height="28" fill="{color}" '
            f'data-state="{html.escape(state)}" data-start-frame="{start}" data-end-frame="{end}" '
            f'data-start-time="{start_time:.6f}" data-end-time="{end_time:.6f}"/>'
        )
    timeline_parts.append("</g>")
    body = (
        bars(hz, max(5.0, max(hz)), 60, "rate (Hz): cap / actual / accepted", ["cap", "actual", "accepted"], "#1f77b4")
        + bars(ms, max(1.0, max(ms)), 310, "latency (ms): mean / p95", ["mean", "p95"], "#9467bd")
        + bars(frac, 1.0, 560, "fractions (0-1): request-drop / unusable / degraded", ["drop", "unusable", "degraded"], "#ff7f0e")
        + '<text x="60" y="370" font-family="sans-serif">degradation intervals by frame/time: green ONLINE_VALID, red DEGRADED_TO_BASELINE</text>'
        + "".join(timeline_parts)
        + f'<text x="60" y="445" font-family="sans-serif">frames 0-{max(0, frame_count - 1)}; interval endpoints carry dataset timestamps (s)</text>'
    )
    return svg(f'P06 online metrics {report["run_id"]}', body, width, height)


def trajectory_svg(asset_root: Path) -> tuple[str, list[Path], list[str]]:
    paths: list[Path] = []
    tracks: list[tuple[str, str, Path]] = []
    run_ids: list[str] = []
    for mode, color in (("baseline", "#1f77b4"), ("semantic-feedback", "#d62728")):
        candidates = sorted((asset_root / "runs/ovorb2_tum_v1" / mode / "fr3_walking_xyz/seed-23011").glob("attempt-*/run_manifest.json"))
        if len(candidates) != 1:
            raise ValueError(f"expected one formal {mode} trajectory for fr3_walking_xyz seed 23011")
        manifest_path = candidates[0]
        manifest = read_json_object(manifest_path, f"formal {mode} run manifest")
        if (manifest.get("valid") is not True or manifest.get("state") != "COMPLETED" or
                manifest.get("mode") != mode or manifest.get("sequence_id") != "fr3_walking_xyz" or
                manifest.get("seed") != 23011 or not isinstance(manifest.get("run_id"), str)):
            raise ValueError(f"formal {mode} run identity is invalid")
        trajectory = require_manifest_artifact(manifest_path.parent, manifest.get("trajectory"), f"formal {mode} trajectory")
        tracks.append((mode, color, trajectory))
        paths.extend((manifest_path, trajectory))
        run_ids.append(str(manifest["run_id"]))
    arrays = []
    for _, _, path in tracks:
        rows = np.loadtxt(path, comments="#", usecols=(1, 2, 3))
        arrays.append(np.atleast_2d(rows))
    all_points = np.vstack(arrays)
    lo, hi = all_points[:, :2].min(axis=0), all_points[:, :2].max(axis=0)
    scale = np.maximum(hi - lo, 1e-9)
    polylines = []
    for (mode, color, _), points in zip(tracks, arrays):
        chosen = points[::max(1, len(points) // 600), :2]
        coordinates = " ".join(f"{80 + 760*(p[0]-lo[0])/scale[0]:.1f},{360 - 280*(p[1]-lo[1])/scale[1]:.1f}" for p in chosen)
        polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coordinates}"/><text x="{80 if mode == "baseline" else 350}" y="395" font-family="sans-serif" fill="{color}">{mode} (seed 23011)</text>')
    body = '<line x1="80" y1="360" x2="840" y2="360" stroke="black"/><line x1="80" y1="70" x2="80" y2="360" stroke="black"/>' + "".join(polylines) + '<text x="700" y="390" font-family="sans-serif">x / m</text><text x="25" y="75" font-family="sans-serif">y / m</text>'
    return svg("Formal trajectories — fr3_walking_xyz (SE(3) study, metres)", body), paths, run_ids


def diagnostic_image(asset_root: Path, frame_id: int, fraction: float, destination: Path) -> tuple[list[Path], list[tuple[int, str, str]]]:
    dataset = asset_root / "data/tum/raw/rgbd_dataset_freiburg3_walking_xyz"
    line = [row for row in (dataset / "associate.txt").read_text(encoding="utf-8").splitlines() if row and not row.startswith("#")][frame_id]
    rgb = dataset / line.split()[1]
    dynamic_root = asset_root / "cache/dynamic/v1/fr3_walking_xyz"
    dynamic_index = [json.loads(row) for row in (dynamic_root / "cache_index.jsonl").read_text(encoding="utf-8").splitlines() if row.strip()]
    dynamic_entry = next((item for item in dynamic_index if int(item["frame_id"]) == frame_id), None)
    if dynamic_entry is None:
        raise ValueError(f"dynamic score absent for diagnostic {frame_id}")
    scores = require_manifest_artifact(dynamic_root, dynamic_entry, "diagnostic dynamic score")
    raw = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
    score = np.load(scores)
    if raw is None or raw.shape[:2] != score.shape:
        raise ValueError(f"invalid fixed diagnostic source {frame_id}")
    score_u8 = np.clip(score * 255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(score_u8, cv2.COLORMAP_TURBO)
    semantic_index = [json.loads(row) for row in (asset_root / "cache/semantic/v1/fr3_walking_xyz/cache_index.jsonl").read_text(encoding="utf-8").splitlines() if row.strip()]
    entry = next((item for item in semantic_index if int(item["frame_id"]) == frame_id), None)
    if entry is None:
        raise ValueError(f"semantic packet absent for diagnostic {frame_id}")
    semantic_root = asset_root / "cache/semantic/v1/fr3_walking_xyz"
    packet_path = require_manifest_artifact(semantic_root, entry, "diagnostic semantic packet")
    packet = read_cache_frame(packet_path)
    if packet.frame_id != frame_id or packet.sequence_id != "fr3_walking_xyz" or packet.source_image_sha256 != sha256(rgb):
        raise ValueError(f"semantic packet identity mismatch for diagnostic {frame_id}")
    masked = raw.copy()
    colors_rgb = ((46, 204, 113), (231, 76, 60), (52, 152, 219), (241, 196, 15),
                  (155, 89, 182), (26, 188, 156), (230, 126, 34), (236, 240, 241))
    legend: list[tuple[int, str, str]] = []
    for index, instance in enumerate(packet.instances):
        instance_mask = decode_binary_mask_rle(instance.mask_rle)
        if instance_mask.shape != score.shape:
            raise ValueError("semantic mask dimensions differ from score map")
        rgb_color = colors_rgb[index % len(colors_rgb)]
        color = np.asarray(rgb_color[::-1], dtype=np.uint8)
        masked[instance_mask] = (0.35 * masked[instance_mask] + 0.65 * color).astype(np.uint8)
        legend.append((int(instance.local_id), str(instance.label), "#" + "".join(f"{channel:02x}" for channel in rgb_color)))
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints = orb.detect(raw, None)
    retained = raw.copy()
    for point in keypoints:
        x, y = int(round(point.pt[0])), int(round(point.pt[1]))
        if 0 <= x < score.shape[1] and 0 <= y < score.shape[0]:
            color = (0, 255, 255) if score[y, x] < 0.70 else (0, 0, 255)
            cv2.circle(retained, (x, y), 2, color, -1)
    labels = ["RGB", "semantic instance masks", "dynamic score 0-1", "ORB yellow kept red filtered"]
    panes = [raw, masked, heat, retained]
    label_h = 34
    canvas = np.zeros((2 * (raw.shape[0] + label_h), 2 * raw.shape[1], 3), dtype=np.uint8)
    for index, pane in enumerate(panes):
        row, col = divmod(index, 2); y = row * (raw.shape[0] + label_h); x = col * raw.shape[1]
        canvas[y:y + raw.shape[0], x:x + raw.shape[1]] = pane
        cv2.putText(canvas, labels[index], (x + 8, y + raw.shape[0] + 23), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"fr3_walking_xyz frame {frame_id} ({fraction:.0%}, frozen causal cache)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(destination), canvas)
    return [rgb, scores, packet_path, destination], legend


def validated_map(asset_root: Path, sequence: str, directory_name: str) -> dict[str, Any]:
    root = asset_root / "artifacts/maps" / directory_name
    integrity_path = asset_root / "artifacts/maps" / f"{sequence}-integrity.json"
    integrity = read_json_object(integrity_path, f"{sequence} map integrity")
    if integrity.get("valid") is not True or Path(str(integrity.get("map_root", ""))).resolve() != root.resolve():
        raise ValueError(f"{sequence} map integrity redirects or is invalid")
    manifest_path = root / "map_manifest.json"
    if sha256(manifest_path) != integrity.get("map_manifest_sha256"):
        raise ValueError(f"{sequence} map manifest hash mismatch")
    manifest = read_json_object(manifest_path, f"{sequence} map manifest")
    if (manifest.get("valid") is not True or manifest.get("schema") != "ovorb.map.v1" or
            manifest.get("sequence_id") != sequence or manifest.get("study_id") != "p07-static-map-v1"):
        raise ValueError(f"{sequence} map manifest identity is invalid")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{sequence} map outputs are absent")
    front = require_output_artifact(root, outputs, "screenshots/front.png", f"{sequence} front screenshot")
    objects_path = require_output_artifact(root, outputs, "objects.json", f"{sequence} objects")
    objects = json.loads(objects_path.read_text(encoding="utf-8"))
    if not isinstance(objects, list):
        raise ValueError(f"{sequence} objects are malformed")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or counts.get("static_objects") != len(objects):
        raise ValueError(f"{sequence} map counts do not match objects")
    return {"root": root, "integrity_path": integrity_path, "integrity": integrity, "manifest_path": manifest_path,
            "manifest": manifest, "front": front, "objects_path": objects_path, "objects": objects}


def validated_online_evidence(asset_root: Path) -> dict[str, Any]:
    root = asset_root / "runs/p06-online/fr3_walking_xyz/seed-23011/demo/attempt-006"
    manifest_path = root / "run_manifest.json"
    manifest = read_json_object(manifest_path, "P06 online run manifest")
    if (manifest.get("valid") is not True or manifest.get("state") != "COMPLETED" or
            manifest.get("sequence_id") != "fr3_walking_xyz" or manifest.get("mode") != "online-semantic-feedback" or
            not isinstance(manifest.get("run_id"), str)):
        raise ValueError("P06 online run identity is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("P06 online artifact identities are absent")
    report_path = require_manifest_artifact(root, artifacts.get("online_report"), "P06 online report")
    telemetry_path = require_manifest_artifact(root, artifacts.get("telemetry"), "P06 online telemetry")
    report = read_json_object(report_path, "P06 online report")
    if report.get("run_id") != manifest.get("run_id") or report.get("sequence_id") != manifest.get("sequence_id"):
        raise ValueError("P06 online report identity differs from its run manifest")
    with telemetry_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(report.get("frame_count", -1)):
        raise ValueError("P06 online telemetry frame count mismatch")
    return {"root": root, "manifest_path": manifest_path, "manifest": manifest,
            "report_path": report_path, "report": report, "telemetry_path": telemetry_path, "rows": rows}


def query_rows(objects: list[dict[str, Any]], query: str) -> str:
    matched = [item for item in objects if query in str(item.get("normalized_label", "")).split() or query in item.get("aliases", [])]
    matched.sort(key=lambda item: (-float(item.get("confidence", 0.0)), str(item.get("object_id", ""))))
    if not matched:
        return "no exact/token-containment result (honest empty evidence)"
    return "; ".join(f"{item['object_id']} {item['normalized_label']} confidence={float(item['confidence']):.3f}" for item in matched[:4])


def object_boxes_svg(sequence: str, objects: list[dict[str, Any]]) -> str:
    if not objects:
        return svg(f"{sequence} object boxes — T_world_camera", '<text x="80" y="210" font-family="sans-serif" font-size="18">0 static object boxes (measured limitation)</text>')
    polygons: list[tuple[dict[str, Any], np.ndarray]] = []
    for item in objects:
        center = np.asarray(item["centroid"], dtype=float)[:2]
        extent = np.asarray(item["extent"], dtype=float)[:2]
        orientation = np.asarray(item["orientation"], dtype=float)[:2, :2]
        corners = np.asarray(((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))) * extent
        polygons.append((item, corners @ orientation.T + center))
    all_points = np.vstack([points for _, points in polygons])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    span = np.maximum(high - low, 1e-9)
    elements = ['<line x1="70" y1="360" x2="850" y2="360" stroke="black"/><line x1="70" y1="65" x2="70" y2="360" stroke="black"/>']
    for item, points in polygons:
        coordinates = " ".join(f"{70 + 760*(point[0]-low[0])/span[0]:.1f},{350 - 270*(point[1]-low[1])/span[1]:.1f}" for point in points)
        elements.append(f'<polygon points="{coordinates}" fill="none" stroke="#d62728" stroke-width="1"/>')
        center = points.mean(axis=0)
        x = 70 + 760 * (center[0] - low[0]) / span[0]
        y = 350 - 270 * (center[1] - low[1]) / span[1]
        elements.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="8">{html.escape(str(item["object_id"]))}</text>')
    elements.append('<text x="720" y="395" font-family="sans-serif">x / m</text><text x="20" y="75" font-family="sans-serif">y / m</text>')
    return svg(f"{sequence} oriented object-box footprints — T_world_camera (m)", "".join(elements))


def collect_delivery_evidence(asset_root: Path, commit: str, maps: list[dict[str, Any]]) -> dict[str, Any]:
    run_root = asset_root / "runs/ovorb2_tum_v1"
    registration_path = run_root / "study_registration.json"
    registration = read_json_object(registration_path, "P08 registration")
    datasets = registration.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"fr1_desk", "fr1_room", "fr3_sitting_halfsphere", "fr3_sitting_xyz", "fr3_walking_halfsphere", "fr3_walking_xyz"}:
        raise ValueError("P08 registration does not contain the six-sequence identity")
    data: dict[str, Any] = {}
    caches: dict[str, Any] = {}
    for sequence, registered in sorted(datasets.items()):
        if not isinstance(registered, dict):
            raise ValueError(f"malformed registered dataset: {sequence}")
        data[sequence] = {
            "dataset_manifest_sha256": registered["dataset_manifest_sha256"],
            "source_tree_sha256": registered["source_tree_sha256"],
            "association_sha256": registered["association_sha256"],
            "groundtruth_sha256": registered["groundtruth_sha256"],
        }
        caches[sequence] = {
            "semantic_manifest_sha256": registered["semantic_manifest_sha256"],
            "dynamic_manifest_sha256": registered["cache_identity"]["manifest_sha256"],
            "dynamic_index_sha256": registered["cache_identity"]["index_sha256"],
            "dynamic_completion_sha256": registered["cache_identity"]["completion_sha256"],
        }
    run_manifests = sorted(run_root.glob("baseline/*/seed-*/attempt-*/run_manifest.json")) + sorted(run_root.glob("semantic-feedback/*/seed-*/attempt-*/run_manifest.json"))
    if len(run_manifests) != 60:
        raise ValueError("delivery identity requires exactly 60 formal run manifests")
    run_hashes: dict[str, str] = {}
    for path in run_manifests:
        manifest = read_json_object(path, "formal delivery run")
        if manifest.get("valid") is not True or manifest.get("state") != "COMPLETED" or not isinstance(manifest.get("run_id"), str):
            raise ValueError(f"invalid formal delivery run: {path}")
        run_hashes[str(manifest["run_id"])] = sha256(path)
    map_hashes = {
        item["manifest"]["sequence_id"]: {
            "integrity_sha256": sha256(item["integrity_path"]),
            "map_manifest_sha256": sha256(item["manifest_path"]),
        }
        for item in maps
    }
    return {
        "data": data, "caches": caches,
        "formal_runs": {
            "registration_sha256": sha256(registration_path),
            "matrix_sha256": sha256(run_root / "run_matrix.csv"),
            "registry_sha256": sha256(run_root / "run_registry.jsonl"),
            "valid_run_count": len(run_hashes), "run_manifest_sha256": run_hashes,
        },
        "maps": map_hashes,
        "reproduction": find_reproduction_identity(asset_root, commit),
    }


def render(asset_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "visual_assets"; assets.mkdir(exist_ok=True)
    sources: list[Path] = []
    trajectory, trajectory_paths, trajectory_run_ids = trajectory_svg(asset_root); sources.extend(trajectory_paths)
    diagnostics: list[tuple[float, Path, list[tuple[int, str, str]]]] = []
    for fraction, frame in ((.25, 207), (.50, 413), (.75, 617)):
        destination = assets / f"diagnostic-{int(fraction * 100)}.png"
        diagnostic_sources, legend = diagnostic_image(asset_root, frame, fraction, destination)
        sources.extend(diagnostic_sources)
        diagnostics.append((fraction, destination, legend))
    desk = validated_map(asset_root, "fr1_desk", "smoke-semantic-feedback-fr1_desk-seed-23011-attempt-002")
    walking = validated_map(asset_root, "fr3_walking_xyz", "smoke-semantic-feedback-fr3_walking_xyz-seed-23011-attempt-001")
    desk_objects, walking_objects = desk["objects"], walking["objects"]
    sources += [desk["integrity_path"], desk["manifest_path"], desk["front"], desk["objects_path"],
                walking["integrity_path"], walking["manifest_path"], walking["front"], walking["objects_path"]]
    online_evidence = validated_online_evidence(asset_root)
    online, online_rows = online_evidence["report"], online_evidence["rows"]
    sources += [online_evidence["manifest_path"], online_evidence["report_path"], online_evidence["telemetry_path"]]
    queries = "<br/>".join(f"<b>{query}</b>: {html.escape(query_rows(desk_objects, query))}" for query in ("chair", "monitor", "person"))
    diagnostic_html = "".join(
        f'<figure><img src="{data_uri(path)}" alt="{fraction:.0%} diagnostic"><figcaption>{fraction:.0%}: fr3_walking_xyz frame {int(fraction == .25 and 207 or fraction == .5 and 413 or 617)}; RGB, instance masks, score, retained-feature diagnostics. Units: pixels / score 0–1. Instance legend: '
        + "; ".join(f'<span style="display:inline-block;width:10px;height:10px;background:{color}"></span> local_id={local_id} {html.escape(label)}' for local_id, label, color in legend)
        + '</figcaption></figure>' for fraction, path, legend in diagnostics)
    boxes = "; ".join(f"{item['object_id']} {item['normalized_label']} box extent(m)={','.join(f'{x:.2f}' for x in item['extent'])}" for item in desk_objects[:8])
    desk_counts, walking_counts = desk["manifest"]["counts"], walking["manifest"]["counts"]
    body = f'''<!doctype html><html><head><meta charset="utf-8"><title>H02 visual acceptance sheet</title><style>body{{font-family:Arial,sans-serif;margin:20px;color:#111}}section{{border:1px solid #777;padding:14px;margin:16px 0;break-inside:avoid}}img{{max-width:100%;height:auto;border:1px solid #aaa}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}}figure{{margin:0}}figcaption{{font-size:13px}}.note{{background:#fff7df;padding:8px}}@media print{{body{{min-width:1100px}}}}</style></head><body>
<h1>H02 — final semantic-map visual acceptance</h1><p>Source mode: approved formal frozen-cache study for localization; P07 maps; P06 online demonstration. Online IPC is not formal localization evidence. Sheet source commit is external-artifact-bound; every image is embedded as a data URI.</p>
<section><h2>1. Paired formal trajectories</h2><img src="{svg_uri(trajectory)}" alt="formal paired trajectories"><p>Sequence <b>fr3_walking_xyz</b>; run IDs {html.escape(' and '.join(trajectory_run_ids))}. Coordinates are x/y metres in the formal SE(3) protocol; blue baseline, red semantic-feedback. Both trajectory bytes are bound by their completed run manifests.</p></section>
<section><h2>2. Fixed 25/50/75% diagnostics</h2><div class="grid">{diagnostic_html}</div><p class="note">Instance-mask colors identify frozen SemanticFramePacket local instances; the packet-derived legend is shown in each pane. Turbo heat map separately encodes dynamic score 0–1; yellow ORB points are retained and red points filtered.</p></section>
<section><h2>3. Static TSDF and object boxes</h2><div class="grid"><figure><img src="{data_uri(desk['front'])}"><figcaption>Control/low-dynamic sequence fr1_desk: {desk['integrity']['cloud_points']} static-cloud points, {desk_counts['static_objects']} static objects, {desk_counts['dynamic_tracks']} dynamic tracks. Manifest-bound front TSDF view.</figcaption></figure><figure><img src="{svg_uri(object_boxes_svg('fr1_desk', desk_objects))}"><figcaption>fr1_desk oriented static object boxes, projected into the T_world_camera x/y plane.</figcaption></figure><figure><img src="{data_uri(walking['front'])}"><figcaption>Dynamic sequence fr3_walking_xyz: {walking['integrity']['cloud_points']} static-cloud points, {walking_counts.get('static_objects', 0)} static objects and {walking_counts.get('dynamic_tracks', 0)} dynamic tracks. Manifest-bound front TSDF view.</figcaption></figure><figure><img src="{svg_uri(object_boxes_svg('fr3_walking_xyz', walking_objects))}"><figcaption>Dynamic-sequence object-box result: zero boxes, retained as an explicit measured limitation.</figcaption></figure></div><p>Control object-box records (T_world_camera map frame; extent metres): {html.escape(boxes)}</p></section>
<section><h2>4. Text query results (fr1_desk map)</h2><p>{queries}</p><p class="note">The person query is preserved as empty when no stable static object is present; moving people are represented in dynamic tracks, not permanently fused into the static object map.</p></section>
<section><h2>5. Representative success and limitation</h2><p><b>Success:</b> fr1_desk static TSDF/object export is valid with queryable static objects. <b>Limitation:</b> fr3_walking_xyz has no static objects/dynamic tracks in this representative export; motion confirmation and conservative masking can yield limited semantic-map evidence. The P08 paired study result is <b>neutral</b>, not a positive improvement claim.</p></section>
<section><h2>6. P06 online timing / rate / latency / drops</h2><img src="{svg_uri(online_plot_svg(online, online_rows))}" alt="online timing"><p>Run ID {online['run_id']}: request cap 5 Hz, actual request rate {online['request_attempt_rate_hz']:.3f} Hz, accepted semantic rate {online['accepted_semantic_rate_hz']:.3f} Hz, mean inference {online['mean_service_inference_ms']:.1f} ms, p95 {online['p95_service_inference_ms']:.1f} ms, maximum IPC call {online['maximum_ipc_call_ms']:.3f} ms, request-drop fraction {online['request_drop_fraction']:.3f}, unusable-result fraction {online['result_unusable_fraction']:.3f}, degraded frame fraction {online['degraded_frame_fraction']:.3f}. This is a hardware-limited degraded-only demonstration.</p></section>
</body></html>'''
    sheet = output / "visual_acceptance_sheet.html"; temporary = sheet.with_name(".visual_acceptance_sheet.html.partial"); temporary.write_text(body, encoding="utf-8"); temporary.replace(sheet)
    required_labels = ("Paired formal trajectories", "Fixed 25/50/75% diagnostics", "Static TSDF and object boxes", "Text query results", "Representative success and limitation", "P06 online timing")
    if (any(label not in body for label in required_labels) or body.count("data:image/") < 7 or
            'src="visual_assets/' in body or "width=\"0\"" in body):
        raise ValueError("visual sheet completeness or clipping-risk validation failed")
    manifest = {"schema_version": 1, "sheet": {"path": str(sheet), "sha256": sha256(sheet)}, "sources": [{"path": str(path), "sha256": sha256(path)} for path in sources], "panels": ["paired-formal-trajectories", "fixed-diagnostics-25-50-75", "static-tsdf-and-boxes", "queries-chair-monitor-person", "success-and-limitation", "p06-online-timing"], "self_contained": True, "html_validation": {"required_labels": list(required_labels), "embedded_image_count": body.count("data:image/"), "minimum_render_width_px": 1100}}
    atomic_json(output / "visual_acceptance_sources.json", manifest)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    summary_path = asset_root / "reports/final/summary.json"
    summary = read_json_object(summary_path, "P08 summary")
    valid_run_count = int(summary.get("valid_run_count", -1))
    if valid_run_count != 60 or int(summary.get("paired_statistics", {}).get("overall", {}).get("pair_count", -1)) != 30:
        raise ValueError("P08 summary does not bind the approved 60-run/30-pair result")
    evidence = collect_delivery_evidence(asset_root, commit, [desk, walking])
    if evidence["formal_runs"]["valid_run_count"] != valid_run_count:
        raise ValueError("delivery formal-run count differs from P08 summary")
    report_artifacts = summary.get("artifacts")
    if not isinstance(report_artifacts, dict):
        raise ValueError("P08 summary report artifacts are absent")
    delivery = {
        "schema_version": 1,
        "final_commit": commit,
        "source_pins": {
            "orb_slam2": "f2e6f51cdc8d067655d90a78c06261378e07e8f3",
            "grounding_dino": "856dde20aee659246248e20734ef9ba5214f5e44",
            "segment_anything": "dca509fe793f601edb92606367a655c15ac00fdf",
        },
        "licenses": {"orb_slam2": "GPL-3.0-or-later", "grounding_dino": "Apache-2.0", "segment_anything": "Apache-2.0", "tum_rgbd": "provider terms; not bundled", "python_lock": {"path": "requirements/semantic.lock", "sha256": sha256(ROOT / "requirements/semantic.lock")}},
        "study": {"valid_runs": valid_run_count, "paired_runs": 30, "outcome": summary.get("outcome_classification"), "summary_sha256": sha256(summary_path)},
        "evidence": evidence,
        "test_build_identity": evidence["reproduction"],
        "report": {"visual_acceptance_sheet_sha256": sha256(sheet), "source_manifest_sha256": sha256(output / "visual_acceptance_sources.json"), "p08_artifacts": report_artifacts},
        "asset_root": str(asset_root),
        "ignored_uncommitted_assets": ["data/", "weights/", "cache/", "runs/", "artifacts/", "reports/", "build*/", ".superpowers/", "Agent Pack state"],
        "known_limitations": ["neutral P08 result", "P06 degraded-only hardware-limited online demonstration", "P07 representative maps are not ground-truth semantic-map accuracy"],
    }
    final_report = f"""# Final report (pre-H02 engineering handoff)\n\n## Engineering completion\n\nTracked reproducibility source is bound to commit `{commit}`. The self-contained H02 sheet is `visual_acceptance_sheet.html` (SHA256 `{sha256(sheet)}`); its inputs are recorded in `visual_acceptance_sources.json`. `delivery_manifest.json` binds source pins, licenses, P08 identity, report hashes, known limitations, and ignored assets.\n\n## Scientific result\n\nThe approved P08 result is **{summary.get('outcome_classification')}**. It is not a positive localization-improvement claim. P06 online timing is a separate degraded-only, hardware-limited demonstration; P07 map evidence is representative rather than semantic-map accuracy ground truth.\n\n## H02 status\n\nThis report does not activate H02. Human visual acceptance remains a controller-owned gate.\n"""
    temporary_report = output / ".FINAL_REPORT.md.partial"; temporary_report.write_text(final_report, encoding="utf-8"); temporary_report.replace(output / "FINAL_REPORT.md")
    delivery["report"]["final_report_sha256"] = sha256(output / "FINAL_REPORT.md")
    atomic_json(output / "delivery_manifest.json", delivery)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/final")
    args = parser.parse_args()
    try:
        manifest = render(args.asset_root.resolve(), args.output.resolve())
        print(json.dumps(manifest["sheet"], sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"VISUAL_ACCEPTANCE_RENDER_FAILED: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
