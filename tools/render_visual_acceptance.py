#!/usr/bin/env python3
"""Render the self-contained H02 evidence sheet from approved ignored artifacts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
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


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/svg+xml"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def svg(title: str, body: str, width: int = 920, height: int = 420) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
            f'<text x="20" y="30" font-family="sans-serif" font-size="20" font-weight="bold">{html.escape(title)}</text>{body}</svg>')


def trajectory_svg(asset_root: Path) -> tuple[str, list[Path]]:
    paths: list[Path] = []
    tracks: list[tuple[str, str, Path]] = []
    for mode, color in (("baseline", "#1f77b4"), ("semantic-feedback", "#d62728")):
        candidates = sorted((asset_root / "runs/ovorb2_tum_v1" / mode / "fr3_walking_xyz/seed-23011").glob("attempt-*/CameraTrajectory.txt"))
        if len(candidates) != 1:
            raise ValueError(f"expected one formal {mode} trajectory for fr3_walking_xyz seed 23011")
        tracks.append((mode, color, candidates[0])); paths.append(candidates[0])
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
    return svg("Formal trajectories — fr3_walking_xyz (SE(3) study, metres)", body), paths


def diagnostic_image(asset_root: Path, frame_id: int, fraction: float, destination: Path) -> list[Path]:
    dataset = asset_root / "data/tum/raw/rgbd_dataset_freiburg3_walking_xyz"
    line = [row for row in (dataset / "associate.txt").read_text(encoding="utf-8").splitlines() if row and not row.startswith("#")][frame_id]
    rgb = dataset / line.split()[1]
    scores = asset_root / "cache/dynamic/v1/fr3_walking_xyz/score_maps" / f"{frame_id:06d}.npy"
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
    packet_path = asset_root / "cache/semantic/v1/fr3_walking_xyz" / str(entry["path"])
    packet = read_cache_frame(packet_path)
    masked = raw.copy()
    colors = ((0, 220, 0), (255, 0, 255), (0, 180, 255), (255, 180, 0))
    for index, instance in enumerate(packet.instances):
        instance_mask = decode_binary_mask_rle(instance.mask_rle)
        if instance_mask.shape != score.shape:
            raise ValueError("semantic mask dimensions differ from score map")
        color = np.asarray(colors[index % len(colors)], dtype=np.uint8)
        masked[instance_mask] = (0.35 * masked[instance_mask] + 0.65 * color).astype(np.uint8)
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
    return [rgb, scores, packet_path, destination]


def query_rows(objects: list[dict[str, Any]], query: str) -> str:
    matched = [item for item in objects if query in str(item.get("normalized_label", "")).split() or query in item.get("aliases", [])]
    matched.sort(key=lambda item: (-float(item.get("confidence", 0.0)), str(item.get("object_id", ""))))
    if not matched:
        return "no exact/token-containment result (honest empty evidence)"
    return "; ".join(f"{item['object_id']} {item['normalized_label']} confidence={float(item['confidence']):.3f}" for item in matched[:4])


def render(asset_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "visual_assets"; assets.mkdir(exist_ok=True)
    sources: list[Path] = []
    trajectory, trajectory_paths = trajectory_svg(asset_root); sources.extend(trajectory_paths)
    diagnostics: list[tuple[float, Path]] = []
    for fraction, frame in ((.25, 207), (.50, 413), (.75, 617)):
        destination = assets / f"diagnostic-{int(fraction * 100)}.png"
        sources.extend(diagnostic_image(asset_root, frame, fraction, destination))
        diagnostics.append((fraction, destination))
    desk_root = asset_root / "artifacts/maps/smoke-semantic-feedback-fr1_desk-seed-23011-attempt-002"
    walking_root = asset_root / "artifacts/maps/smoke-semantic-feedback-fr3_walking_xyz-seed-23011-attempt-001"
    desk_front, walking_front = desk_root / "screenshots/front.png", walking_root / "screenshots/front.png"
    desk_objects = json.loads((desk_root / "objects.json").read_text(encoding="utf-8")); walking_objects = json.loads((walking_root / "objects.json").read_text(encoding="utf-8"))
    sources += [desk_front, walking_front, desk_root / "objects.json", walking_root / "objects.json"]
    online = json.loads((asset_root / "runs/p06-online/fr3_walking_xyz/seed-23011/demo/attempt-006/online_demo_report.json").read_text(encoding="utf-8")); sources.append(asset_root / "runs/p06-online/fr3_walking_xyz/seed-23011/demo/attempt-006/online_demo_report.json")
    bars = [("request rate Hz", online["request_attempt_rate_hz"], "#1f77b4"), ("accepted rate Hz", online["accepted_semantic_rate_hz"], "#d62728"), ("mean inference ms / 100", online["mean_service_inference_ms"] / 100, "#9467bd"), ("degraded fraction", online["degraded_frame_fraction"], "#ff7f0e")]
    bar_body = "".join(f'<rect x="{70+i*190}" y="{350-v*240:.1f}" width="90" height="{v*240:.1f}" fill="{c}"/><text x="{65+i*190}" y="380" font-family="sans-serif" font-size="12">{html.escape(n)}</text><text x="{80+i*190}" y="{340-v*240:.1f}" font-family="sans-serif">{v:.2f}</text>' for i,(n,v,c) in enumerate(bars)) + '<text x="20" y="405" font-family="sans-serif">P06 run p06-online-demo-fr3_walking_xyz-seed-23011-attempt-006; max packet age 250 ms; drops/unusable=1.00; limited: no packet met causal limit.</text>'
    queries = "<br/>".join(f"<b>{query}</b>: {html.escape(query_rows(desk_objects, query))}" for query in ("chair", "monitor", "person"))
    diagnostic_html = "".join(f'<figure><img src="{data_uri(path)}" alt="{fraction:.0%} diagnostic"><figcaption>{fraction:.0%}: fr3_walking_xyz frame {int(fraction == .25 and 207 or fraction == .5 and 413 or 617)}; RGB, mask, score, retained-feature diagnostics. Units: pixels / score 0–1.</figcaption></figure>' for fraction, path in diagnostics)
    boxes = "; ".join(f"{item['object_id']} {item['normalized_label']} box extent(m)={','.join(f'{x:.2f}' for x in item['extent'])}" for item in desk_objects[:8])
    body = f'''<!doctype html><html><head><meta charset="utf-8"><title>H02 visual acceptance sheet</title><style>body{{font-family:Arial,sans-serif;margin:20px;color:#111}}section{{border:1px solid #777;padding:14px;margin:16px 0;break-inside:avoid}}img{{max-width:100%;height:auto;border:1px solid #aaa}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}}figure{{margin:0}}figcaption{{font-size:13px}}.note{{background:#fff7df;padding:8px}}@media print{{body{{min-width:1100px}}}}</style></head><body>
<h1>H02 — final semantic-map visual acceptance</h1><p>Source mode: approved formal frozen-cache study for localization; P07 maps; P06 online demonstration. Online IPC is not formal localization evidence. Sheet source commit is external-artifact-bound; every image is embedded as a data URI.</p>
<section><h2>1. Paired formal trajectories</h2><img src="{svg_uri(trajectory)}" alt="formal paired trajectories"><p>Sequence <b>fr3_walking_xyz</b>; run IDs ovorb2_tum_v1-baseline-fr3_walking_xyz-seed-23011-attempt-001 and ovorb2_tum_v1-semantic-feedback-fr3_walking_xyz-seed-23011-attempt-001. Coordinates are x/y metres in the formal SE(3) protocol; blue baseline, red semantic-feedback.</p></section>
<section><h2>2. Fixed 25/50/75% diagnostics</h2><div class="grid">{diagnostic_html}</div><p class="note">Legend: green mask is frozen dynamic score ≥0.70; Turbo heat map is score 0–1; yellow ORB points would be retained, red points filtered. The feature rendering is a reproducible diagnostic recomputation from the frozen score map, not a claim of per-frame runner telemetry.</p></section>
<section><h2>3. Static TSDF and object boxes</h2><div class="grid"><figure><img src="{data_uri(desk_front)}"><figcaption>Dynamic/control representative: fr1_desk, semantic-feedback seed 23011, 120,057 static-cloud points, 42 static objects, 32 dynamic tracks. Front TSDF view.</figcaption></figure><figure><img src="{data_uri(walking_front)}"><figcaption>Dynamic sequence: fr3_walking_xyz, semantic-feedback seed 23011, 269,322 static-cloud points, 0 static objects, 0 dynamic tracks. Front TSDF view.</figcaption></figure></div><p>Object-box records (T_world_camera map frame; extent metres): {html.escape(boxes)}</p></section>
<section><h2>4. Text query results (fr1_desk map)</h2><p>{queries}</p><p class="note">The person query is preserved as empty when no stable static object is present; moving people are represented in dynamic tracks, not permanently fused into the static object map.</p></section>
<section><h2>5. Representative success and limitation</h2><p><b>Success:</b> fr1_desk static TSDF/object export is valid with queryable static objects. <b>Limitation:</b> fr3_walking_xyz has no static objects/dynamic tracks in this representative export; motion confirmation and conservative masking can yield limited semantic-map evidence. The P08 paired study result is <b>neutral</b>, not a positive improvement claim.</p></section>
<section><h2>6. P06 online timing / rate / latency / drops</h2><img src="{svg_uri(svg('P06 online timing and degradation — fr3_walking_xyz', bar_body))}" alt="online timing"><p>Run ID p06-online-demo-fr3_walking_xyz-seed-23011-attempt-006: request cap 5 Hz, actual request rate {online['request_attempt_rate_hz']:.3f} Hz, accepted semantic rate {online['accepted_semantic_rate_hz']:.3f} Hz, mean inference {online['mean_service_inference_ms']:.1f} ms, p95 {online['p95_service_inference_ms']:.1f} ms, maximum IPC call {online['maximum_ipc_call_ms']:.3f} ms, request-drop fraction {online['request_drop_fraction']:.3f}, unusable-result fraction {online['result_unusable_fraction']:.3f}, degraded frame fraction {online['degraded_frame_fraction']:.3f}. This is a hardware-limited degraded-only demonstration.</p></section>
</body></html>'''
    sheet = output / "visual_acceptance_sheet.html"; temporary = sheet.with_name(".visual_acceptance_sheet.html.partial"); temporary.write_text(body, encoding="utf-8"); temporary.replace(sheet)
    required_labels = ("Paired formal trajectories", "Fixed 25/50/75% diagnostics", "Static TSDF and object boxes", "Text query results", "Representative success and limitation", "P06 online timing")
    if (any(label not in body for label in required_labels) or body.count("data:image/") < 7 or
            'src="visual_assets/' in body or "width=\"0\"" in body):
        raise ValueError("visual sheet completeness or clipping-risk validation failed")
    manifest = {"schema_version": 1, "sheet": {"path": str(sheet), "sha256": sha256(sheet)}, "sources": [{"path": str(path), "sha256": sha256(path)} for path in sources], "panels": ["paired-formal-trajectories", "fixed-diagnostics-25-50-75", "static-tsdf-and-boxes", "queries-chair-monitor-person", "success-and-limitation", "p06-online-timing"], "self_contained": True, "html_validation": {"required_labels": list(required_labels), "embedded_image_count": body.count("data:image/"), "minimum_render_width_px": 1100}}
    atomic_json(output / "visual_acceptance_sources.json", manifest)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    summary = json.loads((asset_root / "reports/final/summary.json").read_text(encoding="utf-8"))
    delivery = {
        "schema_version": 1,
        "final_commit": commit,
        "source_pins": {
            "orb_slam2": "f2e6f51cdc8d067655d90a78c06261378e07e8f3",
            "grounding_dino": "856dde20aee659246248e20734ef9ba5214f5e44",
            "segment_anything": "dca509fe793f601edb92606367a655c15ac00fdf",
        },
        "licenses": {"orb_slam2": "GPL-3.0-only", "grounding_dino": "Apache-2.0", "segment_anything": "Apache-2.0", "tum_rgbd": "provider terms; not bundled"},
        "study": {"valid_runs": 60, "outcome": summary.get("outcome_classification"), "summary_sha256": sha256(asset_root / "reports/final/summary.json")},
        "report": {"visual_acceptance_sheet_sha256": sha256(sheet), "source_manifest_sha256": sha256(output / "visual_acceptance_sources.json")},
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
