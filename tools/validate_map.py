#!/usr/bin/env python3
"""Validate all P07 map payloads and emit a machine-readable integrity report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from PIL import Image


FROZEN_CONFIG_VALUE_SHA256 = "9606608f73cfd9eaa79a7a6b8751bb0724bd7ef3ce7932fa6a97cae2253a3266"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid {label}")
    return array


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _is_commit(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 40 and
            all(character in "0123456789abcdef" for character in value))


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest_claims(manifest: dict[str, object]) -> None:
    required = {
        "schema", "schema_version", "study_id", "sequence_id", "seed",
        "producer_commit", "command", "created_utc", "hostname", "valid",
        "pose_convention", "pose_timestamp_identity", "config", "inputs",
        "counts", "cache_completion", "semantic_cache_producer",
        "dynamic_cache_producer", "outputs",
    }
    if set(manifest) != required:
        raise ValueError("map manifest fields do not match schema")
    if (manifest.get("schema") != "ovorb.map.v1" or
            manifest.get("schema_version") != 1 or
            manifest.get("study_id") != "p07-static-map-v1" or
            manifest.get("seed") != 23011 or manifest.get("valid") is not True or
            manifest.get("pose_convention") != "T_world_camera"):
        raise ValueError("map manifest pose convention or frozen identity is invalid")
    if (manifest.get("pose_timestamp_identity") !=
            "RGB timestamp rounded to six decimal places"):
        raise ValueError("map pose timestamp identity mismatch")
    if (not isinstance(manifest.get("sequence_id"), str) or
            not manifest["sequence_id"] or
            not isinstance(manifest.get("command"), list) or
            not manifest["command"] or
            not all(isinstance(item, str) and item for item in manifest["command"]) or
            not isinstance(manifest.get("created_utc"), str) or
            not isinstance(manifest.get("hostname"), str) or
            not manifest["hostname"] or
            not _is_commit(manifest.get("producer_commit")) or
            not _is_commit(manifest.get("semantic_cache_producer")) or
            not _is_commit(manifest.get("dynamic_cache_producer"))):
        raise ValueError("map manifest provenance is invalid")
    if _sha256_json(manifest.get("config")) != FROZEN_CONFIG_VALUE_SHA256:
        raise ValueError("map manifest configuration hash mismatch")
    inputs = manifest.get("inputs")
    expected_inputs = {
        "config", "association", "dataset_manifest", "dataset_source_tree_sha256",
        "settings", "selected_run_manifest", "trajectory", "semantic_manifest",
        "semantic_index", "semantic_completion", "dynamic_manifest",
        "dynamic_index", "dynamic_tracks", "dynamic_completion",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_inputs:
        raise ValueError("map manifest input identities are incomplete")
    if not _is_sha256(inputs["dataset_source_tree_sha256"]):
        raise ValueError("map dataset tree identity is invalid")
    for name in expected_inputs - {"dataset_source_tree_sha256"}:
        value = inputs[name]
        if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"} or
                not isinstance(value["path"], str) or not value["path"] or
                not _is_sha256(value["sha256"]) or
                type(value["size_bytes"]) is not int or value["size_bytes"] < 0):
            raise ValueError(f"invalid map input identity: {name}")
    counts = manifest.get("counts")
    expected_counts = {
        "association_frames", "trajectory_poses", "integrated_frames",
        "excluded_frames", "dynamic_pixel_exclusions", "invalid_depth_pixels",
        "object_observations", "static_objects", "dynamic_tracks",
    }
    if not isinstance(counts, dict) or set(counts) != expected_counts:
        raise ValueError("map manifest counts are incomplete")
    numeric_names = expected_counts - {"excluded_frames"}
    if any(type(counts[name]) is not int or counts[name] < 0 for name in numeric_names):
        raise ValueError("map manifest counts are invalid")
    excluded = counts["excluded_frames"]
    if (not isinstance(excluded, dict) or set(excluded) != {"MISSING_EXACT_POSE"} or
            type(excluded["MISSING_EXACT_POSE"]) is not int or
            excluded["MISSING_EXACT_POSE"] < 0 or
            counts["association_frames"] !=
            counts["integrated_frames"] + excluded["MISSING_EXACT_POSE"]):
        raise ValueError("map frame accounting is invalid")
    completion = manifest.get("cache_completion")
    if (not isinstance(completion, dict) or
            set(completion) != {"semantic", "dynamic"} or
            not all(isinstance(value, dict) for value in completion.values())):
        raise ValueError("map cache completion claims are invalid")


def _validate_objects(root: Path, value: object) -> int:
    if not isinstance(value, list):
        raise ValueError("objects.json must contain a list")
    ids: set[str] = set()
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise ValueError(f"object {index} is not a JSON object")
        required = {
            "object_id", "normalized_label", "aliases", "confidence",
            "confidence_history", "observation_range", "centroid",
            "orientation", "extent", "point_count", "source_track",
            "box_fallback",
        }
        if set(record) != required:
            raise ValueError(f"object {index} fields do not match schema")
        object_id = record["object_id"]
        confidence = record["confidence"]
        history = record["confidence_history"]
        if (not isinstance(object_id, str) or not object_id or object_id in ids or
                not isinstance(record["normalized_label"], str) or
                not record["normalized_label"] or
                not isinstance(record["aliases"], list) or
                not all(isinstance(item, str) and item for item in record["aliases"]) or
                type(confidence) not in (int, float) or
                not math.isfinite(float(confidence)) or
                not 0.0 <= float(confidence) <= 1.0 or
                not isinstance(history, list) or not history or
                not all(type(item) in (int, float) and math.isfinite(float(item))
                        and 0.0 <= float(item) <= 1.0 for item in history) or
                not isinstance(record["source_track"], str) or
                not record["source_track"] or
                record["box_fallback"] not in (None, "AXIS_ALIGNED_DEGENERATE")):
            raise ValueError(f"invalid object metadata: {object_id}")
        ids.add(object_id)
        _finite_array(record["observation_range"], (2,), "observation range")
        _finite_array(record["centroid"], (3,), "centroid")
        orientation = _finite_array(record["orientation"], (3, 3), "orientation")
        if (not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-6) or
                not math.isclose(float(np.linalg.det(orientation)), 1.0, abs_tol=1e-6)):
            raise ValueError(f"object orientation is not a proper rotation: {object_id}")
        extent = _finite_array(record["extent"], (3,), "extent")
        point_count = record["point_count"]
        if (np.any(extent < 0.0) or type(point_count) is not int or point_count <= 0):
            raise ValueError(f"invalid object support: {object_id}")
        cloud = o3d.io.read_point_cloud(str(root / "objects" / f"{object_id}.ply"))
        points = np.asarray(cloud.points, dtype=np.float64)
        if (points.shape != (point_count, 3) or not np.all(np.isfinite(points))):
            raise ValueError(f"object cloud support mismatch: {object_id}")
        centroid = _finite_array(record["centroid"], (3,), "centroid")
        local = (points - centroid) @ orientation
        if np.any(np.abs(local) > extent / 2.0 + 1e-5):
            raise ValueError(f"object box does not contain its support: {object_id}")
    return len(value)


def validate_map(map_root: Path) -> dict[str, object]:
    root = Path(map_root).resolve()
    manifest_path = root / "map_manifest.json"
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != "ovorb.map.v1":
        raise ValueError("invalid map manifest schema")
    _validate_manifest_claims(manifest)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("map manifest has no outputs")
    declared: set[Path] = set()
    for relative_value, metadata in outputs.items():
        relative = Path(str(relative_value))
        if relative.is_absolute() or ".." in relative.parts or relative == Path("map_manifest.json"):
            raise ValueError(f"unsafe map output path: {relative}")
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid map output metadata: {relative}")
        path = root / relative
        declared.add(path)
        if not path.is_file():
            raise ValueError(f"missing map output: {relative}")
        if path.stat().st_size != metadata.get("size_bytes"):
            raise ValueError(f"size mismatch for map output: {relative}")
        if _sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"hash mismatch for map output: {relative}")
    observed = {
        path for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed != declared:
        raise ValueError("map contains undeclared or unmaterialized outputs")

    cloud = o3d.io.read_point_cloud(str(root / "static_cloud.ply"))
    mesh = o3d.io.read_triangle_mesh(str(root / "static_mesh.ply"))
    cloud_points = np.asarray(cloud.points, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if (cloud_points.ndim != 2 or cloud_points.shape[1:] != (3,) or
            cloud_points.shape[0] == 0 or not np.all(np.isfinite(cloud_points)) or
            vertices.ndim != 2 or vertices.shape[1:] != (3,) or
            vertices.shape[0] == 0 or not np.all(np.isfinite(vertices)) or
            triangles.ndim != 2 or triangles.shape[1:] != (3,) or
            triangles.shape[0] == 0 or np.any(triangles < 0) or
            np.any(triangles >= vertices.shape[0])):
        raise ValueError("static PLY geometry is empty or invalid")
    static_objects = _validate_objects(root, _load_json(root / "objects.json"))
    dynamic_rows = 0
    dynamic_track_states: dict[int, bool] = {}
    for line_number, line in enumerate(
            (root / "dynamic_tracks.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid dynamic track row {line_number}") from exc
        track_id = row.get("track_id") if isinstance(row, dict) else None
        strong_dynamic = row.get("strong_dynamic") if isinstance(row, dict) else None
        if (type(track_id) is not int or track_id < 0 or
                type(strong_dynamic) is not bool):
            raise ValueError(f"invalid dynamic track row {line_number}")
        dynamic_track_states[track_id] = (
            dynamic_track_states.get(track_id, False) or strong_dynamic
        )
        dynamic_rows += 1
    if any(not observed_strong for observed_strong in dynamic_track_states.values()):
        raise ValueError("exported track never reaches strong dynamic state")
    counts = manifest["counts"]
    if (counts["static_objects"] != static_objects or
            counts["dynamic_tracks"] != len(dynamic_track_states)):
        raise ValueError("map manifest object or dynamic-track count mismatch")
    screenshots = sorted(path for path in declared if path.suffix.lower() == ".png")
    if {str(path.relative_to(root)) for path in screenshots} != {
            "screenshots/front.png", "screenshots/top.png"}:
        raise ValueError("map does not contain the fixed screenshots")
    for path in screenshots:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != (960, 720) or image.mode != "RGB":
                raise ValueError(f"invalid screenshot: {path.name}")
    required_outputs = {
        "static_cloud.ply", "static_mesh.ply", "objects.json",
        "dynamic_tracks.jsonl", "screenshots/front.png", "screenshots/top.png",
        *{f"objects/{object_id}.ply" for object_id in {
            item["object_id"] for item in _load_json(root / "objects.json")
        }},
    }
    if set(outputs) != required_outputs:
        raise ValueError("map output set does not match its object records")
    return {
        "schema": "ovorb.map-integrity.v1",
        "valid": True,
        "map_root": str(root),
        "map_manifest_sha256": _sha256_file(manifest_path),
        "declared_outputs": len(declared),
        "cloud_points": int(cloud_points.shape[0]),
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_triangles": int(triangles.shape[0]),
        "static_objects": static_objects,
        "dynamic_tracks": len(dynamic_track_states),
        "dynamic_track_rows": dynamic_rows,
        "screenshots": [str(path.relative_to(root)) for path in screenshots],
    }


def _write_json_atomic(path: Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.partial"
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_map(args.map_root)
    if args.report is not None:
        _write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
