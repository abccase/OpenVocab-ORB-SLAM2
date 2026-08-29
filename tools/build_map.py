#!/usr/bin/env python3
"""Build a reproducible static TSDF and semantic object map."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import socket
import subprocess
from pathlib import Path
import sys
from typing import Mapping

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_py.openvocab_slam.cache import read_cache_frame
from semantic_py.openvocab_slam.dynamic_cache import hash_dataset_tree
from semantic_py.openvocab_slam.map import (
    CameraIntrinsics,
    ObjectAggregationConfig,
    ObjectPointObservation,
    ScreenshotView,
    StaticTsdfVolume,
    TsdfConfig,
    TWorldCamera,
    aggregate_static_objects,
    export_map_artifacts,
)
from semantic_py.openvocab_slam.schemas import decode_binary_mask_rle


FROZEN_MAP_CONFIG_SHA256 = "cfcc4341c8e005b3e375c92058cbc2412194d73ed82ffd841305a9eabf2adb09"


@dataclass(frozen=True)
class SelectedRun:
    manifest_path: Path
    trajectory_path: Path
    manifest: dict[str, object]


@dataclass(frozen=True)
class AssociationRow:
    frame_id: int
    rgb_timestamp_lexeme: str
    rgb_path: Path
    depth_timestamp_lexeme: str
    depth_path: Path


@dataclass(frozen=True)
class MapBuildConfig:
    schema: str
    study_id: str
    seed: int
    tsdf: TsdfConfig
    static_score_max_exclusive: float
    points_per_observation: int
    objects: ObjectAggregationConfig
    screenshot_views: tuple[ScreenshotView, ...]

    @classmethod
    def from_path(cls, path: Path) -> "MapBuildConfig":
        value = _read_json_object(path, "map configuration")
        if set(value) != {"schema", "study_id", "seed", "tsdf", "objects", "screenshots"}:
            raise ValueError("map configuration fields do not match schema")
        if (value["schema"] != "ovorb.map-config.v1" or
                value["study_id"] != "p07-static-map-v1" or
                value["seed"] != 23011):
            raise ValueError("map configuration frozen identity mismatch")
        if _sha256_file(path) != FROZEN_MAP_CONFIG_SHA256:
            raise ValueError("map configuration frozen hash mismatch")
        tsdf = value["tsdf"]
        objects = value["objects"]
        screenshots = value["screenshots"]
        if not isinstance(tsdf, dict) or not isinstance(objects, dict) or not isinstance(screenshots, list):
            raise ValueError("map configuration sections have invalid types")
        if set(tsdf) != {
            "voxel_length_m", "sdf_trunc_m", "depth_trunc_m", "dynamic_threshold"
        }:
            raise ValueError("TSDF configuration fields do not match schema")
        if set(objects) != {
            "static_score_max_exclusive", "points_per_observation",
            "dbscan_eps_m", "dbscan_min_samples", "trim_quantile",
            "min_object_points", "max_object_points", "degeneracy_ratio",
        }:
            raise ValueError("object configuration fields do not match schema")
        static_limit = float(objects["static_score_max_exclusive"])
        points_per_observation = int(objects["points_per_observation"])
        if (not math.isfinite(static_limit) or not 0.0 < static_limit <= 1.0 or
                points_per_observation <= 0):
            raise ValueError("invalid object static confidence configuration")
        views = tuple(ScreenshotView(
            str(item["name"]),
            float(item["elevation_degrees"]),
            float(item["azimuth_degrees"]),
        ) for item in screenshots if isinstance(item, dict))
        if len(views) != len(screenshots) or not views or len({view.name for view in views}) != len(views):
            raise ValueError("invalid screenshot view configuration")
        return cls(
            schema=str(value["schema"]),
            study_id=str(value["study_id"]),
            seed=int(value["seed"]),
            tsdf=TsdfConfig(**{key: float(item) for key, item in tsdf.items()}),
            static_score_max_exclusive=static_limit,
            points_per_observation=points_per_observation,
            objects=ObjectAggregationConfig(
                dbscan_eps_m=float(objects["dbscan_eps_m"]),
                dbscan_min_samples=int(objects["dbscan_min_samples"]),
                trim_quantile=float(objects["trim_quantile"]),
                min_object_points=int(objects["min_object_points"]),
                max_object_points=int(objects["max_object_points"]),
                degeneracy_ratio=float(objects["degeneracy_ratio"]),
            ),
            screenshot_views=views,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: {path}")
    return value


def append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"registry row {line_number} is not an object")
        rows.append(value)
    return rows


def select_registered_run(
    registry_path: Path,
    sequence_id: str,
    seed: int,
) -> SelectedRun:
    latest: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(registry_path):
        run_id = row.get("run_id")
        if isinstance(run_id, str) and run_id:
            latest[run_id] = row
    matches = [
        row for row in latest.values()
        if row.get("mode") == "semantic-feedback"
        and row.get("sequence_id") == sequence_id
        and row.get("seed") == seed
        and row.get("state") == "COMPLETED"
        and row.get("valid") is True
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one valid registered semantic-feedback run for "
            f"{sequence_id} seed={seed}; found {len(matches)}"
        )
    selected = matches[0]
    run_dir = Path(str(selected.get("cwd", ""))).resolve()
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid selected run manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("selected run manifest is not an object")
    if (manifest.get("schema_version") != 2 or manifest.get("study") != "smoke" or
            manifest.get("mode") != "semantic-feedback" or
            manifest.get("state") != "COMPLETED" or manifest.get("valid") is not True or
            manifest.get("exit_code") != 0 or manifest.get("invalid_reason") is not None or
            type(manifest.get("expected_frames")) is not int or
            type(manifest.get("frame_count")) is not int or
            manifest.get("expected_frames") != manifest.get("frame_count") or
            int(manifest["frame_count"]) <= 0):
        raise ValueError("selected run completion identity is invalid")
    for key in ("run_id", "mode", "sequence_id", "seed", "state", "valid"):
        if manifest.get(key) != selected.get(key):
            raise ValueError(f"selected run manifest disagrees with registry: {key}")
    trajectory_value = manifest.get("trajectory")
    if not isinstance(trajectory_value, dict):
        raise ValueError("selected run manifest lacks trajectory metadata")
    trajectory_path = _validated_run_artifact(
        run_dir, trajectory_value, "trajectory"
    )
    trajectory_poses = load_tum_trajectory(trajectory_path)
    if trajectory_value.get("pose_count") != len(trajectory_poses):
        raise ValueError("selected trajectory pose count mismatch")
    keyframe_value = manifest.get("keyframe_trajectory")
    if not isinstance(keyframe_value, dict):
        raise ValueError("selected run lacks keyframe trajectory metadata")
    keyframe_path = _validated_run_artifact(
        run_dir, keyframe_value, "keyframe trajectory"
    )
    if keyframe_value.get("pose_count") != len(load_tum_trajectory(keyframe_path)):
        raise ValueError("selected keyframe trajectory pose count mismatch")
    for key in ("telemetry", "timings", "final_state", "stdout", "stderr"):
        value = manifest.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"selected run lacks {key} metadata")
        _validated_run_artifact(run_dir, value, key)
    if manifest["telemetry"].get("format") != "csv":
        raise ValueError("selected run telemetry format mismatch")
    cache_identity = manifest.get("cache_identity")
    verified_inputs = manifest.get("verified_inputs")
    registration = manifest.get("registration_identity")
    if (not _valid_cache_identity(cache_identity) or
            not isinstance(verified_inputs, dict) or
            not isinstance(registration, dict) or
            registration.get("study") != manifest["study"] or
            registration.get("mode") != manifest["mode"] or
            registration.get("sequence_id") != manifest["sequence_id"] or
            registration.get("seed") != manifest["seed"] or
            registration.get("expected_frames") != manifest["expected_frames"] or
            registration.get("cache_identity") != cache_identity or
            registration.get("verified_inputs") != verified_inputs):
        raise ValueError("selected run registration identity mismatch")
    if manifest != selected:
        raise ValueError("selected run manifest differs from final registry record")
    return SelectedRun(manifest_path.resolve(), trajectory_path, manifest)


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _valid_cache_identity(value: object) -> bool:
    return (isinstance(value, dict) and
            set(value) == {"manifest_sha256", "completion_sha256", "index_sha256"} and
            all(_is_sha256(item) for item in value.values()))


def _validated_run_artifact(
    run_dir: Path, value: Mapping[str, object], label: str
) -> Path:
    relative = Path(str(value.get("path", "")))
    if relative.is_absolute() or relative.parent != Path(".") or not relative.name:
        raise ValueError(f"unsafe selected run {label} path")
    path = run_dir / relative
    if not path.is_file() or path.stat().st_size != value.get("size_bytes"):
        raise ValueError(f"selected {label} size mismatch")
    if _sha256_file(path) != value.get("sha256"):
        raise ValueError(f"selected {label} hash mismatch")
    return path.resolve()


def _timestamp_key(value: str | float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("timestamp must be finite")
    return f"{numeric:.6f}"


def _quaternion_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("trajectory quaternion is invalid")
    w, x, y, z = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_tum_trajectory(path: Path) -> dict[str, TWorldCamera]:
    poses: dict[str, TWorldCamera] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 8:
            raise ValueError(f"trajectory row {line_number} must have 8 fields")
        values = [float(field) for field in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"trajectory row {line_number} is not finite")
        key = _timestamp_key(fields[0])
        if key in poses:
            raise ValueError(f"duplicate trajectory timestamp identity: {key}")
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _quaternion_rotation(*values[4:8])
        matrix[:3, 3] = values[1:4]
        poses[key] = TWorldCamera.from_matrix(matrix)
    if not poses:
        raise ValueError("trajectory is empty")
    return poses


def pose_for_association_timestamp(
    poses: Mapping[str, TWorldCamera], timestamp_lexeme: str
) -> TWorldCamera | None:
    return poses.get(_timestamp_key(timestamp_lexeme))


def read_associations(path: Path) -> list[AssociationRow]:
    rows: list[AssociationRow] = []
    previous = -math.inf
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"association row {line_number} must have four fields")
        timestamp = float(fields[0])
        depth_timestamp = float(fields[2])
        rgb_path, depth_path = Path(fields[1]), Path(fields[3])
        if (not math.isfinite(timestamp) or not math.isfinite(depth_timestamp) or
                timestamp <= previous or rgb_path.is_absolute() or depth_path.is_absolute() or
                ".." in rgb_path.parts or ".." in depth_path.parts or
                rgb_path.parts[:1] != ("rgb",) or depth_path.parts[:1] != ("depth",)):
            raise ValueError(f"association row {line_number} is invalid")
        rows.append(AssociationRow(
            len(rows), fields[0], rgb_path, fields[2], depth_path
        ))
        previous = timestamp
    if not rows:
        raise ValueError("association is empty")
    return rows


def load_camera_settings(path: Path, width: int, height: int) -> tuple[CameraIntrinsics, float]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"unable to open camera settings: {path}")
    try:
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=float(storage.getNode("Camera.fx").real()),
            fy=float(storage.getNode("Camera.fy").real()),
            cx=float(storage.getNode("Camera.cx").real()),
            cy=float(storage.getNode("Camera.cy").real()),
        )
        depth_scale = float(storage.getNode("DepthMapFactor").real())
    finally:
        storage.release()
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("invalid depth scale")
    return intrinsics, depth_scale


def _backproject_mask_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: TWorldCamera,
    limit: int,
    depth_trunc_m: float,
) -> np.ndarray:
    valid = (
        np.asarray(mask, dtype=bool) & np.isfinite(depth_m) &
        (depth_m > 0.0) & (depth_m <= depth_trunc_m)
    )
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if rows.size > limit:
        selected = np.linspace(0, rows.size - 1, limit, dtype=int)
        rows, columns = rows[selected], columns[selected]
    depth = depth_m[rows, columns]
    camera = np.column_stack((
        (columns - intrinsics.cx) * depth / intrinsics.fx,
        (rows - intrinsics.cy) * depth / intrinsics.fy,
        depth,
    ))
    return pose.transform_points(camera)


def _read_verified_index(root: Path, name: str) -> list[dict[str, object]]:
    return _read_jsonl(root / name)


def _cache_payload_path(root: Path, value: object, expected_parent: str) -> Path:
    relative = Path(str(value))
    if (relative.is_absolute() or ".." in relative.parts or
            relative.parent != Path(expected_parent)):
        raise ValueError(f"unsafe cache payload path: {relative}")
    return root / relative


def classify_track_rows(
    rows: list[dict[str, object]],
) -> tuple[dict[tuple[int, int], dict[str, object]], set[int]]:
    by_instance: dict[tuple[int, int], dict[str, object]] = {}
    dynamic_ids: set[int] = set()
    for row in rows:
        frame_id = row.get("frame_id")
        local_id = row.get("local_id")
        track_id = row.get("track_id")
        strong_dynamic = row.get("strong_dynamic")
        if (type(frame_id) is not int or frame_id < 0 or
                type(local_id) is not int or local_id < 0 or
                (track_id is not None and
                 (type(track_id) is not int or track_id < 0)) or
                type(strong_dynamic) is not bool):
            raise ValueError("invalid dynamic track row identity")
        key = (frame_id, local_id)
        if key in by_instance:
            raise ValueError(f"duplicate dynamic track row: {key}")
        by_instance[key] = row
        if track_id is not None and strong_dynamic:
            dynamic_ids.add(track_id)
    return by_instance, dynamic_ids


def canonical_track_id(track_id: int) -> str:
    if type(track_id) is not int or track_id < 0:
        raise ValueError("track ID must be a nonnegative integer")
    return f"track-{track_id:06d}"


def validate_selected_cache_binding(
    run_manifest: Mapping[str, object],
    expected_cache_identity: Mapping[str, object],
    expected_verified_inputs: Mapping[str, object],
) -> None:
    if run_manifest.get("cache_identity") != dict(expected_cache_identity):
        raise ValueError("selected run cache identity does not match current cache")
    observed = run_manifest.get("verified_inputs")
    if not isinstance(observed, dict) or any(
            observed.get(key) != value
            for key, value in expected_verified_inputs.items()):
        raise ValueError("selected run verified inputs do not match current cache")


def _validate_cache_completion(
    cache_root: Path,
    *,
    sequence_id: str,
    association_sha256: str,
    source_tree_sha256: str,
    dataset_manifest_sha256: str,
    expected_frame_count: int,
    study_id: str,
    dynamic: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = cache_root / "cache_manifest.json"
    index_path = cache_root / "cache_index.jsonl"
    complete_path = cache_root / "cache_complete.json"
    manifest = _read_json_object(manifest_path, "cache manifest")
    complete = _read_json_object(complete_path, "cache completion")
    expected_schema = (
        "ovorb.dynamic-cache.v1" if dynamic else "ovorb.semantic-cache.v1"
    )
    if (manifest.get("schema") != expected_schema or
            manifest.get("study_id") != study_id or
            manifest.get("sequence_id") != sequence_id or
            manifest.get("association_sha256") != association_sha256 or
            manifest.get("source_tree_sha256") != source_tree_sha256 or
            manifest.get("expected_frame_count") != expected_frame_count):
        label = "dynamic" if dynamic else "semantic"
        raise ValueError(f"{label} cache manifest identity mismatch")
    if (complete.get("manifest_sha256") != _sha256_file(manifest_path) or
            complete.get("index_sha256") != _sha256_file(index_path) or
            complete.get("frame_count") != expected_frame_count or
            len(_read_jsonl(index_path)) != expected_frame_count):
        raise ValueError("cache completion hash mismatch")
    if dynamic:
        tracks_path = cache_root / "dynamic_tracks.jsonl"
        identity_path = cache_root / "semantic_identity.jsonl"
        diagnostics_path = cache_root / "diagnostics_index.jsonl"
        dynamic_config = manifest.get("dynamic_config")
        if (manifest.get("dataset_manifest_sha256") != dataset_manifest_sha256 or
                not isinstance(dynamic_config, dict) or
                manifest.get("dynamic_config_sha256") != _sha256_json(dynamic_config) or
                complete.get("tracks_sha256") != _sha256_file(tracks_path) or
                complete.get("semantic_identity_sha256") != _sha256_file(identity_path) or
                complete.get("semantic_identity_sha256") !=
                manifest.get("semantic_identity_sha256") or
                complete.get("diagnostics_index_sha256") !=
                _sha256_file(diagnostics_path) or
                len(_read_jsonl(identity_path)) != expected_frame_count or
                len(_read_jsonl(tracks_path)) !=
                int(manifest.get("expected_instance_count", -1))):
            raise ValueError("dynamic track hash mismatch")
    return manifest, complete


def _artifact_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _require_clean_tree(root: Path) -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
    )
    if status:
        raise ValueError("product tree must be clean before real map generation")


def build_sequence_map(
    *,
    project_root: Path,
    sequence_id: str,
    registry_path: Path,
    config_path: Path,
    dataset_root: Path,
    dataset_manifest_path: Path,
    settings_path: Path,
    semantic_cache_root: Path,
    dynamic_cache_root: Path,
    output_root: Path,
    command: list[str],
) -> Path:
    root = Path(project_root).resolve()
    config = MapBuildConfig.from_path(config_path)
    selected = select_registered_run(registry_path, sequence_id, config.seed)
    association_path = dataset_root / "associate.txt"
    association_sha = _sha256_file(association_path)
    associations = read_associations(association_path)
    dataset_manifest = _read_json_object(dataset_manifest_path, "dataset manifest")
    if (dataset_manifest.get("sequence_id") != sequence_id or
            dataset_manifest.get("association_sha256") != association_sha or
            dataset_manifest.get("validation_status") != "VALID" or
            int(dataset_manifest.get("counts", {}).get("associations", -1)) != len(associations)):
        raise ValueError("dataset manifest identity mismatch")
    source_tree_sha = hash_dataset_tree(dataset_root)
    if source_tree_sha != dataset_manifest.get("extracted_tree_sha256"):
        raise ValueError("dataset source tree hash mismatch")
    run_verified = selected.manifest.get("verified_inputs")
    if (not isinstance(run_verified, dict) or
            run_verified.get("source_tree_sha256") != source_tree_sha or
            selected.manifest.get("association_sha256") != association_sha or
            selected.manifest.get("dataset_manifest_sha256") != _sha256_file(dataset_manifest_path)):
        raise ValueError("selected run input identity mismatch")

    semantic_manifest, semantic_complete = _validate_cache_completion(
        semantic_cache_root,
        sequence_id=sequence_id,
        association_sha256=association_sha,
        source_tree_sha256=source_tree_sha,
        dataset_manifest_sha256=_sha256_file(dataset_manifest_path),
        expected_frame_count=len(associations),
        study_id="ovorb2_tum_v1",
        dynamic=False,
    )
    dynamic_manifest, dynamic_complete = _validate_cache_completion(
        dynamic_cache_root,
        sequence_id=sequence_id,
        association_sha256=association_sha,
        source_tree_sha256=source_tree_sha,
        dataset_manifest_sha256=_sha256_file(dataset_manifest_path),
        expected_frame_count=len(associations),
        study_id="ovorb2_tum_v1",
        dynamic=True,
    )
    if (dynamic_manifest.get("semantic_manifest_sha256") !=
            _sha256_file(semantic_cache_root / "cache_manifest.json") or
            dynamic_manifest.get("semantic_index_sha256") !=
            _sha256_file(semantic_cache_root / "cache_index.jsonl") or
            int(dynamic_manifest.get("expected_frame_count", -1)) != len(associations)):
        raise ValueError("dynamic cache is not bound to the semantic cache")
    current_cache_identity = {
        "manifest_sha256": _sha256_file(dynamic_cache_root / "cache_manifest.json"),
        "completion_sha256": _sha256_file(dynamic_cache_root / "cache_complete.json"),
        "index_sha256": _sha256_file(dynamic_cache_root / "cache_index.jsonl"),
    }
    current_verified_inputs = {
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "source_tree_sha256": source_tree_sha,
        "dynamic_manifest_sha256": current_cache_identity["manifest_sha256"],
        "dynamic_completion_sha256": current_cache_identity["completion_sha256"],
        "dynamic_index_sha256": current_cache_identity["index_sha256"],
        "dynamic_config_sha256": dynamic_manifest.get("dynamic_config_sha256"),
        "semantic_manifest_sha256": _sha256_file(
            semantic_cache_root / "cache_manifest.json"
        ),
        "semantic_identity_sha256": dynamic_manifest.get("semantic_identity_sha256"),
        "inference_config_sha256": semantic_manifest.get("inference_config_sha256"),
        "prompt_sha256": semantic_manifest.get("prompt_sha256"),
        "protocol": {
            "dynamic": dynamic_manifest.get("schema"),
            "semantic": semantic_manifest.get("schema"),
        },
    }
    validate_selected_cache_binding(
        selected.manifest, current_cache_identity, current_verified_inputs
    )
    height = int(dataset_manifest["image_dimensions"]["height"])
    width = int(dataset_manifest["image_dimensions"]["width"])
    intrinsics, depth_scale = load_camera_settings(settings_path, width, height)
    if not math.isclose(
            depth_scale,
            float(dynamic_manifest["dynamic_config"]["depth_scale"]),
            rel_tol=0.0,
            abs_tol=1e-12):
        raise ValueError("map and dynamic-cache depth scales differ")
    if not math.isclose(
            config.tsdf.dynamic_threshold,
            float(dynamic_manifest["dynamic_config"]["dynamic_enter_threshold"]),
            rel_tol=0.0,
            abs_tol=1e-12):
        raise ValueError("map and dynamic-cache dynamic thresholds differ")

    semantic_index = _read_verified_index(semantic_cache_root, "cache_index.jsonl")
    dynamic_index = _read_verified_index(dynamic_cache_root, "cache_index.jsonl")
    track_rows = _read_verified_index(dynamic_cache_root, "dynamic_tracks.jsonl")
    if len(semantic_index) != len(associations) or len(dynamic_index) != len(associations):
        raise ValueError("cache index coverage mismatch")
    track_by_instance, dynamic_track_ids = classify_track_rows(track_rows)

    poses = load_tum_trajectory(selected.trajectory_path)
    volume = StaticTsdfVolume(intrinsics, config.tsdf)
    observations: list[ObjectPointObservation] = []
    integrated = 0
    exclusion_counts = {"MISSING_EXACT_POSE": 0}
    dynamic_pixel_exclusions = 0
    invalid_depth_pixels = 0
    for association, semantic_entry, dynamic_entry in zip(
            associations, semantic_index, dynamic_index):
        frame_id = association.frame_id
        if (int(semantic_entry.get("frame_id", -1)) != frame_id or
                int(dynamic_entry.get("frame_id", -1)) != frame_id or
                _timestamp_key(semantic_entry.get("timestamp")) !=
                _timestamp_key(association.rgb_timestamp_lexeme) or
                _timestamp_key(dynamic_entry.get("timestamp")) !=
                _timestamp_key(association.rgb_timestamp_lexeme)):
            raise ValueError(f"cache index alignment mismatch at frame {frame_id}")
        semantic_path = _cache_payload_path(
            semantic_cache_root, semantic_entry["path"], "frames"
        )
        score_path = _cache_payload_path(
            dynamic_cache_root, dynamic_entry["path"], "score_maps"
        )
        if (_sha256_file(semantic_path) != semantic_entry.get("sha256") or
                _sha256_file(score_path) != dynamic_entry.get("sha256") or
                dynamic_entry.get("semantic_packet_sha256") != semantic_entry.get("sha256")):
            raise ValueError(f"cache payload hash mismatch at frame {frame_id}")
        pose = pose_for_association_timestamp(poses, association.rgb_timestamp_lexeme)
        if pose is None:
            exclusion_counts["MISSING_EXACT_POSE"] += 1
            continue
        color_bgr = cv2.imread(str(dataset_root / association.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(dataset_root / association.depth_path), cv2.IMREAD_UNCHANGED)
        if (color_bgr is None or color_bgr.shape != (height, width, 3) or
                depth_raw is None or depth_raw.shape != (height, width)):
            raise ValueError(f"invalid RGB-D payload at frame {frame_id}")
        score_map = np.load(score_path, allow_pickle=False)
        if score_map.dtype != np.float32 or score_map.shape != (height, width) or not np.all(np.isfinite(score_map)):
            raise ValueError(f"invalid dynamic score map at frame {frame_id}")
        depth_m = np.asarray(depth_raw, dtype=np.float32) / np.float32(depth_scale)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        dynamic_pixel_exclusions += int(np.count_nonzero(
            score_map >= config.tsdf.dynamic_threshold
        ))
        invalid_depth_pixels += int(np.count_nonzero(
            ~np.isfinite(depth_m) | (depth_m <= 0.0) |
            (depth_m > config.tsdf.depth_trunc_m)
        ))
        volume.integrate(color_rgb, depth_m, score_map, pose)
        integrated += 1

        packet = read_cache_frame(semantic_path)
        if (packet.frame_id != frame_id or
                _timestamp_key(packet.timestamp) != _timestamp_key(association.rgb_timestamp_lexeme)):
            raise ValueError(f"semantic packet identity mismatch at frame {frame_id}")
        for instance in packet.instances:
            track = track_by_instance.get((frame_id, instance.local_id))
            if track is None:
                raise ValueError(f"missing track row for frame {frame_id} instance {instance.local_id}")
            track_id = track.get("track_id")
            if (type(track_id) is not int or track_id in dynamic_track_ids or
                    float(track.get("score_map_probability", math.inf)) >=
                    config.static_score_max_exclusive):
                continue
            points = _backproject_mask_points(
                decode_binary_mask_rle(instance.mask_rle) &
                (score_map < config.static_score_max_exclusive),
                depth_m,
                intrinsics,
                pose,
                config.points_per_observation,
                config.tsdf.depth_trunc_m,
            )
            if points.shape[0]:
                observations.append(ObjectPointObservation(
                    track_id=canonical_track_id(track_id),
                    label=instance.label,
                    confidence=instance.score,
                    timestamp=float(association.rgb_timestamp_lexeme),
                    strong_dynamic=False,
                    points_world=points,
                ))
    if integrated == 0:
        raise ValueError("no frames have an exact valid pose")
    objects = aggregate_static_objects(observations, config.objects)
    dynamic_output_rows = [
        row for row in track_rows if row.get("track_id") in dynamic_track_ids
    ]
    producer_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    map_dir = Path(output_root) / str(selected.manifest["run_id"])
    manifest_base: dict[str, object] = {
        "schema": "ovorb.map.v1",
        "schema_version": 1,
        "study_id": config.study_id,
        "sequence_id": sequence_id,
        "seed": config.seed,
        "producer_commit": producer_commit,
        "command": command,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "valid": True,
        "pose_convention": "T_world_camera",
        "pose_timestamp_identity": "RGB timestamp rounded to six decimal places",
        "config": _read_json_object(config_path, "map configuration"),
        "inputs": {
            "config": _artifact_identity(config_path),
            "association": _artifact_identity(association_path),
            "dataset_manifest": _artifact_identity(dataset_manifest_path),
            "dataset_source_tree_sha256": source_tree_sha,
            "settings": _artifact_identity(settings_path),
            "selected_run_manifest": _artifact_identity(selected.manifest_path),
            "trajectory": _artifact_identity(selected.trajectory_path),
            "semantic_manifest": _artifact_identity(semantic_cache_root / "cache_manifest.json"),
            "semantic_index": _artifact_identity(semantic_cache_root / "cache_index.jsonl"),
            "semantic_completion": _artifact_identity(semantic_cache_root / "cache_complete.json"),
            "dynamic_manifest": _artifact_identity(dynamic_cache_root / "cache_manifest.json"),
            "dynamic_index": _artifact_identity(dynamic_cache_root / "cache_index.jsonl"),
            "dynamic_tracks": _artifact_identity(dynamic_cache_root / "dynamic_tracks.jsonl"),
            "dynamic_completion": _artifact_identity(dynamic_cache_root / "cache_complete.json"),
        },
        "counts": {
            "association_frames": len(associations),
            "trajectory_poses": len(poses),
            "integrated_frames": integrated,
            "excluded_frames": exclusion_counts,
            "dynamic_pixel_exclusions": dynamic_pixel_exclusions,
            "invalid_depth_pixels": invalid_depth_pixels,
            "object_observations": len(observations),
            "static_objects": len(objects),
            "dynamic_tracks": len(dynamic_track_ids),
        },
        "cache_completion": {
            "semantic": semantic_complete,
            "dynamic": dynamic_complete,
        },
        "semantic_cache_producer": semantic_manifest.get("producer_commit"),
        "dynamic_cache_producer": dynamic_manifest.get("producer_commit"),
    }
    export_map_artifacts(
        map_dir,
        volume=volume,
        objects=objects,
        dynamic_track_rows=dynamic_output_rows,
        manifest_base=manifest_base,
        screenshot_views=config.screenshot_views,
    )
    return map_dir


def _dataset_spec(project_root: Path, sequence_id: str) -> tuple[Path, Path, Path]:
    experiment = _read_json_object(
        project_root / "config/EXPERIMENT_MANIFEST.yaml", "experiment manifest"
    )
    matches = [item for item in experiment.get("datasets", [])
               if isinstance(item, dict) and item.get("id") == sequence_id]
    if len(matches) != 1:
        raise ValueError(f"unknown sequence: {sequence_id}")
    item = matches[0]
    archive = str(item["archive"])
    if not archive.endswith(".tgz"):
        raise ValueError("dataset archive must end in .tgz")
    return (
        project_root / "data/tum/raw" / archive[:-4],
        project_root / "data/tum/manifests" / f"{sequence_id}.json",
        project_root / "Examples/RGB-D" / str(item["settings"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--registry", type=Path, default=Path("runs/registry.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config/P07_MAP.json"))
    parser.add_argument("--semantic-cache-root", type=Path, default=Path("cache/semantic/v1"))
    parser.add_argument("--dynamic-cache-root", type=Path, default=Path("cache/dynamic/v1"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/maps"))
    parser.add_argument("--allow-dirty-for-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if not args.allow_dirty_for_test:
        _require_clean_tree(root)
    dataset_root, dataset_manifest, settings = _dataset_spec(root, args.sequence)
    output = build_sequence_map(
        project_root=root,
        sequence_id=args.sequence,
        registry_path=args.registry.resolve(),
        config_path=args.config.resolve(),
        dataset_root=dataset_root.resolve(),
        dataset_manifest_path=dataset_manifest.resolve(),
        settings_path=settings.resolve(),
        semantic_cache_root=(args.semantic_cache_root / args.sequence).resolve(),
        dynamic_cache_root=(args.dynamic_cache_root / args.sequence).resolve(),
        output_root=args.output_root.resolve(),
        command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    print(f"VALID map={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
