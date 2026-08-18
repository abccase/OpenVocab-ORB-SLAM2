"""Generate and validate immutable causal dynamic-score caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .association import AssociationObservation, associate_tracks
from .cache import read_cache_frame, validate_cache
from .config import (
    DynamicConfirmationConfig,
    FORMAL_BASELINE_COMPATIBILITY_COMMIT,
    FORMAL_BASELINE_PRODUCER_COMMIT,
)
from .geometry import centroid_from_mask
from .motion import DynamicTrack, TrackState
from .schemas import CacheManifest, decode_binary_mask_rle


_SHA256_FIELDS = (
    "semantic_manifest_sha256",
    "semantic_index_sha256",
    "semantic_identity_sha256",
    "dataset_manifest_sha256",
    "source_tree_sha256",
    "association_sha256",
    "bootstrap_trajectory_sha256",
    "bootstrap_run_manifest_sha256",
    "intrinsics_sha256",
    "dynamic_config_sha256",
)


@dataclass(frozen=True)
class DynamicCacheManifest:
    schema: str
    study_id: str
    sequence_id: str
    semantic_manifest_sha256: str
    semantic_index_sha256: str
    semantic_identity_sha256: str
    dataset_manifest_sha256: str
    source_tree_sha256: str
    association_sha256: str
    bootstrap_trajectory_sha256: str
    bootstrap_run_manifest_sha256: str
    intrinsics_sha256: str
    dynamic_config_sha256: str
    dynamic_config: dict[str, object]
    producer_commit: str
    expected_frame_count: int
    expected_instance_count: int
    diagnostic_frames: tuple[dict[str, object], ...]
    score_dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.schema or not self.study_id or not self.sequence_id:
            raise ValueError("dynamic cache identity fields must not be empty")
        for field in _SHA256_FIELDS:
            value = str(getattr(self, field))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field} must be a lowercase SHA256")
        if len(self.producer_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.producer_commit
        ):
            raise ValueError("producer_commit must be a lowercase 40-character Git SHA")
        if (
            self.expected_frame_count <= 0
            or self.expected_instance_count < 0
            or self.score_dtype != "float32"
        ):
            raise ValueError("dynamic cache dimensions or score dtype are invalid")
        if _sha256_json(self.dynamic_config) != self.dynamic_config_sha256:
            raise ValueError("dynamic config hash mismatch")

    def to_primitive(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "study_id": self.study_id,
            "sequence_id": self.sequence_id,
            "semantic_manifest_sha256": self.semantic_manifest_sha256,
            "semantic_index_sha256": self.semantic_index_sha256,
            "semantic_identity_sha256": self.semantic_identity_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "association_sha256": self.association_sha256,
            "bootstrap_trajectory_sha256": self.bootstrap_trajectory_sha256,
            "bootstrap_run_manifest_sha256": self.bootstrap_run_manifest_sha256,
            "intrinsics_sha256": self.intrinsics_sha256,
            "dynamic_config_sha256": self.dynamic_config_sha256,
            "dynamic_config": self.dynamic_config,
            "producer_commit": self.producer_commit,
            "expected_frame_count": self.expected_frame_count,
            "expected_instance_count": self.expected_instance_count,
            "diagnostic_frames": list(self.diagnostic_frames),
            "score_dtype": self.score_dtype,
        }

    @classmethod
    def from_primitive(cls, value: dict[str, Any]) -> "DynamicCacheManifest":
        required = {
            "schema", "study_id", "sequence_id", "semantic_manifest_sha256",
            "semantic_index_sha256", "semantic_identity_sha256",
            "dataset_manifest_sha256", "source_tree_sha256", "association_sha256",
            "bootstrap_trajectory_sha256", "bootstrap_run_manifest_sha256",
            "intrinsics_sha256", "dynamic_config_sha256", "dynamic_config",
            "producer_commit", "expected_frame_count", "expected_instance_count",
            "diagnostic_frames", "score_dtype",
        }
        if (
            set(value) != required
            or not isinstance(value["dynamic_config"], dict)
            or not isinstance(value["diagnostic_frames"], list)
        ):
            raise ValueError("dynamic cache manifest fields do not match schema")
        return cls(
            schema=str(value["schema"]),
            study_id=str(value["study_id"]),
            sequence_id=str(value["sequence_id"]),
            semantic_manifest_sha256=str(value["semantic_manifest_sha256"]),
            semantic_index_sha256=str(value["semantic_index_sha256"]),
            semantic_identity_sha256=str(value["semantic_identity_sha256"]),
            dataset_manifest_sha256=str(value["dataset_manifest_sha256"]),
            source_tree_sha256=str(value["source_tree_sha256"]),
            association_sha256=str(value["association_sha256"]),
            bootstrap_trajectory_sha256=str(value["bootstrap_trajectory_sha256"]),
            bootstrap_run_manifest_sha256=str(value["bootstrap_run_manifest_sha256"]),
            intrinsics_sha256=str(value["intrinsics_sha256"]),
            dynamic_config_sha256=str(value["dynamic_config_sha256"]),
            dynamic_config=dict(value["dynamic_config"]),
            producer_commit=str(value["producer_commit"]),
            expected_frame_count=int(value["expected_frame_count"]),
            expected_instance_count=int(value["expected_instance_count"]),
            diagnostic_frames=tuple(dict(item) for item in value["diagnostic_frames"]),
            score_dtype=str(value["score_dtype"]),
        )


@dataclass(frozen=True)
class DynamicCacheJob:
    project_root: Path
    sequence_id: str
    dataset_root: Path
    association_path: Path
    semantic_cache_root: Path
    trajectory_path: Path
    bootstrap_run_manifest_path: Path
    dataset_manifest_path: Path
    cache_root: Path
    intrinsics: np.ndarray
    config: DynamicConfirmationConfig
    manifest: DynamicCacheManifest


@dataclass(frozen=True)
class DynamicCacheValidation:
    valid: bool
    errors: tuple[str, ...]
    frame_count: int


@dataclass(frozen=True)
class DynamicCacheResult:
    cache_root: Path
    frame_index: tuple[dict[str, object], ...]
    track_rows: tuple[dict[str, object], ...]
    diagnostic_index: tuple[dict[str, object], ...]


def build_dynamic_job(
    project_root: Path,
    sequence_id: str,
    dataset_root: Path,
    semantic_cache_root: Path,
    trajectory_path: Path,
    bootstrap_run_manifest_path: Path,
    intrinsics: np.ndarray,
    *,
    producer_commit: str,
    dataset_manifest_path: Path,
    config: DynamicConfirmationConfig | None = None,
) -> DynamicCacheJob:
    root = Path(project_root).resolve()
    dataset = Path(dataset_root).resolve()
    semantic_root = Path(semantic_cache_root).resolve()
    trajectory = Path(trajectory_path).resolve()
    run_manifest_path = Path(bootstrap_run_manifest_path).resolve()
    dataset_manifest = Path(dataset_manifest_path).resolve()
    association = dataset / "associate.txt"
    cfg = config or DynamicConfirmationConfig.frozen()
    matrix = _validate_intrinsics(intrinsics)
    semantic_manifest_path = semantic_root / "cache_manifest.json"
    semantic_index_path = semantic_root / "cache_index.jsonl"
    semantic_manifest = CacheManifest.from_primitive(_load_json(semantic_manifest_path))
    validation = validate_cache(semantic_root, semantic_manifest)
    if not validation.valid:
        raise ValueError("semantic cache is invalid: " + "; ".join(validation.errors))
    if semantic_manifest.sequence_id != sequence_id:
        raise ValueError("semantic cache sequence mismatch")
    source_tree_sha256 = hash_dataset_tree(dataset)
    association_sha256 = _sha256_file(association)
    if semantic_manifest.source_tree_sha256 != source_tree_sha256:
        raise ValueError("semantic cache source tree mismatch")
    if semantic_manifest.association_sha256 != association_sha256:
        raise ValueError("semantic cache association mismatch")
    rows = _read_association(association)
    if len(rows) != semantic_manifest.expected_frame_count:
        raise ValueError("association count does not match semantic cache")
    dataset_value = _load_json(dataset_manifest)
    _validate_dataset_manifest(
        dataset_value,
        sequence_id,
        source_tree_sha256,
        association_sha256,
        len(rows),
    )
    run_manifest = _load_json(run_manifest_path)
    if run_manifest.get("state") != "COMPLETED" or run_manifest.get("valid") is not True:
        raise ValueError("bootstrap run manifest is not valid and complete")
    trajectory_poses = _read_trajectory(trajectory)
    _validate_bootstrap_manifest(
        run_manifest,
        sequence_id,
        trajectory,
        association_sha256,
        _sha256_file(dataset_manifest),
        len(rows),
        len(trajectory_poses),
    )
    config_value = cfg.to_primitive()
    semantic_entries = _read_jsonl(semantic_index_path)
    semantic_identity = _semantic_identity_rows(semantic_root, semantic_entries)
    _validate_semantic_dimensions(dataset_value, semantic_identity)
    expected_instance_count = sum(len(row["instances"]) for row in semantic_identity)
    diagnostic_frames = _select_diagnostic_frames(
        rows,
        semantic_identity,
        cfg.diagnostic_fractions,
    )
    manifest = DynamicCacheManifest(
        schema=cfg.schema,
        study_id=semantic_manifest.study_id,
        sequence_id=sequence_id,
        semantic_manifest_sha256=_sha256_file(semantic_manifest_path),
        semantic_index_sha256=_sha256_file(semantic_index_path),
        semantic_identity_sha256=_sha256_jsonl(semantic_identity),
        dataset_manifest_sha256=_sha256_file(dataset_manifest),
        source_tree_sha256=source_tree_sha256,
        association_sha256=association_sha256,
        bootstrap_trajectory_sha256=_sha256_file(trajectory),
        bootstrap_run_manifest_sha256=_sha256_file(run_manifest_path),
        intrinsics_sha256=_sha256_json(matrix.tolist()),
        dynamic_config_sha256=cfg.sha256(),
        dynamic_config=config_value,
        producer_commit=producer_commit,
        expected_frame_count=len(rows),
        expected_instance_count=expected_instance_count,
        diagnostic_frames=diagnostic_frames,
    )
    return DynamicCacheJob(
        project_root=root,
        sequence_id=sequence_id,
        dataset_root=dataset,
        association_path=association,
        semantic_cache_root=semantic_root,
        trajectory_path=trajectory,
        bootstrap_run_manifest_path=run_manifest_path,
        dataset_manifest_path=dataset_manifest,
        cache_root=root / "cache/dynamic/v1" / sequence_id,
        intrinsics=matrix,
        config=cfg,
        manifest=manifest,
    )


def generate_dynamic_cache(job: DynamicCacheJob) -> DynamicCacheResult:
    """Generate frames in timestamp order; frame ``t`` reads no later input."""
    _validate_bound_inputs(job)
    cache_root = job.cache_root
    manifest_path = cache_root / "cache_manifest.json"
    if manifest_path.exists():
        observed = DynamicCacheManifest.from_primitive(_load_json(manifest_path))
        if observed != job.manifest:
            raise ValueError("dynamic cache manifest identity mismatch")
    else:
        _write_json_atomic(manifest_path, job.manifest.to_primitive())

    semantic_index = _read_jsonl(job.semantic_cache_root / "cache_index.jsonl")
    semantic_identity = _semantic_identity_rows(job.semantic_cache_root, semantic_index)
    if _sha256_jsonl(semantic_identity) != job.manifest.semantic_identity_sha256:
        raise ValueError("semantic identity hash mismatch")
    semantic_identity_path = cache_root / "semantic_identity.jsonl"
    if semantic_identity_path.exists():
        if _sha256_file(semantic_identity_path) != job.manifest.semantic_identity_sha256:
            raise ValueError("existing semantic identity mismatch")
    else:
        _write_jsonl_atomic(semantic_identity_path, semantic_identity)

    complete_path = cache_root / "cache_complete.json"
    if complete_path.exists():
        validation = validate_dynamic_cache(
            cache_root,
            job.manifest,
            job.dataset_root,
        )
        if not validation.valid:
            raise ValueError("existing complete dynamic cache is invalid: " + "; ".join(validation.errors))
        return _load_result(cache_root)

    existing_index = _read_jsonl(cache_root / "cache_index.jsonl")
    if [int(row.get("frame_id", -1)) for row in existing_index] != list(range(len(existing_index))):
        raise ValueError("partial dynamic cache index is not a contiguous prefix")
    if len(existing_index) > job.manifest.expected_frame_count:
        raise ValueError("partial dynamic cache has extra frames")

    associations = _read_association(job.association_path)
    if len(semantic_index) != len(associations):
        raise ValueError("semantic cache index coverage mismatch")
    poses = _read_trajectory(job.trajectory_path)
    active_tracks: list[DynamicTrack] = []
    next_track_id = 0
    track_rows: list[dict[str, object]] = []
    frame_index = list(existing_index)
    diagnostic_index: list[dict[str, object]] = []
    diagnostic_by_frame = {
        int(item["frame_id"]): item for item in job.manifest.diagnostic_frames
    }

    for frame_id, association_row in enumerate(associations):
        timestamp, rgb_relative, depth_timestamp, depth_relative = association_row
        semantic_entry = semantic_index[frame_id]
        if (
            int(semantic_entry.get("frame_id", -1)) != frame_id
            or float(semantic_entry.get("timestamp", math.nan)) != timestamp
        ):
            raise ValueError("semantic cache index is not aligned with associations")
        packet_path = job.semantic_cache_root / str(semantic_entry["path"])
        packet = read_cache_frame(packet_path)
        if packet.frame_id != frame_id or packet.timestamp != timestamp:
            raise ValueError("semantic packet is not aligned with associations")
        if packet.source_image_sha256 != _sha256_file(job.dataset_root / rgb_relative):
            raise ValueError("semantic packet source image hash mismatch")
        masks = [decode_binary_mask_rle(instance.mask_rle) for instance in packet.instances]
        score_map = np.zeros((packet.image_height, packet.image_width), dtype=np.float32)
        pose = poses.get(timestamp)
        unknown_reason = None
        if depth_timestamp > timestamp:
            unknown_reason = "FUTURE_DEPTH_TIMESTAMP"
        elif pose is None:
            unknown_reason = "MISSING_EXACT_BOOTSTRAP_POSE"

        if unknown_reason is not None:
            for track in [item for item in active_tracks if not item.terminated]:
                track.mark_missed(timestamp, job.config)
            for instance, mask in zip(packet.instances, masks):
                score_map[mask] = np.maximum(
                    score_map[mask], np.float32(job.config.unknown_dynamic_probability)
                )
                track_rows.append(
                    _unknown_row(frame_id, timestamp, instance.local_id, instance.label,
                                 job.config.unknown_dynamic_probability,
                                 unknown_reason)
                )
        else:
            depth = cv2.imread(str(job.dataset_root / depth_relative), cv2.IMREAD_UNCHANGED)
            if (
                depth is None
                or depth.ndim != 2
                or depth.shape != (packet.image_height, packet.image_width)
            ):
                raise ValueError(f"invalid depth image for frame {frame_id}")
            depth_m = np.asarray(depth, dtype=np.float64) / job.config.depth_scale
            current_tracks = [item for item in active_tracks if not item.terminated]
            for track in current_tracks:
                track.predict(timestamp, job.config)
            observations: list[AssociationObservation] = []
            geometry_by_observation: list[tuple[np.ndarray, np.ndarray]] = []
            invalid_reasons: dict[int, str] = {}
            for instance, mask in zip(packet.instances, masks):
                geometry = centroid_from_mask(
                    mask,
                    depth_m,
                    job.intrinsics,
                    pose,
                    min_valid_depth_pixels=job.config.min_valid_depth_pixels,
                    diagnostic_sample_limit=job.config.diagnostic_sample_limit,
                )
                if not geometry.valid:
                    invalid_reasons[instance.local_id] = str(geometry.reason)
                    continue
                observations.append(
                    AssociationObservation(
                        instance.local_id,
                        instance.label,
                        np.asarray(geometry.centroid_world, dtype=np.float64),
                        mask,
                    )
                )
                geometry_by_observation.append(
                    (
                        np.asarray(geometry.centroid_world, dtype=np.float64),
                        np.asarray(geometry.mad_world, dtype=np.float64),
                    )
                )
            association = associate_tracks(current_tracks, observations, job.config)
            states: dict[int, tuple[int, TrackState, str, np.ndarray, np.ndarray]] = {}
            for track_id, observation_index in association.assignments.items():
                track = next(item for item in current_tracks if item.track_id == track_id)
                observation = observations[observation_index]
                centroid, mad = geometry_by_observation[observation_index]
                state = track.update(centroid, timestamp=timestamp, mad=mad, config=job.config)
                track.last_mask = observation.mask.copy()
                states[observation.local_id] = (track.track_id, state, "TRACK_UPDATED", centroid, mad)
            unassigned_track_ids = set(association.unassigned_tracks)
            for track in current_tracks:
                if track.track_id in unassigned_track_ids:
                    track.mark_missed(timestamp, job.config)
            for observation_index in association.unassigned_observations:
                observation = observations[observation_index]
                centroid, mad = geometry_by_observation[observation_index]
                track = DynamicTrack.new(
                    next_track_id,
                    observation.label,
                    centroid,
                    timestamp=timestamp,
                    config=job.config,
                )
                next_track_id += 1
                track.last_mask = observation.mask.copy()
                active_tracks.append(track)
                state = TrackState(track.track_id, 0.0, False, 1, 0, "NEW_TRACK")
                states[observation.local_id] = (track.track_id, state, "NEW_TRACK", centroid, mad)

            for instance, mask in zip(packet.instances, masks):
                if instance.local_id in invalid_reasons:
                    probability = job.config.unknown_dynamic_probability
                    row = _unknown_row(
                        frame_id, timestamp, instance.local_id, instance.label,
                        probability, invalid_reasons[instance.local_id],
                    )
                else:
                    track_id, state, reason, centroid, mad = states[instance.local_id]
                    probability = _score_probability(state, job.config)
                    row = {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "local_id": instance.local_id,
                        "track_id": track_id,
                        "label": instance.label,
                        "centroid_world": centroid.tolist(),
                        "mad_world": mad.tolist(),
                        "dynamic_probability": state.dynamic_probability,
                        "score_map_probability": probability,
                        "strong_dynamic": state.strong_dynamic,
                        "observation_count": state.observation_count,
                        "confirming_observations": state.confirming_observations,
                        "reason": reason,
                    }
                score_map[mask] = np.maximum(score_map[mask], np.float32(probability))
                track_rows.append(row)

        encoded = _encode_npy(score_map)
        digest = hashlib.sha256(encoded).hexdigest()
        relative = Path("score_maps") / f"{frame_id:06d}.npy"
        entry: dict[str, object] = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "path": relative.as_posix(),
            "sha256": digest,
            "semantic_packet_sha256": str(semantic_entry["sha256"]),
            "height": packet.image_height,
            "width": packet.image_width,
            "dtype": "float32",
        }
        target = cache_root / relative
        if frame_id < len(existing_index):
            if existing_index[frame_id] != entry or _sha256_file(target) != digest:
                raise ValueError(f"existing dynamic frame mismatch: {frame_id}")
        else:
            if target.exists():
                if _sha256_file(target) != digest:
                    raise ValueError(f"orphan dynamic frame mismatch: {frame_id}")
            else:
                _write_bytes_atomic(target, encoded)
            frame_index.append(entry)
            _write_jsonl_atomic(cache_root / "cache_index.jsonl", frame_index)

        diagnostic = diagnostic_by_frame.get(frame_id)
        if diagnostic is not None:
            diagnostic_relative = Path(str(diagnostic["path"]))
            diagnostic_bytes = _render_diagnostic_overlay(
                job.dataset_root / str(diagnostic["source_path"]),
                score_map,
            )
            diagnostic_digest = hashlib.sha256(diagnostic_bytes).hexdigest()
            diagnostic_entry: dict[str, object] = {
                "fraction": diagnostic["fraction"],
                "frame_id": frame_id,
                "timestamp": timestamp,
                "path": diagnostic_relative.as_posix(),
                "sha256": diagnostic_digest,
                "source_image_sha256": packet.source_image_sha256,
                "score_map_sha256": digest,
                "width": packet.image_width,
                "height": packet.image_height,
            }
            diagnostic_target = cache_root / diagnostic_relative
            if diagnostic_target.exists():
                if _sha256_file(diagnostic_target) != diagnostic_digest:
                    raise ValueError(f"existing diagnostic overlay mismatch: {frame_id}")
            else:
                _write_bytes_atomic(diagnostic_target, diagnostic_bytes)
            diagnostic_index.append(diagnostic_entry)

    _write_jsonl_atomic(cache_root / "dynamic_tracks.jsonl", track_rows)
    _write_jsonl_atomic(cache_root / "diagnostics_index.jsonl", diagnostic_index)
    validation = validate_dynamic_cache(
        cache_root,
        job.manifest,
        job.dataset_root,
        require_complete=False,
    )
    if not validation.valid:
        raise ValueError("generated dynamic cache is invalid: " + "; ".join(validation.errors))
    _write_json_atomic(
        complete_path,
        {
            "manifest_sha256": _sha256_file(manifest_path),
            "index_sha256": _sha256_file(cache_root / "cache_index.jsonl"),
            "tracks_sha256": _sha256_file(cache_root / "dynamic_tracks.jsonl"),
            "semantic_identity_sha256": _sha256_file(semantic_identity_path),
            "diagnostics_index_sha256": _sha256_file(
                cache_root / "diagnostics_index.jsonl"
            ),
            "frame_count": validation.frame_count,
        },
    )
    final_validation = validate_dynamic_cache(
        cache_root,
        job.manifest,
        job.dataset_root,
    )
    if not final_validation.valid:
        raise ValueError("completed dynamic cache is invalid: " + "; ".join(final_validation.errors))
    return DynamicCacheResult(
        cache_root,
        tuple(frame_index),
        tuple(track_rows),
        tuple(diagnostic_index),
    )


def validate_dynamic_cache(
    cache_root: Path,
    expected: DynamicCacheManifest,
    dataset_root: Path,
    *,
    require_complete: bool = True,
) -> DynamicCacheValidation:
    root = Path(cache_root)
    source_root = Path(dataset_root)
    errors: list[str] = []
    manifest_path = root / "cache_manifest.json"
    try:
        observed = DynamicCacheManifest.from_primitive(_load_json(manifest_path))
        if observed != expected:
            errors.append("manifest identity mismatch")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid dynamic cache manifest: {exc}")
    identity_path = root / "semantic_identity.jsonl"
    semantic_identity: list[dict[str, Any]] = []
    try:
        semantic_identity = _read_jsonl(identity_path)
        if _sha256_file(identity_path) != expected.semantic_identity_sha256:
            raise ValueError("semantic identity hash mismatch")
        if len(semantic_identity) != expected.expected_frame_count:
            raise ValueError("semantic identity frame coverage mismatch")
        for frame_id, row in enumerate(semantic_identity):
            if set(row) != {
                "frame_id", "timestamp", "semantic_packet_sha256",
                "source_image_sha256", "width", "height", "instances",
            }:
                raise ValueError("semantic identity fields mismatch")
            if int(row["frame_id"]) != frame_id or not isinstance(row["instances"], list):
                raise ValueError("semantic identity ordering mismatch")
            local_ids = [int(item["local_id"]) for item in row["instances"]]
            if local_ids != list(range(len(local_ids))):
                raise ValueError("semantic instance local IDs mismatch")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid semantic identity: {exc}")
    try:
        index = _read_jsonl(root / "cache_index.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return DynamicCacheValidation(False, tuple(errors + [f"invalid dynamic cache index: {exc}"]), 0)
    referenced: set[Path] = set()
    timestamps: set[float] = set()
    score_maps: dict[int, np.ndarray] = {}
    for expected_frame_id, entry in enumerate(index):
        try:
            if set(entry) != {
                "frame_id", "timestamp", "path", "sha256",
                "semantic_packet_sha256", "height", "width", "dtype",
            }:
                raise ValueError("index fields mismatch")
            if int(entry["frame_id"]) != expected_frame_id:
                raise ValueError("frame IDs are not contiguous")
            timestamp = float(entry["timestamp"])
            if not math.isfinite(timestamp) or timestamp in timestamps:
                raise ValueError("timestamp is invalid or duplicate")
            timestamps.add(timestamp)
            if expected_frame_id >= len(semantic_identity):
                raise ValueError("missing semantic identity")
            identity = semantic_identity[expected_frame_id]
            if (
                timestamp != float(identity["timestamp"])
                or entry["semantic_packet_sha256"]
                != identity["semantic_packet_sha256"]
                or int(entry["height"]) != int(identity["height"])
                or int(entry["width"]) != int(identity["width"])
            ):
                raise ValueError("semantic identity mismatch")
            relative = Path(str(entry["path"]))
            if relative.is_absolute() or ".." in relative.parts or relative.parent != Path("score_maps"):
                raise ValueError("unsafe score-map path")
            path = root / relative
            referenced.add(path)
            if _sha256_file(path) != entry["sha256"]:
                raise ValueError("score-map hash mismatch")
            with path.open("rb") as stream:
                score_map = np.load(stream, allow_pickle=False)
            shape = (int(entry["height"]), int(entry["width"]))
            if score_map.shape != shape or score_map.dtype != np.float32 or entry["dtype"] != "float32":
                raise ValueError("score-map shape or dtype mismatch")
            if not np.all(np.isfinite(score_map)) or np.any(score_map < 0.0) or np.any(score_map > 1.0):
                raise ValueError("score-map values are outside [0, 1]")
            score_maps[expected_frame_id] = score_map
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"invalid dynamic index entry {expected_frame_id}: {exc}")
    if len(index) != expected.expected_frame_count:
        errors.append(f"frame coverage mismatch: {len(index)} != {expected.expected_frame_count}")
    observed_maps = set((root / "score_maps").glob("*.npy")) if (root / "score_maps").is_dir() else set()
    for extra in sorted(observed_maps - referenced):
        errors.append(f"extra score map: {extra.name}")
    tracks_path = root / "dynamic_tracks.jsonl"
    try:
        rows = _read_jsonl(tracks_path)
        expected_instances = {
            (int(frame["frame_id"]), int(instance["local_id"])): (
                float(frame["timestamp"]),
                str(instance["label"]),
            )
            for frame in semantic_identity
            for instance in frame["instances"]
        }
        observed_instances: set[tuple[int, int]] = set()
        for row in rows:
            if set(row) != {
                "frame_id", "timestamp", "local_id", "track_id", "label",
                "centroid_world", "mad_world", "dynamic_probability",
                "score_map_probability", "strong_dynamic", "observation_count",
                "confirming_observations", "reason",
            }:
                raise ValueError("track row fields mismatch")
            frame_id = int(row["frame_id"])
            local_id = int(row["local_id"])
            identity = (frame_id, local_id)
            if identity in observed_instances or identity not in expected_instances:
                raise ValueError("duplicate or unknown instance identity")
            observed_instances.add(identity)
            expected_timestamp, expected_label = expected_instances[identity]
            if (
                float(row["timestamp"]) != expected_timestamp
                or str(row["label"]) != expected_label
            ):
                raise ValueError("instance identity mismatch")
            probability = float(row["score_map_probability"])
            if frame_id < 0 or frame_id >= expected.expected_frame_count:
                raise ValueError("track row frame is outside coverage")
            exit_threshold = float(expected.dynamic_config["dynamic_exit_threshold"])
            if (
                not 0.0 <= probability <= 1.0
                or bool(row["strong_dynamic"]) and probability < exit_threshold
            ):
                raise ValueError("track row dynamic state is invalid")
        if observed_instances != set(expected_instances):
            raise ValueError("missing instance identity")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid dynamic tracks: {exc}")
    diagnostics_path = root / "diagnostics_index.jsonl"
    referenced_diagnostics: set[Path] = set()
    try:
        diagnostic_rows = _read_jsonl(diagnostics_path)
        if len(diagnostic_rows) != len(expected.diagnostic_frames):
            raise ValueError("diagnostic coverage mismatch")
        score_by_frame = {int(row["frame_id"]): row for row in index}
        for declared, row in zip(expected.diagnostic_frames, diagnostic_rows):
            frame_id = int(declared["frame_id"])
            if set(row) != {
                "fraction", "frame_id", "timestamp", "path", "sha256",
                "source_image_sha256", "score_map_sha256", "width", "height",
            }:
                raise ValueError("diagnostic fields mismatch")
            if (
                row["fraction"] != declared["fraction"]
                or int(row["frame_id"]) != frame_id
                or float(row["timestamp"]) != float(declared["timestamp"])
                or row["path"] != declared["path"]
                or row["source_image_sha256"] != declared["source_image_sha256"]
                or row["score_map_sha256"] != score_by_frame[frame_id]["sha256"]
            ):
                raise ValueError("diagnostic identity mismatch")
            relative = Path(str(row["path"]))
            if relative.is_absolute() or ".." in relative.parts or relative.parent != Path("diagnostics"):
                raise ValueError("unsafe diagnostic path")
            source_relative = Path(str(declared["source_path"]))
            if (
                source_relative.is_absolute()
                or ".." in source_relative.parts
                or source_relative.parts[:1] != ("rgb",)
            ):
                raise ValueError("unsafe diagnostic source path")
            source_path = source_root / source_relative
            if _sha256_file(source_path) != declared["source_image_sha256"]:
                raise ValueError("diagnostic source image hash mismatch")
            expected_bytes = _render_diagnostic_overlay(
                source_path,
                score_maps[frame_id],
            )
            expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
            if row["sha256"] != expected_sha256:
                raise ValueError("diagnostic derived-content hash mismatch")
            path = root / relative
            referenced_diagnostics.add(path)
            observed_bytes = path.read_bytes()
            if observed_bytes != expected_bytes:
                raise ValueError("diagnostic derived-content mismatch")
            if hashlib.sha256(observed_bytes).hexdigest() != row["sha256"]:
                raise ValueError("diagnostic hash mismatch")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (int(row["height"]), int(row["width"])):
                raise ValueError("diagnostic dimensions mismatch")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid diagnostics: {exc}")
    observed_diagnostics = (
        set((root / "diagnostics").glob("*.png"))
        if (root / "diagnostics").is_dir()
        else set()
    )
    for extra in sorted(observed_diagnostics - referenced_diagnostics):
        errors.append(f"extra diagnostic overlay: {extra.name}")
    if require_complete:
        try:
            complete = _load_json(root / "cache_complete.json")
            expected_complete = {
                "manifest_sha256": _sha256_file(manifest_path),
                "index_sha256": _sha256_file(root / "cache_index.jsonl"),
                "tracks_sha256": _sha256_file(tracks_path),
                "semantic_identity_sha256": _sha256_file(identity_path),
                "diagnostics_index_sha256": _sha256_file(diagnostics_path),
                "frame_count": len(index),
            }
            if complete != expected_complete:
                errors.append("cache completion identity mismatch")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid cache completion marker: {exc}")
    return DynamicCacheValidation(not errors, tuple(errors), len(index))


def _unknown_row(
    frame_id: int,
    timestamp: float,
    local_id: int,
    label: str,
    probability: float,
    reason: str,
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "local_id": local_id,
        "track_id": None,
        "label": label,
        "centroid_world": None,
        "mad_world": None,
        "dynamic_probability": probability,
        "score_map_probability": probability,
        "strong_dynamic": False,
        "observation_count": 0,
        "confirming_observations": 0,
        "reason": reason,
    }


def _score_probability(state: TrackState, config: DynamicConfirmationConfig) -> float:
    if state.strong_dynamic:
        return state.dynamic_probability
    if (
        state.observation_count < config.min_confirming_observations
        or state.confirming_observations > 0
    ):
        return config.unknown_dynamic_probability
    return state.dynamic_probability


def _validate_bound_inputs(job: DynamicCacheJob) -> None:
    checks = {
        "semantic manifest": (job.semantic_cache_root / "cache_manifest.json", job.manifest.semantic_manifest_sha256),
        "semantic index": (job.semantic_cache_root / "cache_index.jsonl", job.manifest.semantic_index_sha256),
        "association": (job.association_path, job.manifest.association_sha256),
        "bootstrap trajectory": (job.trajectory_path, job.manifest.bootstrap_trajectory_sha256),
        "bootstrap run manifest": (job.bootstrap_run_manifest_path, job.manifest.bootstrap_run_manifest_sha256),
        "dataset manifest": (
            job.dataset_manifest_path,
            job.manifest.dataset_manifest_sha256,
        ),
    }
    for name, (path, expected) in checks.items():
        if _sha256_file(path) != expected:
            raise ValueError(f"{name} hash mismatch")
    if hash_dataset_tree(job.dataset_root) != job.manifest.source_tree_sha256:
        raise ValueError("dataset source tree hash mismatch")
    if _sha256_json(job.intrinsics.tolist()) != job.manifest.intrinsics_sha256:
        raise ValueError("intrinsics hash mismatch")
    if job.config.sha256() != job.manifest.dynamic_config_sha256:
        raise ValueError("dynamic config hash mismatch")
    semantic_manifest = CacheManifest.from_primitive(_load_json(job.semantic_cache_root / "cache_manifest.json"))
    semantic_validation = validate_cache(job.semantic_cache_root, semantic_manifest)
    if not semantic_validation.valid:
        raise ValueError("semantic cache is invalid: " + "; ".join(semantic_validation.errors))
    semantic_entries = _read_jsonl(job.semantic_cache_root / "cache_index.jsonl")
    semantic_identity = _semantic_identity_rows(job.semantic_cache_root, semantic_entries)
    if _sha256_jsonl(semantic_identity) != job.manifest.semantic_identity_sha256:
        raise ValueError("semantic identity hash mismatch")
    associations = _read_association(job.association_path)
    _validate_dataset_manifest(
        dataset_value := _load_json(job.dataset_manifest_path),
        job.sequence_id,
        job.manifest.source_tree_sha256,
        job.manifest.association_sha256,
        len(associations),
    )
    _validate_semantic_dimensions(dataset_value, semantic_identity)
    run_manifest = _load_json(job.bootstrap_run_manifest_path)
    if run_manifest.get("state") != "COMPLETED" or run_manifest.get("valid") is not True:
        raise ValueError("bootstrap run manifest is not valid and complete")
    poses = _read_trajectory(job.trajectory_path)
    _validate_bootstrap_manifest(
        run_manifest,
        job.sequence_id,
        job.trajectory_path,
        job.manifest.association_sha256,
        job.manifest.dataset_manifest_sha256,
        len(associations),
        len(poses),
    )


def _validate_bootstrap_manifest(
    manifest: dict[str, Any],
    sequence_id: str,
    trajectory_path: Path,
    association_sha256: str,
    dataset_manifest_sha256: str,
    expected_frames: int,
    pose_count: int,
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("bootstrap run schema mismatch")
    if manifest.get("study") != "oracle":
        raise ValueError("bootstrap run study mismatch")
    if manifest.get("mode") != "baseline":
        raise ValueError("bootstrap run mode mismatch")
    if manifest.get("compatibility_commit") != FORMAL_BASELINE_COMPATIBILITY_COMMIT:
        raise ValueError("bootstrap run compatibility commit mismatch")
    if manifest.get("producer_commit") != FORMAL_BASELINE_PRODUCER_COMMIT:
        raise ValueError("bootstrap run producer commit mismatch")
    if manifest.get("exit_code") != 0 or manifest.get("invalid_reason") is not None:
        raise ValueError("bootstrap run completion identity mismatch")
    if manifest.get("sequence_id") != sequence_id:
        raise ValueError("bootstrap run sequence mismatch")
    if int(manifest.get("seed", -1)) != 23011:
        raise ValueError("bootstrap run must use frozen seed 23011")
    expected_run_id = f"oracle-{sequence_id}-seed-23011-attempt-001"
    if manifest.get("run_id") != expected_run_id:
        raise ValueError("bootstrap run identity mismatch")
    if manifest.get("association_sha256") != association_sha256:
        raise ValueError("bootstrap run association mismatch")
    if manifest.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("bootstrap run dataset manifest mismatch")
    if (
        int(manifest.get("expected_frames", -1)) != expected_frames
        or int(manifest.get("frame_count", -1)) != expected_frames
    ):
        raise ValueError("bootstrap run frame count mismatch")
    trajectory = manifest.get("trajectory")
    if (
        not isinstance(trajectory, dict)
        or trajectory.get("path") != "CameraTrajectory.txt"
        or int(trajectory.get("pose_count", -1)) != pose_count
        or trajectory.get("sha256") != _sha256_file(trajectory_path)
    ):
        raise ValueError("bootstrap run trajectory hash mismatch")


def _validate_dataset_manifest(
    manifest: dict[str, Any],
    sequence_id: str,
    source_tree_sha256: str,
    association_sha256: str,
    expected_frames: int,
) -> None:
    counts = manifest.get("counts")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("sequence_id") != sequence_id
        or manifest.get("validation_status") != "VALID"
        or manifest.get("extracted_tree_sha256") != source_tree_sha256
        or manifest.get("association_sha256") != association_sha256
        or not isinstance(counts, dict)
        or int(counts.get("associations", -1)) != expected_frames
    ):
        raise ValueError("dataset manifest identity mismatch")


def _validate_semantic_dimensions(
    dataset_manifest: dict[str, Any],
    semantic_identity: list[dict[str, object]],
) -> None:
    dimensions = dataset_manifest.get("image_dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("dataset image dimensions are missing")
    expected = (int(dimensions.get("width", -1)), int(dimensions.get("height", -1)))
    if expected[0] <= 0 or expected[1] <= 0 or any(
        (int(row["width"]), int(row["height"])) != expected
        for row in semantic_identity
    ):
        raise ValueError("semantic frame dimensions do not match dataset manifest")


def _semantic_identity_rows(
    semantic_root: Path,
    entries: list[dict[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in entries:
        packet = read_cache_frame(semantic_root / str(entry["path"]))
        rows.append(
            {
                "frame_id": packet.frame_id,
                "timestamp": packet.timestamp,
                "semantic_packet_sha256": str(entry["sha256"]),
                "source_image_sha256": packet.source_image_sha256,
                "width": packet.image_width,
                "height": packet.image_height,
                "instances": [
                    {"local_id": instance.local_id, "label": instance.label}
                    for instance in packet.instances
                ],
            }
        )
    return rows


def _select_diagnostic_frames(
    associations: list[tuple[float, Path, float, Path]],
    semantic_identity: list[dict[str, object]],
    fractions: tuple[float, float, float],
) -> tuple[dict[str, object], ...]:
    if len(associations) < 3:
        raise ValueError("dynamic cache requires at least three diagnostic frames")
    timestamps = [row[0] for row in associations]
    start, end = timestamps[0], timestamps[-1]
    selected: list[dict[str, object]] = []
    used: set[int] = set()
    for fraction in fractions:
        target = start + fraction * (end - start)
        frame_id = min(
            (index for index in range(len(timestamps)) if index not in used),
            key=lambda index: (abs(timestamps[index] - target), index),
        )
        used.add(frame_id)
        identity = semantic_identity[frame_id]
        selected.append(
            {
                "fraction": fraction,
                "frame_id": frame_id,
                "timestamp": timestamps[frame_id],
                "source_image_sha256": identity["source_image_sha256"],
                "source_path": associations[frame_id][1].as_posix(),
                "path": f"diagnostics/frame-{frame_id:06d}.png",
            }
        )
    return tuple(selected)


def _validate_intrinsics(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or not np.allclose(matrix[2], [0.0, 0.0, 1.0]):
        raise ValueError("intrinsics must use the supported pinhole convention")
    return matrix.copy()


def _read_association(path: Path) -> list[tuple[float, Path, float, Path]]:
    rows: list[tuple[float, Path, float, Path]] = []
    previous = -math.inf
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: association row must have four fields")
        rgb_timestamp, depth_timestamp = float(fields[0]), float(fields[2])
        if not all(math.isfinite(value) for value in (rgb_timestamp, depth_timestamp)) or rgb_timestamp <= previous:
            raise ValueError(f"{path}:{line_number}: invalid or non-monotonic timestamp")
        rgb, depth = Path(fields[1]), Path(fields[3])
        if (
            rgb.is_absolute() or ".." in rgb.parts or rgb.parts[:1] != ("rgb",)
            or depth.is_absolute() or ".." in depth.parts or depth.parts[:1] != ("depth",)
        ):
            raise ValueError(f"{path}:{line_number}: unsafe associated path")
        rows.append((rgb_timestamp, rgb, depth_timestamp, depth))
        previous = rgb_timestamp
    if not rows:
        raise ValueError("association has no frames")
    return rows


def _read_trajectory(path: Path) -> dict[float, np.ndarray]:
    poses: dict[float, np.ndarray] = {}
    previous = -math.inf
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"{path}:{line_number}: trajectory row must have eight fields")
        values = tuple(float(field) for field in fields)
        if not all(math.isfinite(value) for value in values) or values[0] <= previous:
            raise ValueError(f"{path}:{line_number}: invalid or non-monotonic trajectory")
        timestamp, tx, ty, tz, qx, qy, qz, qw = values
        quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise ValueError(f"{path}:{line_number}: zero quaternion")
        qx, qy, qz, qw = quaternion / norm
        rotation = np.array(
            [
                [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
                [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
                [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = [tx, ty, tz]
        poses[timestamp] = pose
        previous = timestamp
    if not poses:
        raise ValueError("bootstrap trajectory has no poses")
    return poses


def hash_dataset_tree(root: Path) -> str:
    base = Path(root)
    digest = hashlib.sha256()
    for path in sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix()):
        if path.is_symlink():
            raise ValueError(f"dataset tree contains symlink: {path}")
        if not path.is_file() or path.name == "associate.txt":
            continue
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _jsonl_bytes(values: list[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for value in values
    )


def _sha256_jsonl(values: list[dict[str, object]]) -> str:
    return hashlib.sha256(_jsonl_bytes(values)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL row")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
        rows.append(value)
    return rows


def _encode_npy(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _render_diagnostic_overlay(rgb_path: Path, score_map: np.ndarray) -> bytes:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape[:2] != score_map.shape:
        raise ValueError(f"invalid diagnostic RGB image: {rgb_path}")
    level = np.rint(np.asarray(score_map, dtype=np.float64) * 255.0).astype(np.uint8)
    heat = np.zeros_like(rgb)
    heat[..., 1] = 255 - level
    heat[..., 2] = level
    overlay = rgb.copy()
    active = level > 0
    overlay[active] = (
        (rgb[active].astype(np.uint16) + heat[active].astype(np.uint16)) // 2
    ).astype(np.uint8)
    ok, encoded = cv2.imencode(
        ".png",
        overlay,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    if not ok:
        raise ValueError("unable to encode diagnostic overlay")
    return encoded.tobytes()


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial"
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _write_bytes_atomic(path, payload)


def _write_jsonl_atomic(path: Path, values: list[dict[str, object]]) -> None:
    _write_bytes_atomic(path, _jsonl_bytes(values))


def _load_result(cache_root: Path) -> DynamicCacheResult:
    return DynamicCacheResult(
        Path(cache_root),
        tuple(_read_jsonl(Path(cache_root) / "cache_index.jsonl")),
        tuple(_read_jsonl(Path(cache_root) / "dynamic_tracks.jsonl")),
        tuple(_read_jsonl(Path(cache_root) / "diagnostics_index.jsonl")),
    )
