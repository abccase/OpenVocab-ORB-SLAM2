#!/usr/bin/env python3
"""Run and validate reproducible ORB-SLAM2 TUM baseline conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RunCondition:
    sequence_id: str
    seed: int
    sequence_root: Path
    settings: Path
    dataset_manifest: Path | None = None
    experiment_manifest: Path | None = None
    semantic_manifest: Path | None = None
    prompt_config: Path | None = None


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    valid: bool
    frame_count: int
    invalid_reason: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path | None, value: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_trajectory(path: Path) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    previous = -math.inf
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"{path}:{line_number}: trajectory row must have eight columns")
        try:
            row = tuple(float(field) for field in fields)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric value") from exc
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{path}:{line_number}: non-finite trajectory value")
        if row[0] <= previous:
            raise ValueError(f"{path}:{line_number}: timestamps are not strictly increasing")
        previous = row[0]
        rows.append(row)
    if not rows:
        raise ValueError(f"trajectory has no poses: {path}")
    return rows


def _association_count(path: Path) -> int:
    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))
    if count == 0:
        raise ValueError(f"association file has no rows: {path}")
    return count


def _association_timestamps(path: Path) -> list[str]:
    timestamps: list[str] = []
    previous = -math.inf
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: association row must have four fields")
        value = float(fields[0])
        if not math.isfinite(value) or value <= previous:
            raise ValueError(f"{path}:{line_number}: invalid association timestamp")
        previous = value
        timestamps.append(fields[0])
    if not timestamps:
        raise ValueError(f"association file has no rows: {path}")
    return timestamps


def _validate_telemetry(path: Path, expected_frames: int) -> int:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid telemetry JSON") from exc
        required = {"frame_index", "timestamp", "tracking_state", "pose_valid", "tracking_time_seconds"}
        if not required <= set(row):
            raise ValueError(f"{path}:{line_number}: incomplete telemetry row")
        if row["frame_index"] != len(rows):
            raise ValueError(f"{path}:{line_number}: non-contiguous frame index")
        numeric = (float(row["timestamp"]), float(row["tracking_time_seconds"]))
        if not all(math.isfinite(value) for value in numeric) or numeric[1] < 0:
            raise ValueError(f"{path}:{line_number}: invalid telemetry numeric value")
        rows.append(row)
    if len(rows) != expected_frames:
        raise ValueError(f"telemetry coverage mismatch: {len(rows)} != {expected_frames}")
    return len(rows)


_OV_TELEMETRY_FIELDS = [
    "frame_index", "timestamp", "tracking_state", "pose_valid",
    "tracking_time_seconds", "raw_keypoints", "used_keypoints",
    "removed_dynamic", "retained_uncertain", "removed_uncertain",
    "semantic_accessed", "semantic_state", "cache_load_seconds",
    "policy_seconds", "pacing_lateness_seconds",
]


def _canonical_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer {label}") from exc
    if str(parsed) != value:
        raise ValueError(f"non-canonical integer {label}")
    return parsed


def _validate_ov_telemetry(path: Path, expected_frames: int, mode: str,
                           expected_timestamps: Sequence[str]) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != _OV_TELEMETRY_FIELDS:
            raise ValueError(f"{path}: unexpected telemetry CSV header")
        rows = list(reader)
    if len(rows) != expected_frames:
        raise ValueError(f"telemetry coverage mismatch: {len(rows)} != {expected_frames}")
    for index, row in enumerate(rows):
        if _canonical_int(row["frame_index"], "frame_index") != index:
            raise ValueError(f"{path}: non-contiguous frame index")
        if row["tracking_state"] not in {"-1", "0", "1", "2", "3"}:
            raise ValueError(f"{path}: invalid tracking state")
        if row["pose_valid"] not in {"0", "1"}:
            raise ValueError(f"{path}: invalid pose-valid flag")
        if float(row["timestamp"]) != float(expected_timestamps[index]):
            raise ValueError(f"{path}: timestamp differs from association")
        numeric = [
            float(row["timestamp"]), float(row["tracking_time_seconds"]),
            float(row["cache_load_seconds"]), float(row["policy_seconds"]),
            float(row["pacing_lateness_seconds"]),
        ]
        counts = [
            _canonical_int(row["raw_keypoints"], "raw_keypoints"),
            _canonical_int(row["used_keypoints"], "used_keypoints"),
            _canonical_int(row["removed_dynamic"], "removed_dynamic"),
            _canonical_int(row["retained_uncertain"], "retained_uncertain"),
            _canonical_int(row["removed_uncertain"], "removed_uncertain"),
        ]
        if not all(math.isfinite(value) and value >= 0 for value in numeric[1:]):
            raise ValueError(f"{path}: invalid timing value")
        if not math.isfinite(numeric[0]) or any(value < 0 for value in counts):
            raise ValueError(f"{path}: invalid telemetry value")
        if counts[1] + counts[2] + counts[4] != counts[0]:
            raise ValueError(f"{path}: feature accounting invariant failed")
        if counts[3] > counts[1]:
            raise ValueError(f"{path}: retained uncertain exceeds used keypoints")
        if row["semantic_accessed"] not in {"0", "1"}:
            raise ValueError(f"{path}: invalid semantic-access flag")
        semantic_accessed = int(row["semantic_accessed"])
        if mode == "baseline":
            if (semantic_accessed != 0 or row["semantic_state"] != "BASELINE" or
                    counts[0] != counts[1] or counts[2] != 0 or
                    counts[3] != 0 or counts[4] != 0 or
                    numeric[2] != 0.0 or numeric[3] != 0.0):
                raise ValueError(f"{path}: baseline accessed semantic state")
        elif semantic_accessed != 1 or row["semantic_state"] != "CACHE_VALID":
            raise ValueError(f"{path}: semantic-feedback frame lacks valid cache state")
    return len(rows)


def _artifact(path: Path, *, pose_count: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": path.name, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    if pose_count is not None:
        value["pose_count"] = pose_count
    return value


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: {path}")
    return value


def _formal_prompt_sha256(path: Path) -> str:
    prompt_file = _read_json_object(path, "prompt config")
    raw = prompt_file.get("frozen_formal_prompt")
    if not isinstance(raw, str):
        raise ValueError("prompt config lacks frozen_formal_prompt")
    terms: list[str] = []
    for part in raw.replace("\n", " ").split("."):
        term = " ".join(part.strip().lower().split())
        if term and term not in terms:
            terms.append(term)
    if not terms:
        raise ValueError("formal prompt has no terms")
    normalized = " . ".join(terms) + " ."
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _verified_inputs(condition: RunCondition, association: Path,
                     expected_frames: int, mode: str,
                     cache_root: Path | None,
                     cache_identity: dict[str, str] | None) -> dict[str, object]:
    association_sha = _sha256_file(association)
    if condition.dataset_manifest is None:
        if mode == "semantic-feedback":
            raise ValueError("semantic-feedback requires a dataset manifest")
        return {
            "dataset_manifest_sha256": None, "source_tree_sha256": None,
            "dynamic_manifest_sha256": None, "dynamic_config_sha256": None,
            "semantic_manifest_sha256": None, "semantic_identity_sha256": None,
            "inference_config_sha256": None, "prompt_sha256": None,
            "protocol": None,
        }
    dataset_path = Path(condition.dataset_manifest).resolve()
    dataset = _read_json_object(dataset_path, "dataset manifest")
    source_tree = dataset.get("extracted_tree_sha256")
    if (dataset.get("schema_version") != 1 or
            dataset.get("sequence_id") != condition.sequence_id or
            dataset.get("association_sha256") != association_sha or
            not _is_sha256(source_tree)):
        raise ValueError("dataset manifest identity does not match current sequence")
    result: dict[str, object] = {
        "dataset_manifest_sha256": _sha256_file(dataset_path),
        "source_tree_sha256": source_tree,
        "dynamic_manifest_sha256": None, "dynamic_config_sha256": None,
        "semantic_manifest_sha256": None, "semantic_identity_sha256": None,
        "inference_config_sha256": None, "prompt_sha256": None,
        "protocol": None,
    }
    experiment_study: object = None
    if condition.experiment_manifest is not None:
        experiment = _read_json_object(
            Path(condition.experiment_manifest).resolve(), "experiment manifest")
        experiment_study = experiment.get("study_id")
        if experiment_study != "ovorb2_tum_v1":
            raise ValueError("experiment manifest study identity mismatch")
    if mode == "baseline":
        return result
    if cache_root is None or cache_identity is None:
        raise ValueError("semantic cache registration is missing")
    manifest_path = cache_root / "cache_manifest.json"
    completion_path = cache_root / "cache_complete.json"
    index_path = cache_root / "cache_index.jsonl"
    for key, path in (("manifest_sha256", manifest_path),
                      ("completion_sha256", completion_path),
                      ("index_sha256", index_path)):
        if _sha256_file(path) != cache_identity[key]:
            raise ValueError(f"trusted dynamic cache {key} mismatch")
    dynamic = _read_json_object(manifest_path, "dynamic cache manifest")
    completion = _read_json_object(completion_path, "dynamic cache completion")
    if (dynamic.get("schema") != "ovorb.dynamic-cache.v1" or
            dynamic.get("sequence_id") != condition.sequence_id or
            dynamic.get("expected_frame_count") != expected_frames or
            dynamic.get("association_sha256") != association_sha or
            dynamic.get("dataset_manifest_sha256") != result["dataset_manifest_sha256"] or
            dynamic.get("source_tree_sha256") != source_tree or
            dynamic.get("study_id") != experiment_study):
        raise ValueError("dynamic cache manifest is not bound to current dataset manifest")
    if (completion.get("manifest_sha256") != cache_identity["manifest_sha256"] or
            completion.get("index_sha256") != cache_identity["index_sha256"] or
            completion.get("frame_count") != expected_frames):
        raise ValueError("dynamic cache completion identity mismatch")
    if condition.semantic_manifest is None or condition.prompt_config is None:
        raise ValueError("semantic-feedback requires semantic manifest and prompt config")
    semantic_path = Path(condition.semantic_manifest).resolve()
    prompt_path = Path(condition.prompt_config).resolve()
    semantic = _read_json_object(semantic_path, "semantic cache manifest")
    semantic_sha = _sha256_file(semantic_path)
    prompt_file_sha = _sha256_file(prompt_path)
    prompt_sha = _formal_prompt_sha256(prompt_path)
    if (dynamic.get("semantic_manifest_sha256") != semantic_sha or
            semantic.get("schema") != "ovorb.semantic-cache.v1" or
            semantic.get("sequence_id") != condition.sequence_id or
            semantic.get("source_tree_sha256") != source_tree or
            semantic.get("association_sha256") != association_sha or
            semantic.get("prompt_sha256") != prompt_sha or
            semantic.get("study_id") != dynamic.get("study_id")):
        raise ValueError("semantic manifest is not bound to current dataset/cache inputs")
    result.update({
        "dynamic_manifest_sha256": cache_identity["manifest_sha256"],
        "dynamic_completion_sha256": cache_identity["completion_sha256"],
        "dynamic_index_sha256": cache_identity["index_sha256"],
        "dynamic_config_sha256": dynamic.get("dynamic_config_sha256"),
        "semantic_manifest_sha256": semantic_sha,
        "semantic_identity_sha256": dynamic.get("semantic_identity_sha256"),
        "inference_config_sha256": semantic.get("inference_config_sha256"),
        "prompt_sha256": prompt_sha,
        "prompt_config_sha256": prompt_file_sha,
        "protocol": {"dynamic": dynamic.get("schema"),
                     "semantic": semantic.get("schema")},
    })
    return result


def _completed_attempt(
    attempt: Path,
    compatibility_commit: str,
    producer_commit: str,
    executable_sha256: str,
    expected_frames: int,
) -> RunResult | None:
    manifest_path = attempt / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state") != "COMPLETED" or manifest.get("valid") is not True:
            return None
        if manifest.get("compatibility_commit") != compatibility_commit:
            return None
        if manifest.get("producer_commit") != producer_commit:
            return None
        if manifest.get("executable", {}).get("sha256") != executable_sha256:
            return None
        trajectory_path = attempt / "CameraTrajectory.txt"
        keyframe_path = attempt / "KeyFrameTrajectory.txt"
        telemetry_path = attempt / "frame_telemetry.jsonl"
        trajectory = parse_trajectory(trajectory_path)
        parse_trajectory(keyframe_path)
        frame_count = _validate_telemetry(telemetry_path, expected_frames)
        if _sha256_file(trajectory_path) != manifest["trajectory"]["sha256"]:
            return None
        if _sha256_file(telemetry_path) != manifest["telemetry"]["sha256"]:
            return None
        return RunResult(attempt, True, frame_count, None)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def run_baseline_condition(
    condition: RunCondition,
    *,
    executable: Path,
    vocabulary: Path,
    output_root: Path,
    compatibility_commit: str,
    producer_commit: str,
    registry: Path | None = None,
    study: str = "oracle",
) -> RunResult:
    sequence_root = Path(condition.sequence_root).resolve()
    association = sequence_root / "associate.txt"
    expected_frames = _association_count(association)
    executable = Path(executable).resolve()
    vocabulary = Path(vocabulary).resolve()
    settings = Path(condition.settings).resolve()
    executable_sha256 = _sha256_file(executable)
    condition_root = Path(output_root) / condition.sequence_id / f"seed-{condition.seed}"
    attempts = sorted(condition_root.glob("attempt-*")) if condition_root.is_dir() else []
    for attempt in attempts:
        completed = _completed_attempt(
            attempt,
            compatibility_commit,
            producer_commit,
            executable_sha256,
            expected_frames,
        )
        if completed is not None:
            return completed
    attempt_number = len(attempts) + 1
    run_dir = condition_root / f"attempt-{attempt_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    telemetry = run_dir / "frame_telemetry.jsonl"
    command = [str(executable), str(vocabulary), str(settings), str(sequence_root), str(association)]
    if study not in {"smoke", "oracle"}:
        raise ValueError(f"unsupported baseline study: {study}")
    run_id = f"{study}-{condition.sequence_id}-seed-{condition.seed}-attempt-{attempt_number:03d}"
    started_utc = _utc_now()
    base_manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "study": study,
        "mode": "baseline",
        "sequence_id": condition.sequence_id,
        "seed": condition.seed,
        "compatibility_commit": compatibility_commit,
        "producer_commit": producer_commit,
        "executable": {"path": str(executable), "sha256": executable_sha256},
        "vocabulary": {"path": str(vocabulary), "sha256": _sha256_file(vocabulary)},
        "settings": {"path": str(settings), "sha256": _sha256_file(settings)},
        "command": command,
        "cwd": str(run_dir.resolve()),
        "expected_frames": expected_frames,
        "association_sha256": _sha256_file(association),
        "dataset_manifest_sha256": _sha256_file(condition.dataset_manifest) if condition.dataset_manifest else None,
        "start_time_utc": started_utc,
        "state": "REGISTERED",
        "valid": False,
    }
    _write_json_atomic(run_dir / "run_manifest.json", base_manifest)
    _append_jsonl(registry, {**base_manifest, "expected_outputs": [str(run_dir / name) for name in ("CameraTrajectory.txt", "KeyFrameTrajectory.txt", "frame_telemetry.jsonl", "run_manifest.json")]})
    running = {**base_manifest, "state": "RUNNING"}
    _write_json_atomic(run_dir / "run_manifest.json", running)
    _append_jsonl(registry, running)
    environment = os.environ.copy()
    environment["ORB_SLAM2_FRAME_TELEMETRY"] = str(telemetry.resolve())
    environment["ORB_SLAM2_RUN_SEED"] = str(condition.seed)
    monotonic_start = time.monotonic()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=run_dir, env=environment, stdout=stdout, stderr=stderr, check=False)
    wall_seconds = time.monotonic() - monotonic_start
    ended_utc = _utc_now()
    invalid_reason: str | None = None
    trajectory_rows: list[tuple[float, ...]] = []
    keyframe_rows: list[tuple[float, ...]] = []
    frame_count = 0
    if completed.returncode != 0:
        invalid_reason = f"process exited {completed.returncode}"
    else:
        try:
            trajectory_rows = parse_trajectory(run_dir / "CameraTrajectory.txt")
            keyframe_rows = parse_trajectory(run_dir / "KeyFrameTrajectory.txt")
            frame_count = _validate_telemetry(telemetry, expected_frames)
        except (OSError, ValueError) as exc:
            invalid_reason = str(exc)
    valid = invalid_reason is None
    final: dict[str, object] = {
        **base_manifest,
        "state": "COMPLETED" if valid else "FAILED",
        "valid": valid,
        "end_time_utc": ended_utc,
        "exit_code": completed.returncode,
        "wall_time_seconds": wall_seconds,
        "invalid_reason": invalid_reason,
    }
    if valid:
        final.update(
            {
                "trajectory": _artifact(run_dir / "CameraTrajectory.txt", pose_count=len(trajectory_rows)),
                "keyframe_trajectory": _artifact(run_dir / "KeyFrameTrajectory.txt", pose_count=len(keyframe_rows)),
                "telemetry": _artifact(telemetry),
                "frame_count": frame_count,
                "stdout": _artifact(run_dir / "stdout.log"),
                "stderr": _artifact(run_dir / "stderr.log"),
            }
        )
    _write_json_atomic(run_dir / "run_manifest.json", final)
    _append_jsonl(registry, final)
    return RunResult(run_dir, valid, frame_count, invalid_reason)


def _completed_ov_attempt(
    attempt: Path,
    *,
    registration_identity: dict[str, object],
    expected_frames: int,
    expected_timestamps: Sequence[str],
) -> RunResult | None:
    manifest_path = attempt / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("state") != "COMPLETED" or manifest.get("valid") is not True or
                manifest.get("registration_identity") != registration_identity):
            return None
        mode = str(registration_identity["mode"])
        trajectory_path = attempt / "CameraTrajectory.txt"
        keyframe_path = attempt / "KeyFrameTrajectory.txt"
        telemetry_path = attempt / "frame_telemetry.csv"
        parse_trajectory(trajectory_path)
        parse_trajectory(keyframe_path)
        frame_count = _validate_ov_telemetry(
            telemetry_path, expected_frames, mode, expected_timestamps)
        final_state = json.loads((attempt / "final_state.json").read_text(encoding="utf-8"))
        timings = json.loads((attempt / "timings.json").read_text(encoding="utf-8"))
        if (final_state.get("state") != "COMPLETED" or final_state.get("mode") != mode or
                int(final_state.get("frame_count", -1)) != expected_frames or
                int(timings.get("frame_count", -1)) != expected_frames):
            return None
        for name in ("mean_tracking_seconds", "median_tracking_seconds",
                     "mean_pacing_lateness_seconds", "max_pacing_lateness_seconds",
                     "wall_seconds"):
            value = float(timings[name])
            if not math.isfinite(value) or value < 0.0:
                return None
        for key, path in (("trajectory", trajectory_path),
                          ("keyframe_trajectory", keyframe_path),
                          ("telemetry", telemetry_path),
                          ("final_state", attempt / "final_state.json"),
                          ("timings", attempt / "timings.json"),
                          ("stdout", attempt / "stdout.log"),
                          ("stderr", attempt / "stderr.log")):
            if _sha256_file(path) != manifest[key]["sha256"]:
                return None
        return RunResult(attempt, True, frame_count, None)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def run_ov_condition(
    condition: RunCondition,
    *,
    mode: str,
    executable: Path,
    vocabulary: Path,
    output_root: Path,
    compatibility_commit: str,
    producer_commit: str,
    study: str,
    registry: Path | None = None,
    cache_root: Path | None = None,
    cache_identity: dict[str, str] | None = None,
) -> RunResult:
    if mode not in {"baseline", "semantic-feedback"}:
        raise ValueError(f"unsupported ORB-SLAM2 mode: {mode}")
    if study not in {"equivalence", "smoke"}:
        raise ValueError(f"unsupported semantic integration study: {study}")
    if mode == "baseline":
        if cache_root is not None or cache_identity is not None:
            raise ValueError("baseline mode must not receive semantic cache assets")
    else:
        if cache_root is None or cache_identity is None:
            raise ValueError("semantic-feedback requires a trusted cache identity")
        required_identity = {"manifest_sha256", "completion_sha256", "index_sha256"}
        if set(cache_identity) != required_identity or any(
                not _is_sha256(cache_identity[key]) for key in required_identity):
            raise ValueError("semantic-feedback cache identity is incomplete")

    sequence_root = Path(condition.sequence_root).resolve()
    association = sequence_root / "associate.txt"
    timestamp_lexemes = _association_timestamps(association)
    expected_frames = len(timestamp_lexemes)
    executable = Path(executable).resolve()
    vocabulary = Path(vocabulary).resolve()
    settings = Path(condition.settings).resolve()
    executable_sha256 = _sha256_file(executable)
    resolved_cache_root = Path(cache_root).resolve() if cache_root is not None else None
    verified_inputs = _verified_inputs(
        condition, association, expected_frames, mode, resolved_cache_root,
        cache_identity)
    command = [
        str(executable), str(vocabulary), str(settings), str(sequence_root),
        str(association), mode, condition.sequence_id, str(condition.seed),
    ]
    if mode == "semantic-feedback":
        assert resolved_cache_root is not None and cache_identity is not None
        command.extend([
            str(resolved_cache_root), cache_identity["manifest_sha256"],
            cache_identity["completion_sha256"], cache_identity["index_sha256"],
        ])
    dataset_manifest_sha = (_sha256_file(condition.dataset_manifest)
                            if condition.dataset_manifest else None)
    experiment_manifest_sha = (_sha256_file(condition.experiment_manifest)
                               if condition.experiment_manifest else None)
    registration_identity: dict[str, object] = {
        "study": study, "mode": mode, "sequence_id": condition.sequence_id,
        "seed": condition.seed, "compatibility_commit": compatibility_commit,
        "producer_commit": producer_commit,
        "executable": {"path": str(executable), "sha256": executable_sha256},
        "vocabulary": {"path": str(vocabulary), "sha256": _sha256_file(vocabulary)},
        "settings": {"path": str(settings), "sha256": _sha256_file(settings)},
        "association": {"path": str(association), "sha256": _sha256_file(association)},
        "dataset": {"root": str(sequence_root), "manifest_sha256": dataset_manifest_sha},
        "source_tree_sha256": verified_inputs["source_tree_sha256"],
        "experiment_manifest_sha256": experiment_manifest_sha,
        "prompt_sha256": verified_inputs["prompt_sha256"],
        "verified_inputs": verified_inputs,
        "command": command,
        "cache_root": str(resolved_cache_root) if resolved_cache_root else None,
        "cache_identity": cache_identity,
        "expected_frames": expected_frames,
        "pacing": "dataset_timestamp_paced_relative",
    }
    condition_root = Path(output_root) / condition.sequence_id / f"seed-{condition.seed}"
    attempts = sorted(condition_root.glob("attempt-*")) if condition_root.is_dir() else []
    for attempt in attempts:
        completed_attempt = _completed_ov_attempt(
            attempt, registration_identity=registration_identity,
            expected_frames=expected_frames, expected_timestamps=timestamp_lexemes,
        )
        if completed_attempt is not None:
            return completed_attempt

    attempt_number = len(attempts) + 1
    run_dir = condition_root / f"attempt-{attempt_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    run_id = f"{study}-{mode}-{condition.sequence_id}-seed-{condition.seed}-attempt-{attempt_number:03d}"
    base_manifest: dict[str, object] = {
        "schema_version": 2,
        "run_id": run_id,
        "study": study,
        "mode": mode,
        "sequence_id": condition.sequence_id,
        "seed": condition.seed,
        "compatibility_commit": compatibility_commit,
        "producer_commit": producer_commit,
        "executable": {"path": str(executable), "sha256": executable_sha256},
        "vocabulary": {"path": str(vocabulary), "sha256": _sha256_file(vocabulary)},
        "settings": {"path": str(settings), "sha256": _sha256_file(settings)},
        "command": command,
        "cwd": str(run_dir.resolve()),
        "expected_frames": expected_frames,
        "association_sha256": _sha256_file(association),
        "dataset_manifest_sha256": _sha256_file(condition.dataset_manifest) if condition.dataset_manifest else None,
        "cache_root": str(resolved_cache_root) if resolved_cache_root else None,
        "cache_identity": cache_identity,
        "registration_identity": registration_identity,
        "verified_inputs": verified_inputs,
        "pacing": "dataset_timestamp_paced_relative",
        "start_time_utc": _utc_now(),
        "state": "REGISTERED",
        "valid": False,
    }
    _write_json_atomic(run_dir / "run_manifest.json", base_manifest)
    _append_jsonl(registry, {
        **base_manifest,
        "expected_outputs": [str(run_dir / name) for name in (
            "CameraTrajectory.txt", "KeyFrameTrajectory.txt", "frame_telemetry.csv",
            "timings.json", "final_state.json", "run_manifest.json")],
    })
    running = {**base_manifest, "state": "RUNNING"}
    _write_json_atomic(run_dir / "run_manifest.json", running)
    _append_jsonl(registry, running)
    environment = os.environ.copy()
    environment["ORB_SLAM2_RUN_SEED"] = str(condition.seed)
    monotonic_start = time.monotonic()
    launch_error: OSError | None = None
    completed: subprocess.CompletedProcess[bytes] | None = None
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, \
            (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(command, cwd=run_dir, env=environment,
                                       stdout=stdout, stderr=stderr, check=False)
        except OSError as exc:
            launch_error = exc
    wall_seconds = time.monotonic() - monotonic_start
    invalid_reason: str | None = None
    trajectory_rows: list[tuple[float, ...]] = []
    keyframe_rows: list[tuple[float, ...]] = []
    frame_count = 0
    if launch_error is not None:
        invalid_reason = f"launch failed: {launch_error}"
    elif completed is None:
        invalid_reason = "launch failed: subprocess returned no result"
    elif completed.returncode != 0:
        invalid_reason = f"process exited {completed.returncode}"
    else:
        try:
            trajectory_rows = parse_trajectory(run_dir / "CameraTrajectory.txt")
            keyframe_rows = parse_trajectory(run_dir / "KeyFrameTrajectory.txt")
            frame_count = _validate_ov_telemetry(
                run_dir / "frame_telemetry.csv", expected_frames, mode,
                timestamp_lexemes)
            final_state = json.loads((run_dir / "final_state.json").read_text(encoding="utf-8"))
            timings = json.loads((run_dir / "timings.json").read_text(encoding="utf-8"))
            if (final_state.get("state") != "COMPLETED" or final_state.get("mode") != mode or
                    int(final_state.get("frame_count", -1)) != expected_frames or
                    int(timings.get("frame_count", -1)) != expected_frames):
                raise ValueError("executable final state or timings are incomplete")
            for name in ("mean_tracking_seconds", "median_tracking_seconds",
                         "mean_pacing_lateness_seconds", "max_pacing_lateness_seconds",
                         "wall_seconds"):
                value = float(timings[name])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"invalid executable timing aggregate: {name}")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            invalid_reason = str(exc)
    valid = invalid_reason is None
    final: dict[str, object] = {
        **base_manifest,
        "state": "COMPLETED" if valid else "FAILED",
        "valid": valid,
        "end_time_utc": _utc_now(),
        "exit_code": completed.returncode if completed is not None else None,
        "wall_time_seconds": wall_seconds,
        "invalid_reason": invalid_reason,
    }
    if valid:
        final.update({
            "trajectory": _artifact(run_dir / "CameraTrajectory.txt", pose_count=len(trajectory_rows)),
            "keyframe_trajectory": _artifact(run_dir / "KeyFrameTrajectory.txt", pose_count=len(keyframe_rows)),
            "telemetry": {**_artifact(run_dir / "frame_telemetry.csv"), "format": "csv"},
            "timings": _artifact(run_dir / "timings.json"),
            "final_state": _artifact(run_dir / "final_state.json"),
            "frame_count": frame_count,
            "stdout": _artifact(run_dir / "stdout.log"),
            "stderr": _artifact(run_dir / "stderr.log"),
        })
    _write_json_atomic(run_dir / "run_manifest.json", final)
    _append_jsonl(registry, final)
    return RunResult(run_dir, valid, frame_count, invalid_reason)


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "semantic-feedback"), required=True)
    parser.add_argument("--study", choices=("smoke", "oracle", "equivalence"), required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--all-sequences", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("config/EXPERIMENT_MANIFEST.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/tum/raw"))
    parser.add_argument("--data-manifests", type=Path, default=Path("data/tum/manifests"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("runs/registry.jsonl"))
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--vocabulary", type=Path, default=Path("Vocabulary/ORBvoc.txt"))
    parser.add_argument("--dynamic-cache-root", type=Path, default=Path("cache/dynamic/v1"))
    parser.add_argument("--semantic-cache-root", type=Path, default=Path("cache/semantic/v1"))
    parser.add_argument("--prompt-config", type=Path, default=Path("config/PROMPTS.yaml"))
    parser.add_argument("--dynamic-cache-identities", type=Path,
                        default=Path("config/DYNAMIC_CACHE_IDENTITY.json"))
    args = parser.parse_args()
    root = Path.cwd()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    datasets = manifest["datasets"]
    selected = datasets if args.all_sequences else [item for item in datasets if item["id"] == args.sequence]
    if not selected:
        parser.error("select one known --sequence or use --all-sequences")
    seeds = manifest["random_seeds"] if args.all_seeds else [args.seed]
    if any(seed is None for seed in seeds):
        parser.error("select --seed or use --all-seeds")
    compatibility_commit = _git_output(root, "rev-parse", "baseline/ubuntu22^{}")
    producer_commit = _git_output(root, "rev-parse", "HEAD")
    use_ov_runner = args.mode == "semantic-feedback" or args.study == "equivalence"
    effective_mode = args.mode
    if args.study == "oracle" and effective_mode != "baseline":
        parser.error("oracle study is baseline-only")
    executable = args.executable or Path(
        "Examples/RGB-D/rgbd_tum_ov" if use_ov_runner else "Examples/RGB-D/rgbd_tum")
    output_root = args.output_root or (
        Path("runs") / args.study / effective_mode if use_ov_runner
        else Path("runs") / args.study
    )
    identities: dict[str, object] | None = None
    if effective_mode == "semantic-feedback":
        identities = json.loads(args.dynamic_cache_identities.read_text(encoding="utf-8"))
        if identities.get("cache_schema") != "ovorb.dynamic-cache.v1":
            parser.error("dynamic cache identity schema mismatch")
    failures = 0
    for item in selected:
        sequence_root = args.data_root / item["archive"][:-4]
        condition = RunCondition(
            item["id"],
            int(seeds[0]),
            sequence_root,
            Path("Examples/RGB-D") / item["settings"],
            args.data_manifests / f"{item['id']}.json",
            args.manifest,
            args.semantic_cache_root / item["id"] / "cache_manifest.json",
            args.prompt_config,
        )
        for seed in seeds:
            condition = RunCondition(
                condition.sequence_id, int(seed), condition.sequence_root,
                condition.settings, condition.dataset_manifest,
                condition.experiment_manifest, condition.semantic_manifest,
                condition.prompt_config)
            if use_ov_runner:
                identity = None
                cache_root = None
                if effective_mode == "semantic-feedback":
                    assert identities is not None
                    try:
                        identity = dict(identities["sequences"][condition.sequence_id])
                    except (KeyError, TypeError) as exc:
                        parser.error(f"missing cache identity for {condition.sequence_id}: {exc}")
                    cache_root = args.dynamic_cache_root / condition.sequence_id
                result = run_ov_condition(
                    condition, mode=effective_mode, executable=executable,
                    vocabulary=args.vocabulary, output_root=output_root,
                    compatibility_commit=compatibility_commit,
                    producer_commit=producer_commit, registry=args.registry,
                    study=args.study, cache_root=cache_root, cache_identity=identity,
                )
            else:
                result = run_baseline_condition(
                    condition, executable=executable, vocabulary=args.vocabulary,
                    output_root=output_root, compatibility_commit=compatibility_commit,
                    producer_commit=producer_commit, registry=args.registry,
                    study=args.study,
                )
            print(f"{'VALID' if result.valid else 'INVALID'} {condition.sequence_id} seed={seed} dir={result.run_dir} reason={result.invalid_reason}")
            failures += int(not result.valid)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
