#!/usr/bin/env python3
"""Freeze, register, and resumably execute the P08 formal study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from semantic_py.openvocab_slam.config import FORMAL_BASELINE_COMPATIBILITY_COMMIT  # noqa: E402
from semantic_py.openvocab_slam.experiments import (  # noqa: E402
    ASSOCIATION_MAX_SECONDS,
    RPE_DELTA_SECONDS,
    SEQUENCE_IDS,
    STUDY_ID,
    build_run_matrix,
    compute_trajectory_metrics,
    freeze_run_matrix,
    load_experiment_manifest,
    read_run_matrix,
    sha256_file,
    validate_run_manifest,
)
from tools.run_orb_tum import (  # noqa: E402
    RunCondition,
    _association_timestamps,
    _completed_ov_attempt,
    _formal_prompt_sha256,
    _validate_ov_telemetry,
    parse_trajectory,
    run_ov_condition,
)


DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "runs" / STUDY_ID


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _settings_name(manifest: Mapping[str, object], sequence_id: str) -> str:
    datasets = manifest["datasets"]
    assert isinstance(datasets, list)
    for dataset in datasets:
        if dataset["id"] == sequence_id:
            return str(dataset["settings"])
    raise ValueError(f"dataset missing from manifest: {sequence_id}")


def _cache_identity(cache_root: Path) -> dict[str, str]:
    manifest = cache_root / "cache_manifest.json"
    completion = cache_root / "cache_complete.json"
    index = cache_root / "cache_index.jsonl"
    identity = {
        "manifest_sha256": sha256_file(manifest),
        "completion_sha256": sha256_file(completion),
        "index_sha256": sha256_file(index),
    }
    completed = _read_json(completion, "dynamic cache completion")
    if completed.get("manifest_sha256") != identity["manifest_sha256"]:
        raise ValueError(f"dynamic cache completion does not bind manifest: {cache_root}")
    if completed.get("index_sha256") != identity["index_sha256"]:
        raise ValueError(f"dynamic cache completion does not bind index: {cache_root}")
    return identity


def build_registration(
    manifest: Mapping[str, object],
    manifest_path: Path,
    run_order_path: Path,
    executable: Path,
    data_root: Path,
    data_manifests: Path,
    dynamic_cache_root: Path,
    semantic_cache_root: Path,
) -> dict[str, object]:
    producer_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    datasets: dict[str, object] = {}
    prompt_path = REPOSITORY_ROOT / "config" / "PROMPTS.yaml"
    for sequence_id in SEQUENCE_IDS:
        sequence_root = Path(data_root) / f"rgbd_dataset_freiburg{sequence_id[2:]}"
        dataset_manifest_path = Path(data_manifests) / f"{sequence_id}.json"
        dataset_manifest = _read_json(dataset_manifest_path, "dataset manifest")
        association = sequence_root / "associate.txt"
        groundtruth = sequence_root / "groundtruth.txt"
        dynamic_root = Path(dynamic_cache_root) / sequence_id
        dynamic_manifest = _read_json(dynamic_root / "cache_manifest.json", "dynamic cache manifest")
        semantic_manifest_path = Path(semantic_cache_root) / sequence_id / "cache_manifest.json"
        semantic_manifest = _read_json(semantic_manifest_path, "semantic cache manifest")
        cache_identity = _cache_identity(dynamic_root)
        if (
            dataset_manifest.get("sequence_id") != sequence_id
            or dynamic_manifest.get("sequence_id") != sequence_id
            or semantic_manifest.get("sequence_id") != sequence_id
            or dynamic_manifest.get("source_tree_sha256")
            != dataset_manifest.get("extracted_tree_sha256")
        ):
            raise ValueError(f"dataset/cache identity mismatch: {sequence_id}")
        datasets[sequence_id] = {
            "sequence_root": str(sequence_root.resolve()),
            "dataset_manifest": str(dataset_manifest_path.resolve()),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "source_tree_sha256": dataset_manifest["extracted_tree_sha256"],
            "association": str(association.resolve()),
            "association_sha256": sha256_file(association),
            "expected_frames": len(_association_timestamps(association)),
            "groundtruth": str(groundtruth.resolve()),
            "groundtruth_sha256": sha256_file(groundtruth),
            "settings": str((REPOSITORY_ROOT / "Examples" / "RGB-D" / _settings_name(manifest, sequence_id)).resolve()),
            "settings_sha256": sha256_file(REPOSITORY_ROOT / "Examples" / "RGB-D" / _settings_name(manifest, sequence_id)),
            "cache_root": str(dynamic_root.resolve()),
            "cache_identity": cache_identity,
            "semantic_manifest": str(semantic_manifest_path.resolve()),
            "semantic_manifest_sha256": sha256_file(semantic_manifest_path),
            "semantic_identity_sha256": dynamic_manifest["semantic_identity_sha256"],
            "inference_config_sha256": semantic_manifest["inference_config_sha256"],
            "prompt_config": str(prompt_path.resolve()),
            "prompt_config_sha256": sha256_file(prompt_path),
            "prompt_sha256": semantic_manifest["prompt_sha256"],
            "configuration_sha256": dynamic_manifest["dynamic_config_sha256"],
            "dynamic_schema": dynamic_manifest["schema"],
            "semantic_schema": semantic_manifest["schema"],
        }
    return {
        "schema_version": 1,
        "state": "REGISTERED",
        "registered_utc": _utc_now(),
        "study_id": STUDY_ID,
        "repository": str(REPOSITORY_ROOT),
        "producer_commit": producer_commit,
        "compatibility_commit": FORMAL_BASELINE_COMPATIBILITY_COMMIT,
        "experiment_manifest": str(Path(manifest_path).resolve()),
        "experiment_manifest_sha256": sha256_file(manifest_path),
        "run_order": str(Path(run_order_path).resolve()),
        "run_order_sha256": sha256_file(run_order_path),
        "executable": str(Path(executable).resolve()),
        "executable_sha256": sha256_file(executable),
        "vocabulary": str((REPOSITORY_ROOT / "Vocabulary" / "ORBvoc.txt").resolve()),
        "vocabulary_sha256": sha256_file(REPOSITORY_ROOT / "Vocabulary" / "ORBvoc.txt"),
        "datasets": datasets,
        "expected_runs": 60,
    }


def _without_time(value: Mapping[str, object]) -> dict[str, object]:
    copied = dict(value)
    copied.pop("registered_utc", None)
    return copied


def freeze_and_register(
    manifest_path: Path,
    run_root: Path,
    executable: Path,
    data_root: Path,
    data_manifests: Path,
    dynamic_cache_root: Path,
    semantic_cache_root: Path,
) -> dict[str, object]:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT, text=True,
    ).strip()
    if dirty:
        raise ValueError("tracked worktree must be clean before formal registration")
    manifest = load_experiment_manifest(manifest_path)
    run_order_path = Path(run_root) / "run_matrix.csv"
    order_hash = freeze_run_matrix(build_run_matrix(manifest), run_order_path)
    registration_path = Path(run_root) / "study_registration.json"
    fresh = build_registration(
        manifest, manifest_path, run_order_path, executable, data_root,
        data_manifests, dynamic_cache_root, semantic_cache_root,
    )
    if fresh["run_order_sha256"] != order_hash:
        raise ValueError("run order changed during registration")
    if registration_path.is_file():
        existing = _read_json(registration_path, "study registration")
        fresh["registered_utc"] = existing.get("registered_utc")
        if _without_time(existing) != _without_time(fresh):
            raise ValueError("existing study registration differs from immutable inputs")
        return existing
    if any(Path(run_root).glob("**/attempt-*")):
        raise ValueError("formal attempt exists before study registration")
    _write_json_atomic(registration_path, fresh)
    _append_jsonl(Path(run_root) / "run_registry.jsonl", {
        "kind": "p08_study", "state": "REGISTERED", **fresh,
    })
    return fresh


def validate_registration_current(registration: Mapping[str, object]) -> None:
    if registration.get("study_id") != STUDY_ID or registration.get("expected_runs") != 60:
        raise ValueError("study registration identity mismatch")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    if registration.get("producer_commit") != current_commit:
        raise ValueError("registered producer commit differs from current HEAD")
    identities = (
        (Path(str(registration["experiment_manifest"])), registration["experiment_manifest_sha256"], "experiment manifest"),
        (Path(str(registration["run_order"])), registration["run_order_sha256"], "run order"),
        (Path(str(registration["executable"])), registration["executable_sha256"], "executable"),
        (Path(str(registration["vocabulary"])), registration["vocabulary_sha256"], "vocabulary"),
    )
    for path, expected, label in identities:
        if sha256_file(path) != expected:
            raise ValueError(f"registered {label} hash mismatch")
    load_experiment_manifest(Path(str(registration["experiment_manifest"])))
    read_run_matrix(Path(str(registration["run_order"])))
    datasets = registration.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(SEQUENCE_IDS):
        raise ValueError("registered datasets differ from frozen order")
    for sequence_id in SEQUENCE_IDS:
        dataset = datasets[sequence_id]
        if not isinstance(dataset, Mapping):
            raise ValueError(f"registered dataset is malformed: {sequence_id}")
        if sha256_file(Path(str(dataset["association"]))) != dataset["association_sha256"]:
            raise ValueError(f"registered association hash mismatch: {sequence_id}")
        if sha256_file(Path(str(dataset["groundtruth"]))) != dataset["groundtruth_sha256"]:
            raise ValueError(f"registered groundtruth hash mismatch: {sequence_id}")
        for path_key, hash_key, label in (
            ("dataset_manifest", "dataset_manifest_sha256", "dataset manifest"),
            ("settings", "settings_sha256", "settings"),
            ("semantic_manifest", "semantic_manifest_sha256", "semantic manifest"),
            ("prompt_config", "prompt_config_sha256", "prompt config"),
        ):
            if sha256_file(Path(str(dataset[path_key]))) != dataset[hash_key]:
                raise ValueError(f"registered {label} hash mismatch: {sequence_id}")
        if _cache_identity(Path(str(dataset["cache_root"]))) != dataset["cache_identity"]:
            raise ValueError(f"registered dynamic cache hash mismatch: {sequence_id}")
        dataset_manifest = _read_json(
            Path(str(dataset["dataset_manifest"])), "dataset manifest"
        )
        dynamic_manifest = _read_json(
            Path(str(dataset["cache_root"])) / "cache_manifest.json",
            "dynamic cache manifest",
        )
        semantic_manifest = _read_json(
            Path(str(dataset["semantic_manifest"])), "semantic cache manifest"
        )
        expected_content = {
            "source_tree_sha256": dataset_manifest.get("extracted_tree_sha256"),
            "configuration_sha256": dynamic_manifest.get("dynamic_config_sha256"),
            "semantic_manifest_sha256": dynamic_manifest.get("semantic_manifest_sha256"),
            "semantic_identity_sha256": dynamic_manifest.get("semantic_identity_sha256"),
            "inference_config_sha256": semantic_manifest.get("inference_config_sha256"),
            "prompt_sha256": semantic_manifest.get("prompt_sha256"),
            "dynamic_schema": dynamic_manifest.get("schema"),
            "semantic_schema": semantic_manifest.get("schema"),
        }
        for key, expected in expected_content.items():
            if dataset.get(key) != expected:
                raise ValueError(
                    f"registered {key} differs from source manifests: {sequence_id}"
                )
        if dataset.get("expected_frames") != len(
            _association_timestamps(Path(str(dataset["association"])))
        ):
            raise ValueError(f"registered frame count mismatch: {sequence_id}")
        if _formal_prompt_sha256(Path(str(dataset["prompt_config"]))) != dataset.get(
            "prompt_sha256"
        ):
            raise ValueError(f"registered prompt config mismatch: {sequence_id}")


def _formal_identity(condition: Mapping[str, object], registration: Mapping[str, object]) -> dict[str, object]:
    return {
        "study_id": STUDY_ID,
        "block_id": condition["block_id"],
        "mode": condition["mode"],
        "protocol_manifest_sha256": registration["experiment_manifest_sha256"],
        "run_order_sha256": registration["run_order_sha256"],
        "producer_commit": registration["producer_commit"],
    }


def expected_registration_identity(
    condition: Mapping[str, object],
    registration: Mapping[str, object],
) -> dict[str, object]:
    sequence = str(condition["sequence_id"])
    mode = str(condition["mode"])
    seed = int(condition["seed"])
    dataset = registration["datasets"][sequence]
    if not isinstance(dataset, Mapping):
        raise ValueError("registered dataset identity is malformed")
    cache_identity = dataset["cache_identity"] if mode == "semantic-feedback" else None
    cache_root = str(dataset["cache_root"]) if cache_identity is not None else None
    if mode == "semantic-feedback":
        verified_inputs: dict[str, object] = {
            "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
            "source_tree_sha256": dataset["source_tree_sha256"],
            "dynamic_manifest_sha256": cache_identity["manifest_sha256"],
            "dynamic_completion_sha256": cache_identity["completion_sha256"],
            "dynamic_index_sha256": cache_identity["index_sha256"],
            "dynamic_config_sha256": dataset["configuration_sha256"],
            "semantic_manifest_sha256": dataset["semantic_manifest_sha256"],
            "semantic_identity_sha256": dataset["semantic_identity_sha256"],
            "inference_config_sha256": dataset["inference_config_sha256"],
            "prompt_sha256": dataset["prompt_sha256"],
            "prompt_config_sha256": dataset["prompt_config_sha256"],
            "protocol": {
                "dynamic": dataset["dynamic_schema"],
                "semantic": dataset["semantic_schema"],
            },
        }
    else:
        verified_inputs = {
            "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
            "source_tree_sha256": dataset["source_tree_sha256"],
            "dynamic_manifest_sha256": None,
            "dynamic_config_sha256": None,
            "semantic_manifest_sha256": None,
            "semantic_identity_sha256": None,
            "inference_config_sha256": None,
            "prompt_sha256": None,
            "protocol": None,
        }
    command = [
        str(registration["executable"]), str(registration["vocabulary"]),
        str(dataset["settings"]), str(dataset["sequence_root"]),
        str(dataset["association"]), mode, sequence, str(seed),
    ]
    if cache_identity is not None:
        command.extend([
            str(dataset["cache_root"]),
            str(cache_identity["manifest_sha256"]),
            str(cache_identity["completion_sha256"]),
            str(cache_identity["index_sha256"]),
        ])
    return {
        "study": STUDY_ID,
        "mode": mode,
        "sequence_id": sequence,
        "seed": seed,
        "compatibility_commit": registration["compatibility_commit"],
        "producer_commit": registration["producer_commit"],
        "executable": {
            "path": str(registration["executable"]),
            "sha256": registration["executable_sha256"],
        },
        "vocabulary": {
            "path": str(registration["vocabulary"]),
            "sha256": registration["vocabulary_sha256"],
        },
        "settings": {
            "path": str(dataset["settings"]),
            "sha256": dataset["settings_sha256"],
        },
        "association": {
            "path": str(dataset["association"]),
            "sha256": dataset["association_sha256"],
        },
        "dataset": {
            "root": str(dataset["sequence_root"]),
            "manifest_sha256": dataset["dataset_manifest_sha256"],
        },
        "source_tree_sha256": dataset["source_tree_sha256"],
        "experiment_manifest_sha256": registration["experiment_manifest_sha256"],
        "prompt_sha256": verified_inputs["prompt_sha256"],
        "verified_inputs": verified_inputs,
        "command": command,
        "cache_root": cache_root,
        "cache_identity": cache_identity,
        "expected_frames": dataset["expected_frames"],
        "pacing": "dataset_timestamp_paced_relative",
        "formal_identity": _formal_identity(condition, registration),
    }


def validate_attempt_registration_identity(
    manifest: Mapping[str, object],
    condition: Mapping[str, object],
    registration: Mapping[str, object],
) -> dict[str, object]:
    expected = expected_registration_identity(condition, registration)
    if manifest.get("registration_identity") != expected:
        raise ValueError("attempt differs from reconstructed registered identity")
    top_level_fields = {
        "study": expected["study"],
        "mode": expected["mode"],
        "sequence_id": expected["sequence_id"],
        "seed": expected["seed"],
        "compatibility_commit": expected["compatibility_commit"],
        "producer_commit": expected["producer_commit"],
        "executable": expected["executable"],
        "vocabulary": expected["vocabulary"],
        "settings": expected["settings"],
        "association_sha256": expected["association"]["sha256"],
        "dataset_manifest_sha256": expected["dataset"]["manifest_sha256"],
        "cache_root": expected["cache_root"],
        "cache_identity": expected["cache_identity"],
        "verified_inputs": expected["verified_inputs"],
        "command": expected["command"],
        "expected_frames": expected["expected_frames"],
        "pacing": expected["pacing"],
        "formal_identity": expected["formal_identity"],
    }
    for field, value in top_level_fields.items():
        if manifest.get(field) != value:
            raise ValueError(f"attempt {field} differs from registered identity")
    return expected


def terminalize_failed_validation(attempt: Path, reason: str) -> None:
    attempt = Path(attempt)
    manifest_path = attempt / "run_manifest.json"
    preserved_manifest_path = attempt / "run_manifest.pre_p08_validation.json"
    failure_path = attempt / "p08_validation_failure.json"
    original_hash = sha256_file(manifest_path) if manifest_path.is_file() else None
    if manifest_path.is_file() and not preserved_manifest_path.exists():
        temporary = attempt / f".{preserved_manifest_path.name}.partial"
        with temporary.open("wb") as stream:
            stream.write(manifest_path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(preserved_manifest_path)
    failure = {
        "schema_version": 1,
        "state": "FAILED",
        "valid": False,
        "failed_utc": _utc_now(),
        "reason": reason,
        "original_run_manifest_sha256": original_hash,
    }
    if not failure_path.exists():
        _write_json_atomic(failure_path, failure)
    try:
        manifest = _read_json(manifest_path, "run manifest")
    except ValueError:
        return
    if manifest.get("state") == "FAILED" and manifest.get("valid") is False:
        return
    manifest.update({
        "state": "FAILED",
        "valid": False,
        "invalid_reason": f"P08 strict validation failed: {reason}",
    })
    _write_json_atomic(manifest_path, manifest)


def _parse_trajectory_file(path: Path) -> list[tuple[float, ...]]:
    return parse_trajectory(path)


def _metric_payload(attempt: Path, dataset: Mapping[str, object]) -> dict[str, object]:
    groundtruth_path = Path(str(dataset["groundtruth"]))
    trajectory_path = attempt / "CameraTrajectory.txt"
    metrics = compute_trajectory_metrics(
        _parse_trajectory_file(groundtruth_path), _parse_trajectory_file(trajectory_path)
    )
    return {
        "schema_version": 1,
        "tool": "semantic_py.openvocab_slam.experiments.compute_trajectory_metrics",
        "arguments": {
            "alignment": "SE3",
            "scale_alignment": False,
            "association_max_seconds": ASSOCIATION_MAX_SECONDS,
            "rpe_delta_seconds": RPE_DELTA_SECONDS,
            "rpe_delta_unit": "seconds",
        },
        "groundtruth_sha256": sha256_file(groundtruth_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "metrics": metrics,
    }


def validate_attempt(
    attempt: Path,
    condition: Mapping[str, object],
    registration: Mapping[str, object],
) -> tuple[bool, str, dict[str, object] | None]:
    try:
        manifest_path = Path(attempt) / "run_manifest.json"
        manifest = _read_json(manifest_path, "run manifest")
        sequence = str(condition["sequence_id"])
        dataset = registration["datasets"][sequence]
        expected_identity = validate_attempt_registration_identity(
            manifest, condition, registration
        )
        resumed = _completed_ov_attempt(
            Path(attempt), registration_identity=expected_identity,
            expected_frames=int(manifest["expected_frames"]),
            expected_timestamps=_association_timestamps(Path(str(dataset["association"]))),
        )
        if resumed is None:
            raise ValueError("runner artifacts fail hash/telemetry revalidation")
        if expected_identity.get("formal_identity") != _formal_identity(condition, registration):
            raise ValueError("registration formal identity mismatch")
        metric_path = Path(attempt) / "metric_output.json"
        metric_payload = _read_json(metric_path, "metric output")
        if metric_payload != _metric_payload(Path(attempt), dataset):
            raise ValueError("metric output differs from deterministic replay")
        enriched = dict(manifest)
        telemetry = dict(enriched["telemetry"])
        telemetry["row_count"] = int(enriched["expected_frames"])
        enriched["telemetry"] = telemetry
        enriched["degraded"] = False
        enriched["metric"] = {
            "alignment": "SE3",
            "association_max_seconds": ASSOCIATION_MAX_SECONDS,
            "rpe_delta_seconds": RPE_DELTA_SECONDS,
            "output_sha256": sha256_file(metric_path),
        }
        result = validate_run_manifest(enriched, condition, registration)
        if not result.valid:
            raise ValueError(result.reason)
        return True, "valid", metric_payload
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc), None


def _finalize_metric(
    attempt: Path,
    condition: Mapping[str, object],
    registration: Mapping[str, object],
) -> tuple[bool, str]:
    manifest_path = Path(attempt) / "run_manifest.json"
    manifest = _read_json(manifest_path, "run manifest")
    dataset = registration["datasets"][str(condition["sequence_id"])]
    try:
        payload = _metric_payload(Path(attempt), dataset)
        metric_path = Path(attempt) / "metric_output.json"
        _write_json_atomic(metric_path, payload)
        telemetry = dict(manifest["telemetry"])
        telemetry["row_count"] = int(manifest["expected_frames"])
        manifest.update({
            "degraded": False,
            "telemetry": telemetry,
            "metric": {
                "alignment": "SE3",
                "association_max_seconds": ASSOCIATION_MAX_SECONDS,
                "rpe_delta_seconds": RPE_DELTA_SECONDS,
                "output_sha256": sha256_file(metric_path),
            },
        })
        _write_json_atomic(manifest_path, manifest)
        valid, reason, _ = validate_attempt(Path(attempt), condition, registration)
        if not valid:
            terminalize_failed_validation(Path(attempt), reason)
        return valid, reason
    except (KeyError, OSError, TypeError, ValueError) as exc:
        manifest.update({"state": "FAILED", "valid": False, "invalid_reason": f"metric validation failed: {exc}"})
        _write_json_atomic(manifest_path, manifest)
        return False, str(exc)


def collect_valid_attempts(
    run_root: Path,
    registration: Mapping[str, object],
    rows: list[dict[str, object]],
) -> tuple[dict[int, tuple[Path, dict[str, object]]], list[str]]:
    completed: dict[int, tuple[Path, dict[str, object]]] = {}
    errors: list[str] = []
    for condition in rows:
        condition_root = (
            Path(run_root) / str(condition["mode"]) / str(condition["sequence_id"])
            / f"seed-{condition['seed']}"
        )
        matches: list[tuple[Path, dict[str, object]]] = []
        for attempt in sorted(condition_root.glob("attempt-*")):
            valid, reason, metric = validate_attempt(attempt, condition, registration)
            if valid and metric is not None:
                matches.append((attempt, metric))
            elif (attempt / "run_manifest.json").is_file():
                errors.append(f"{attempt}: {reason}")
        if len(matches) > 1:
            raise ValueError(f"duplicate valid attempts for {condition['block_id']}:{condition['mode']}")
        if matches:
            completed[int(condition["order_index"])] = matches[0]
    return completed, errors


def execute_registered(run_root: Path) -> dict[str, object]:
    registration = _read_json(Path(run_root) / "study_registration.json", "study registration")
    validate_registration_current(registration)
    rows = read_run_matrix(Path(run_root) / "run_matrix.csv")
    completed, _ = collect_valid_attempts(run_root, registration, rows)
    registry = Path(run_root) / "run_registry.jsonl"
    for condition in rows:
        index = int(condition["order_index"])
        if index in completed:
            continue
        sequence = str(condition["sequence_id"])
        mode = str(condition["mode"])
        dataset = registration["datasets"][sequence]
        condition_root = (
            Path(run_root) / mode / sequence / f"seed-{condition['seed']}"
        )
        for attempt in sorted(condition_root.glob("attempt-*")):
            valid, reason, _ = validate_attempt(attempt, condition, registration)
            if not valid:
                terminalize_failed_validation(attempt, reason)
                _append_jsonl(registry, {
                    "kind": "p08_strict_validation",
                    "state": "FAILED",
                    "valid": False,
                    "run_dir": str(attempt),
                    "reason": reason,
                })
        cache_identity = dict(dataset["cache_identity"]) if mode == "semantic-feedback" else None
        result = run_ov_condition(
            RunCondition(
                sequence, int(condition["seed"]), Path(str(dataset["sequence_root"])),
                Path(str(dataset["settings"])), Path(str(dataset["dataset_manifest"])),
                Path(str(registration["experiment_manifest"])),
                Path(str(dataset["semantic_manifest"])), Path(str(dataset["prompt_config"])),
            ),
            mode=mode,
            executable=Path(str(registration["executable"])),
            vocabulary=Path(str(registration["vocabulary"])),
            output_root=Path(run_root) / mode,
            compatibility_commit=str(registration["compatibility_commit"]),
            producer_commit=str(registration["producer_commit"]),
            study=STUDY_ID,
            registry=registry,
            cache_root=Path(str(dataset["cache_root"])) if cache_identity else None,
            cache_identity=cache_identity,
            formal_identity=_formal_identity(condition, registration),
        )
        if not result.valid:
            return {"valid": False, "completed": len(completed), "run_dir": str(result.run_dir), "reason": result.invalid_reason}
        valid, reason = _finalize_metric(result.run_dir, condition, registration)
        if not valid:
            _append_jsonl(registry, {"kind": "p08_metric_validation", "state": "FAILED", "run_dir": str(result.run_dir), "reason": reason})
            return {"valid": False, "completed": len(completed), "run_dir": str(result.run_dir), "reason": reason}
        completed[index] = (result.run_dir, _read_json(result.run_dir / "metric_output.json", "metric output"))
        _append_jsonl(registry, {"kind": "p08_metric_validation", "state": "COMPLETED", "valid": True, "run_dir": str(result.run_dir), "order_index": index})
        print(f"P08_RUN {len(completed)}/60 {mode} {sequence} seed={condition['seed']}", flush=True)
    return {"valid": len(completed) == 60, "completed": len(completed)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-order", action="store_true")
    action.add_argument("--resume", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--executable", type=Path, default=REPOSITORY_ROOT / "Examples/RGB-D/rgbd_tum_ov")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data/tum/raw")
    parser.add_argument("--data-manifests", type=Path, default=REPOSITORY_ROOT / "data/tum/manifests")
    parser.add_argument("--dynamic-cache-root", type=Path, default=REPOSITORY_ROOT / "cache/dynamic/v1")
    parser.add_argument("--semantic-cache-root", type=Path, default=REPOSITORY_ROOT / "cache/semantic/v1")
    args = parser.parse_args()
    try:
        if args.freeze_order:
            registration = freeze_and_register(
                args.manifest, args.run_root, args.executable, args.data_root,
                args.data_manifests, args.dynamic_cache_root, args.semantic_cache_root,
            )
            print(f"P08_ORDER REGISTERED rows=60 sha256={registration['run_order_sha256']}")
        else:
            load_experiment_manifest(args.manifest)
            result = execute_registered(args.run_root)
            print(json.dumps(result, sort_keys=True))
            if not result["valid"]:
                return 1
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"P08_RUNNER INVALID: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
