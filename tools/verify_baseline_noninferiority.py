#!/usr/bin/env python3
"""Independently verify every deterministic and statistical P05 V2 gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from semantic_py.openvocab_slam.p05_noninferiority import (  # noqa: E402
    evaluate_study,
)
from semantic_py.openvocab_slam.p05_protocol import (  # noqa: E402
    SEQUENCE_IDS,
    STUDY_ID,
    expected_blocks,
    load_protocol,
    sha256_file,
    validate_batch_registration,
)
from tools.audit_p05_baseline_access import audit_trace  # noqa: E402
from tools.verify_baseline_equivalence import (  # noqa: E402
    compute_ate_rmse,
    parse_tum_trajectory,
)


CANDIDATE_TELEMETRY_FIELDS = [
    "frame_index", "timestamp", "tracking_state", "pose_valid",
    "tracking_time_seconds", "raw_keypoints", "used_keypoints",
    "removed_dynamic", "retained_uncertain", "removed_uncertain",
    "semantic_accessed", "semantic_state", "cache_load_seconds",
    "policy_seconds", "pacing_lateness_seconds",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _expected_conditions(protocol: Mapping[str, object]) -> list[dict[str, object]]:
    if protocol.get("blocks") != expected_blocks():
        raise ValueError("protocol blocks differ from the frozen order")
    conditions: list[dict[str, object]] = []
    for block in protocol["blocks"]:  # type: ignore[index]
        for implementation in block["execution_order"]:  # type: ignore[index]
            conditions.append({
                "block_id": block["block_id"],
                "sequence_id": block["sequence_id"],
                "repetition_id": block["repetition_id"],
                "implementation": implementation,
            })
    if len(conditions) != 180:
        raise ValueError("verifier expected exactly 180 formal conditions")
    return conditions


def _formal_identity(
    registration: Mapping[str, object],
    condition: Mapping[str, object],
) -> dict[str, object]:
    implementation = str(condition["implementation"])
    identity: dict[str, object] = {
        "study_id": STUDY_ID,
        "block_id": condition["block_id"],
        "implementation": implementation,
        "protocol_manifest_sha256": registration["protocol_manifest_sha256"],
    }
    if implementation == "oracle":
        identity["build_manifest_sha256"] = registration["oracle"][
            "build_manifest_sha256"
        ]
    else:
        identity["candidate_registration_commit"] = registration["candidate"][
            "producer_commit"
        ]
    return identity


def _completed_expected_attempt(
    condition_root: Path,
    registration: Mapping[str, object],
    condition: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    implementation = str(condition["implementation"])
    expected = registration[implementation]
    formal_identity = _formal_identity(registration, condition)
    matches: list[tuple[Path, dict[str, object]]] = []
    for attempt in sorted(condition_root.glob("attempt-*")):
        manifest_path = attempt / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path, "run manifest")
        executable = manifest.get("executable")
        if (
            manifest.get("state") == "COMPLETED"
            and manifest.get("valid") is True
            and manifest.get("study") == STUDY_ID
            and manifest.get("producer_commit") == expected["producer_commit"]
            and isinstance(executable, Mapping)
            and executable.get("sha256") == expected["executable_sha256"]
            and manifest.get("formal_identity") == formal_identity
        ):
            matches.append((attempt, manifest))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one valid attempt under {condition_root}, got {len(matches)}"
        )
    return matches[0]


def _artifact_path(
    run_dir: Path,
    manifest: Mapping[str, object],
    key: str,
) -> Path:
    identity = manifest.get(key)
    if not isinstance(identity, Mapping):
        raise ValueError(f"run manifest lacks {key} artifact identity")
    relative = identity.get("path")
    expected_hash = identity.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError(f"run {key} artifact identity is incomplete")
    path = (run_dir / relative).resolve()
    if run_dir.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"run {key} artifact is missing or outside attempt")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"run artifact hash mismatch for {key}: {run_dir}")
    size = identity.get("size_bytes")
    if size is not None and int(size) != path.stat().st_size:
        raise ValueError(f"run artifact size mismatch for {key}: {run_dir}")
    return path


def _association_timestamps(path: Path) -> list[float]:
    values: list[float] = []
    previous = -math.inf
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid association row {path}:{line_number}")
        value = float(fields[0])
        if not math.isfinite(value) or value <= previous:
            raise ValueError(f"invalid association timestamp {path}:{line_number}")
        previous = value
        values.append(value)
    if not values:
        raise ValueError(f"association contains no frames: {path}")
    return values


def _oracle_pose_fraction(
    path: Path,
    timestamps: list[float],
    timestamp_tolerance_seconds: float,
) -> float:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid oracle telemetry JSON: {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError("oracle telemetry row is not an object")
        required = {
            "frame_index", "timestamp", "tracking_state", "pose_valid",
            "tracking_time_seconds",
        }
        if not required <= row.keys() or row["frame_index"] != len(rows):
            raise ValueError("oracle telemetry is incomplete or non-contiguous")
        timestamp = float(row["timestamp"])
        if (
            not math.isfinite(timestamp)
            or abs(timestamp - timestamps[len(rows)]) > timestamp_tolerance_seconds
        ):
            raise ValueError("oracle telemetry timestamp differs from association")
        tracking_time = float(row["tracking_time_seconds"])
        if not math.isfinite(tracking_time) or tracking_time < 0:
            raise ValueError("oracle telemetry has invalid timing")
        if not isinstance(row["pose_valid"], bool):
            raise ValueError("oracle telemetry pose flag is not boolean")
        rows.append(row)
    if len(rows) != len(timestamps):
        raise ValueError("oracle telemetry frame coverage mismatch")
    return sum(bool(row["pose_valid"]) for row in rows) / len(rows)


def _canonical_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"candidate telemetry has invalid {label}") from exc
    if str(parsed) != value:
        raise ValueError(f"candidate telemetry has non-canonical {label}")
    return parsed


def _candidate_pose_fraction(path: Path, timestamps: list[float]) -> float:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CANDIDATE_TELEMETRY_FIELDS:
            raise ValueError("candidate telemetry CSV header mismatch")
        rows = list(reader)
    if len(rows) != len(timestamps):
        raise ValueError("candidate telemetry frame coverage mismatch")
    valid_pose = 0
    for index, row in enumerate(rows):
        if _canonical_int(row["frame_index"], "frame index") != index:
            raise ValueError("candidate telemetry frame index is non-contiguous")
        if float(row["timestamp"]) != timestamps[index]:
            raise ValueError("candidate telemetry timestamp differs from association")
        if row["tracking_state"] not in {"-1", "0", "1", "2", "3"}:
            raise ValueError("candidate telemetry tracking state is invalid")
        if row["pose_valid"] not in {"0", "1"}:
            raise ValueError("candidate telemetry pose flag is invalid")
        valid_pose += int(row["pose_valid"])
        raw = _canonical_int(row["raw_keypoints"], "raw keypoints")
        used = _canonical_int(row["used_keypoints"], "used keypoints")
        removals = [
            _canonical_int(row[name], name) for name in (
                "removed_dynamic", "retained_uncertain", "removed_uncertain"
            )
        ]
        timings = [
            float(row[name]) for name in (
                "tracking_time_seconds", "cache_load_seconds", "policy_seconds",
                "pacing_lateness_seconds",
            )
        ]
        if not all(math.isfinite(value) and value >= 0 for value in timings):
            raise ValueError("candidate telemetry timing is invalid")
        if (
            row["semantic_accessed"] != "0"
            or row["semantic_state"] != "BASELINE"
            or raw != used
            or removals != [0, 0, 0]
            or timings[1] != 0.0
            or timings[2] != 0.0
        ):
            raise ValueError(f"baseline accessed semantic state: {path}")
    return valid_pose / len(rows)


def _validate_candidate_sidecars(
    run_dir: Path,
    manifest: Mapping[str, object],
    expected_frames: int,
) -> None:
    final_state = _read_json(
        _artifact_path(run_dir, manifest, "final_state"), "candidate final state"
    )
    timings = _read_json(
        _artifact_path(run_dir, manifest, "timings"), "candidate timings"
    )
    if (
        final_state.get("state") != "COMPLETED"
        or final_state.get("mode") != "baseline"
        or int(final_state.get("frame_count", -1)) != expected_frames
        or int(timings.get("frame_count", -1)) != expected_frames
    ):
        raise ValueError("candidate final state or timings are incomplete")
    for name in (
        "mean_tracking_seconds", "median_tracking_seconds",
        "mean_pacing_lateness_seconds", "max_pacing_lateness_seconds",
        "wall_seconds",
    ):
        value = float(timings[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"candidate timing aggregate is invalid: {name}")


def _measure_run(
    run_dir: Path,
    manifest: Mapping[str, object],
    registration: Mapping[str, object],
    condition: Mapping[str, object],
    oracle_timestamp_tolerance_seconds: float,
) -> dict[str, object]:
    sequence_id = str(condition["sequence_id"])
    repetition_id = int(condition["repetition_id"])
    implementation = str(condition["implementation"])
    dataset = registration["datasets"][sequence_id]
    if (
        manifest.get("mode") != "baseline"
        or manifest.get("sequence_id") != sequence_id
        or int(manifest.get("seed", -1)) != repetition_id
        or manifest.get("compatibility_commit") != registration["compatibility_commit"]
        or manifest.get("association_sha256") != dataset["association_sha256"]
        or manifest.get("dataset_manifest_sha256") != dataset["dataset_manifest_sha256"]
        or manifest.get("settings", {}).get("sha256") != dataset["settings_sha256"]
        or manifest.get("vocabulary", {}).get("sha256")
        != registration["vocabulary"]["sha256"]
    ):
        raise ValueError(f"run deterministic identity mismatch: {run_dir}")
    if int(manifest.get("expected_frames", -1)) != int(dataset["expected_frames"]):
        raise ValueError(f"run expected-frame identity mismatch: {run_dir}")

    trajectory_path = _artifact_path(run_dir, manifest, "trajectory")
    keyframe_path = _artifact_path(run_dir, manifest, "keyframe_trajectory")
    telemetry_path = _artifact_path(run_dir, manifest, "telemetry")
    _artifact_path(run_dir, manifest, "stdout")
    _artifact_path(run_dir, manifest, "stderr")
    trajectory = parse_tum_trajectory(trajectory_path)
    parse_tum_trajectory(keyframe_path)
    groundtruth = parse_tum_trajectory(Path(str(dataset["groundtruth"])))
    ate, associated = compute_ate_rmse(
        trajectory, groundtruth, max_difference=0.02
    )
    if not math.isfinite(ate) or ate <= 0:
        raise ValueError(f"ATE must be finite and strictly positive: {run_dir}")
    timestamps = _association_timestamps(Path(str(dataset["association"])))

    if implementation == "candidate":
        verified = manifest.get("verified_inputs")
        registration_identity = manifest.get("registration_identity")
        if (
            manifest.get("cache_root") is not None
            or manifest.get("cache_identity") is not None
            or manifest.get("pacing") != "dataset_timestamp_paced_relative"
            or not isinstance(verified, Mapping)
            or verified.get("source_tree_sha256") != dataset["source_tree_sha256"]
            or verified.get("dataset_manifest_sha256")
            != dataset["dataset_manifest_sha256"]
            or not isinstance(registration_identity, Mapping)
            or registration_identity.get("formal_identity")
            != _formal_identity(registration, condition)
            or registration_identity.get("experiment_manifest_sha256")
            != registration["experiment_manifest"]["sha256"]
            or registration_identity.get("source_tree_sha256")
            != dataset["source_tree_sha256"]
            or registration_identity.get("cache_root") is not None
            or registration_identity.get("cache_identity") is not None
        ):
            raise ValueError(f"candidate baseline registration identity mismatch: {run_dir}")
        valid_fraction = _candidate_pose_fraction(telemetry_path, timestamps)
        _validate_candidate_sidecars(run_dir, manifest, len(timestamps))
    else:
        valid_fraction = _oracle_pose_fraction(
            telemetry_path,
            timestamps,
            oracle_timestamp_tolerance_seconds,
        )
    return {
        "sequence_id": sequence_id,
        "repetition_id": repetition_id,
        "implementation": implementation,
        "producer_commit": manifest["producer_commit"],
        "executable_sha256": manifest["executable"]["sha256"],
        "run_dir": str(run_dir.resolve()),
        "valid_pose_fraction": valid_fraction,
        "ate_rmse_m": ate,
        "associated_poses": associated,
    }


def _validate_registration_inputs(
    protocol: Mapping[str, object],
    protocol_path: Path,
    experiment_path: Path,
    registration: Mapping[str, object],
    repository_root: Path,
    data_root: Path,
    data_manifest_root: Path,
) -> None:
    protocol_hash = sha256_file(protocol_path)
    validate_batch_registration(registration, protocol, protocol_hash)
    if (
        registration["experiment_manifest"]["path"] != str(experiment_path.resolve())
        or registration["experiment_manifest"]["sha256"] != sha256_file(experiment_path)
    ):
        raise ValueError("registration experiment identity mismatch")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository_root, text=True,
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    if dirty:
        raise ValueError("candidate repository is dirty during verification")
    if head != registration["candidate"]["producer_commit"]:
        raise ValueError("candidate repository HEAD differs from registration")
    for implementation in ("oracle", "candidate"):
        executable = Path(str(registration[implementation]["executable"]))
        if (
            not executable.is_file()
            or sha256_file(executable)
            != registration[implementation]["executable_sha256"]
        ):
            raise ValueError(f"{implementation} executable identity mismatch")
    oracle_build = Path(str(registration["oracle"]["build_manifest"]))
    if sha256_file(oracle_build) != registration["oracle"]["build_manifest_sha256"]:
        raise ValueError("oracle build-manifest identity mismatch")
    vocabulary = Path(str(registration["vocabulary"]["path"]))
    if sha256_file(vocabulary) != registration["vocabulary"]["sha256"]:
        raise ValueError("vocabulary identity mismatch")

    for sequence_id in SEQUENCE_IDS:
        dataset = registration["datasets"][sequence_id]
        expected_manifest = (Path(data_manifest_root) / f"{sequence_id}.json").resolve()
        if Path(str(dataset["dataset_manifest"])).resolve() != expected_manifest:
            raise ValueError(f"dataset manifest path mismatch: {sequence_id}")
        manifest = _read_json(expected_manifest, "dataset manifest")
        if (
            sha256_file(expected_manifest) != dataset["dataset_manifest_sha256"]
            or manifest.get("extracted_tree_sha256") != dataset["source_tree_sha256"]
            or not Path(str(dataset["sequence_root"])).resolve().is_relative_to(
                Path(data_root).resolve()
            )
            or sha256_file(Path(str(dataset["association"])))
            != dataset["association_sha256"]
            or sha256_file(Path(str(dataset["groundtruth"])))
            != dataset["groundtruth_sha256"]
            or sha256_file(Path(str(dataset["settings"])))
            != dataset["settings_sha256"]
        ):
            raise ValueError(f"dataset frozen identity mismatch: {sequence_id}")


def _validate_audits(
    audit_root: Path,
    registration: Mapping[str, object],
    repository_root: Path,
) -> dict[str, object]:
    result: dict[str, object] = {}
    forbidden_roots = [
        (Path(repository_root) / "cache/semantic").resolve(),
        (Path(repository_root) / "cache/dynamic").resolve(),
    ]
    forbidden_files = [
        (Path(repository_root) / "config/PROMPTS.yaml").resolve(),
        (Path(repository_root) / "config/SEMANTIC_MODELS.json").resolve(),
        (Path(repository_root) / "config/DYNAMIC_CACHE_IDENTITY.json").resolve(),
    ]
    for sequence_id in SEQUENCE_IDS:
        report_path = Path(audit_root) / sequence_id / "audit_report.json"
        report = _read_json(report_path, "candidate access audit report")
        if (
            report.get("schema_version") != 1
            or report.get("valid") is not True
            or report.get("study_id") != STUDY_ID
            or report.get("sequence_id") != sequence_id
            or report.get("protocol_manifest_sha256")
            != registration["protocol_manifest_sha256"]
            or report.get("candidate_producer_commit")
            != registration["candidate"]["producer_commit"]
            or report.get("candidate_executable_sha256")
            != registration["candidate"]["executable_sha256"]
            or report.get("forbidden_accesses") != []
        ):
            raise ValueError(f"candidate access audit hard gate failed: {sequence_id}")
        traces = report.get("trace_files")
        if not isinstance(traces, list) or not traces:
            raise ValueError(f"candidate access audit trace identity missing: {sequence_id}")
        trace_paths: list[Path] = []
        for identity in traces:
            if not isinstance(identity, Mapping):
                raise ValueError("candidate access audit trace identity is invalid")
            path = Path(str(identity.get("path")))
            if (
                not path.is_file()
                or sha256_file(path) != identity.get("sha256")
                or path.stat().st_size != int(identity.get("size_bytes", -1))
            ):
                raise ValueError(f"candidate access audit trace changed: {sequence_id}")
            trace_paths.append(path)
        cwd = report.get("cwd")
        if not isinstance(cwd, str):
            raise ValueError(f"candidate access audit cwd is missing: {sequence_id}")
        if report.get("forbidden_roots") != [str(path) for path in forbidden_roots]:
            raise ValueError(f"candidate access audit root scope mismatch: {sequence_id}")
        if report.get("forbidden_files") != sorted(str(path) for path in forbidden_files):
            raise ValueError(f"candidate access audit file scope mismatch: {sequence_id}")
        audit_trace(
            trace_paths,
            forbidden_roots,
            forbidden_files,
            cwd=Path(cwd),
        )
        result[sequence_id] = {
            "report_path": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "trace_files": traces,
        }
    return result


def build_report(
    protocol_path: Path,
    experiment_path: Path,
    registration_path: Path,
    oracle_root: Path,
    candidate_root: Path,
    audit_root: Path,
    data_root: Path,
    data_manifest_root: Path,
    repository_root: Path,
) -> dict[str, object]:
    protocol_path = Path(protocol_path).resolve()
    experiment_path = Path(experiment_path).resolve()
    repository_root = Path(repository_root).resolve()
    protocol = load_protocol(protocol_path, experiment_path)
    registration = _read_json(registration_path, "batch registration")
    _validate_registration_inputs(
        protocol, protocol_path, experiment_path, registration,
        repository_root, data_root, data_manifest_root,
    )
    audits = _validate_audits(audit_root, registration, repository_root)
    oracle_timestamp_tolerance_seconds = float(
        protocol["metrics"]["oracle_telemetry_timestamp_tolerance_seconds"]
    )

    measured: list[dict[str, object]] = []
    grouped: dict[str, tuple[list[dict[str, object]], list[dict[str, object]]]] = {
        sequence_id: ([], []) for sequence_id in SEQUENCE_IDS
    }
    for condition in _expected_conditions(protocol):
        implementation = str(condition["implementation"])
        root = Path(oracle_root) if implementation == "oracle" else Path(candidate_root)
        condition_root = (
            root / str(condition["sequence_id"])
            / f"seed-{condition['repetition_id']}"
        )
        run_dir, manifest = _completed_expected_attempt(
            condition_root, registration, condition
        )
        row = _measure_run(
            run_dir,
            manifest,
            registration,
            condition,
            oracle_timestamp_tolerance_seconds,
        )
        measured.append(row)
        target = grouped[str(condition["sequence_id"])][
            0 if implementation == "oracle" else 1
        ]
        target.append(row)

    study = evaluate_study(
        grouped,
        SEQUENCE_IDS,
        protocol["statistics"],  # type: ignore[arg-type]
    )
    failed_sequences = [
        sequence_id for sequence_id, value in study["sequences"].items()
        if not value["valid"]
    ]
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "valid": bool(study["valid"]),
        "generated_utc": _utc_now(),
        "verifier_sha256": sha256_file(Path(__file__)),
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "registration_path": str(Path(registration_path).resolve()),
        "registration_sha256": sha256_file(Path(registration_path)),
        "oracle_producer_commit": registration["oracle"]["producer_commit"],
        "candidate_producer_commit": registration["candidate"]["producer_commit"],
        "deterministic_gates": {
            "valid": True,
            "expected_run_count": 180,
            "measured_run_count": len(measured),
            "paired_block_count": 90,
            "access_audit_count": len(audits),
            "candidate_no_semantic_access": True,
            "artifact_hashes_valid": True,
            "producer_identities_valid": True,
        },
        "statistics": {
            **protocol["statistics"],  # type: ignore[arg-type]
            "numpy_version": next(iter(study["sequences"].values()))[
                "bootstrap"
            ]["numpy_version"],
        },
        "sequences": study["sequences"],
        "failed_sequences": failed_sequences,
        "runs": measured,
        "access_audits": audits,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifests", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    partial = args.output.parent / f".{args.output.name}.partial"
    partial.unlink(missing_ok=True)
    try:
        report = build_report(
            args.protocol,
            args.experiment_manifest,
            args.registration,
            args.oracle_root,
            args.candidate_root,
            args.audit_root,
            args.data_root,
            args.data_manifests,
            args.repository,
        )
        _write_json_atomic(args.output, report)
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        partial.unlink(missing_ok=True)
        print(f"P05_NONINFERIORITY_VERIFY_ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "valid": report["valid"],
        "output": str(args.output.resolve()),
        "failed_sequences": report["failed_sequences"],
    }, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
