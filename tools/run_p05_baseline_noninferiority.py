#!/usr/bin/env python3
"""Register, audit, and execute the frozen P05 V2 baseline matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from semantic_py.openvocab_slam.config import (  # noqa: E402
    FORMAL_BASELINE_COMPATIBILITY_COMMIT,
)
from semantic_py.openvocab_slam.p05_protocol import (  # noqa: E402
    CANDIDATE_POLICY,
    ORACLE_COMMIT,
    SEQUENCE_IDS,
    STUDY_ID,
    expected_blocks,
    load_protocol,
    sha256_file,
    validate_batch_registration,
)
from tools.audit_p05_baseline_access import audit_trace  # noqa: E402
from tools.run_orb_tum import (  # noqa: E402
    RunCondition,
    RunResult,
    _association_timestamps,
    _validate_ov_telemetry,
    _validate_telemetry,
    parse_trajectory,
    run_baseline_condition,
    run_ov_condition,
)


ConditionRunner = Callable[..., RunResult]


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


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


def _association_count(path: Path) -> int:
    count = sum(
        1 for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if count <= 0:
        raise ValueError(f"association file has no frames: {path}")
    return count


def expected_conditions(protocol: Mapping[str, object]) -> list[dict[str, object]]:
    if protocol.get("blocks") != expected_blocks():
        raise ValueError("protocol blocks differ from frozen order")
    result: list[dict[str, object]] = []
    for raw_block in protocol["blocks"]:  # type: ignore[index]
        if not isinstance(raw_block, Mapping):
            raise ValueError("protocol block is not an object")
        for implementation in raw_block["execution_order"]:  # type: ignore[index]
            result.append(
                {
                    "block_id": raw_block["block_id"],
                    "sequence_id": raw_block["sequence_id"],
                    "repetition_id": raw_block["repetition_id"],
                    "implementation": implementation,
                }
            )
    if len(result) != 180:
        raise ValueError("formal matrix must contain exactly 180 conditions")
    return result


def _validated_oracle_build(path: Path) -> dict[str, object]:
    manifest = _read_json(path, "oracle build manifest")
    executable = manifest.get("executable")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("state") != "COMPLETED"
        or manifest.get("source_commit") != ORACLE_COMMIT
        or manifest.get("worktree_clean") is not True
        or not isinstance(executable, Mapping)
    ):
        raise ValueError("oracle build manifest has wrong frozen identity")
    executable_path = Path(str(executable.get("path"))).resolve()
    if not executable_path.is_file():
        raise ValueError("oracle executable is missing")
    if executable.get("sha256") != sha256_file(executable_path):
        raise ValueError("oracle executable differs from build manifest")
    return manifest


def _build_registration(
    protocol: Mapping[str, object],
    protocol_path: Path,
    repository_root: Path,
    oracle_build_manifest_path: Path,
    candidate_executable: Path,
    *,
    registered_utc: str,
) -> dict[str, object]:
    repository_root = Path(repository_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    experiment_path = repository_root / str(protocol["experiment_manifest"])
    loaded = load_protocol(protocol_path, experiment_path)
    if loaded != dict(protocol):
        raise ValueError("supplied protocol differs from tracked protocol manifest")
    dirty = _git_output(repository_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"candidate tracked worktree is dirty: {dirty}")
    candidate_commit = _git_output(repository_root, "rev-parse", "HEAD")
    if len(candidate_commit) != 40:
        raise ValueError("candidate HEAD is not a full commit identity")

    candidate_executable = Path(candidate_executable).resolve()
    if not candidate_executable.is_file():
        raise ValueError("candidate executable is missing")
    oracle_build_manifest_path = Path(oracle_build_manifest_path).resolve()
    oracle_build = _validated_oracle_build(oracle_build_manifest_path)
    oracle_executable = Path(str(oracle_build["executable"]["path"])).resolve()
    experiment = _read_json(experiment_path, "experiment manifest")
    vocabulary = (repository_root / "Vocabulary/ORBvoc.txt").resolve()
    if not vocabulary.is_file():
        raise ValueError("vocabulary is missing")

    datasets: dict[str, object] = {}
    raw_datasets = experiment.get("datasets")
    if not isinstance(raw_datasets, list):
        raise ValueError("experiment datasets are missing")
    for item in raw_datasets:
        if not isinstance(item, Mapping):
            raise ValueError("experiment dataset row is invalid")
        sequence_id = str(item["id"])
        archive = str(item["archive"])
        sequence_root = (
            repository_root / "data/tum/raw" / archive.removesuffix(".tgz")
        ).resolve()
        association = sequence_root / "associate.txt"
        dataset_manifest_path = (
            repository_root / "data/tum/manifests" / f"{sequence_id}.json"
        ).resolve()
        dataset_manifest = _read_json(dataset_manifest_path, "dataset manifest")
        settings = (
            repository_root / "Examples/RGB-D" / str(item["settings"])
        ).resolve()
        groundtruth = sequence_root / "groundtruth.txt"
        if not association.is_file() or not settings.is_file() or not groundtruth.is_file():
            raise ValueError(f"dataset inputs are incomplete for {sequence_id}")
        association_sha = sha256_file(association)
        source_tree = dataset_manifest.get("extracted_tree_sha256")
        if (
            dataset_manifest.get("schema_version") != 1
            or dataset_manifest.get("sequence_id") != sequence_id
            or dataset_manifest.get("association_sha256") != association_sha
            or not isinstance(source_tree, str)
            or len(source_tree) != 64
        ):
            raise ValueError(f"dataset manifest identity mismatch for {sequence_id}")
        datasets[sequence_id] = {
            "sequence_root": str(sequence_root),
            "dataset_manifest": str(dataset_manifest_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "source_tree_sha256": source_tree,
            "association": str(association),
            "association_sha256": association_sha,
            "groundtruth": str(groundtruth),
            "groundtruth_sha256": sha256_file(groundtruth),
            "settings": str(settings),
            "settings_sha256": sha256_file(settings),
            "expected_frames": _association_count(association),
        }
    if tuple(datasets) != SEQUENCE_IDS:
        raise ValueError("registered datasets differ from frozen sequence order")

    conditions = expected_conditions(protocol)
    return {
        "schema_version": 1,
        "state": "REGISTERED",
        "registered_utc": registered_utc,
        "study_id": STUDY_ID,
        "candidate_policy": CANDIDATE_POLICY,
        "protocol_manifest": str(protocol_path),
        "protocol_manifest_sha256": sha256_file(protocol_path),
        "experiment_manifest": {
            "path": str(experiment_path.resolve()),
            "sha256": sha256_file(experiment_path),
        },
        "repository": str(repository_root),
        "compatibility_commit": FORMAL_BASELINE_COMPATIBILITY_COMMIT,
        "vocabulary": {
            "path": str(vocabulary),
            "sha256": sha256_file(vocabulary),
        },
        "oracle": {
            "producer_commit": ORACLE_COMMIT,
            "executable": str(oracle_executable),
            "executable_sha256": sha256_file(oracle_executable),
            "build_manifest": str(oracle_build_manifest_path),
            "build_manifest_sha256": sha256_file(oracle_build_manifest_path),
        },
        "candidate": {
            "producer_commit": candidate_commit,
            "executable": str(candidate_executable),
            "executable_sha256": sha256_file(candidate_executable),
        },
        "datasets": datasets,
        "matrix": {
            "paired_block_count": 90,
            "condition_count": 180,
            "execution": "sequential",
            "conditions": conditions,
        },
        "runner_sha256": sha256_file(Path(__file__)),
    }


def _without_time(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("registered_utc", None)
    return result


def register_batch(
    protocol: Mapping[str, object],
    protocol_path: Path,
    repository_root: Path,
    oracle_build_manifest_path: Path,
    candidate_executable: Path,
    registration_path: Path,
    registry_path: Path,
) -> dict[str, object]:
    registration_path = Path(registration_path)
    existing: dict[str, object] | None = None
    registered_utc = _utc_now()
    if registration_path.is_file():
        existing = _read_json(registration_path, "batch registration")
        registered_utc = str(existing.get("registered_utc"))
    fresh = _build_registration(
        protocol,
        protocol_path,
        repository_root,
        oracle_build_manifest_path,
        candidate_executable,
        registered_utc=registered_utc,
    )
    validate_batch_registration(
        fresh, protocol, sha256_file(Path(protocol_path))
    )
    if existing is not None:
        validate_batch_registration(
            existing, protocol, sha256_file(Path(protocol_path))
        )
        if _without_time(existing) != _without_time(fresh):
            raise ValueError("existing batch registration differs from current identities")
        return existing
    if any(registration_path.parent.glob("**/attempt-*")):
        raise ValueError("formal attempt exists before immutable batch registration")
    _write_json_atomic(registration_path, fresh)
    _append_jsonl(
        Path(registry_path),
        {
            "schema_version": 1,
            "kind": "p05_baseline_noninferiority_v2_batch",
            "state": "REGISTERED",
            "study_id": STUDY_ID,
            "registered_utc": registered_utc,
            "registration_path": str(registration_path.resolve()),
            "registration_sha256": sha256_file(registration_path),
            "candidate_producer_commit": fresh["candidate"]["producer_commit"],
            "oracle_producer_commit": ORACLE_COMMIT,
            "expected_runs": 180,
        },
    )
    return fresh


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


def _artifact_path(
    attempt: Path,
    manifest: Mapping[str, object],
    key: str,
) -> Path:
    identity = manifest.get(key)
    if not isinstance(identity, Mapping):
        raise ValueError(f"run manifest lacks {key} artifact identity")
    raw_path = identity.get("path")
    expected_hash = identity.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"run manifest has invalid {key} artifact identity")
    path = (attempt / raw_path).resolve()
    if attempt.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"run {key} artifact is missing or outside attempt")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"run {key} artifact hash mismatch")
    return path


def _attempt_artifacts_valid(
    attempt: Path,
    manifest: Mapping[str, object],
    implementation: str,
    dataset: Mapping[str, object],
) -> bool:
    try:
        trajectory = _artifact_path(attempt, manifest, "trajectory")
        keyframe = _artifact_path(attempt, manifest, "keyframe_trajectory")
        telemetry = _artifact_path(attempt, manifest, "telemetry")
        _artifact_path(attempt, manifest, "stdout")
        _artifact_path(attempt, manifest, "stderr")
        parse_trajectory(trajectory)
        parse_trajectory(keyframe)
        expected_frames = int(dataset["expected_frames"])
        if int(manifest.get("expected_frames", -1)) != expected_frames:
            raise ValueError("run expected-frame identity mismatch")
        if implementation == "oracle":
            _validate_telemetry(telemetry, expected_frames)
        else:
            if (
                manifest.get("mode") != "baseline"
                or manifest.get("cache_root") is not None
                or manifest.get("cache_identity") is not None
            ):
                raise ValueError("candidate baseline registered semantic cache assets")
            timestamps = _association_timestamps(Path(str(dataset["association"])))
            _validate_ov_telemetry(telemetry, expected_frames, "baseline", timestamps)
            final_state_path = _artifact_path(attempt, manifest, "final_state")
            timings_path = _artifact_path(attempt, manifest, "timings")
            final_state = _read_json(final_state_path, "run final state")
            timings = _read_json(timings_path, "run timings")
            if (
                final_state.get("state") != "COMPLETED"
                or final_state.get("mode") != "baseline"
                or int(final_state.get("frame_count", -1)) != expected_frames
                or int(timings.get("frame_count", -1)) != expected_frames
            ):
                raise ValueError("candidate final state or timings are incomplete")
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def validate_resume(
    run_root: Path,
    registration: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, Path]:
    validate_batch_registration(
        registration, protocol, str(registration["protocol_manifest_sha256"])
    )
    completed: dict[str, Path] = {}
    for condition in expected_conditions(protocol):
        implementation = str(condition["implementation"])
        expected_producer = registration[implementation]["producer_commit"]
        expected_executable = registration[implementation]["executable_sha256"]
        expected_formal = _formal_identity(registration, condition)
        dataset = registration["datasets"][str(condition["sequence_id"])]
        condition_root = (
            Path(run_root) / implementation / str(condition["sequence_id"])
            / f"seed-{condition['repetition_id']}"
        )
        matches: list[Path] = []
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
                and manifest.get("producer_commit") == expected_producer
                and isinstance(executable, Mapping)
                and executable.get("sha256") == expected_executable
                and manifest.get("formal_identity") == expected_formal
                and _attempt_artifacts_valid(
                    attempt, manifest, implementation, dataset
                )
            ):
                matches.append(attempt)
        if len(matches) > 1:
            raise ValueError(
                f"duplicate valid attempts for {implementation}:{condition['block_id']}"
            )
        if matches:
            completed[f"{implementation}:{condition['block_id']}"] = matches[0]
    return completed


def validate_audit_reports(
    audit_root: Path,
    registration: Mapping[str, object],
) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    candidate = registration["candidate"]
    for sequence_id in SEQUENCE_IDS:
        path = Path(audit_root) / sequence_id / "audit_report.json"
        if not path.is_file():
            raise ValueError(f"candidate access audit report is missing: {sequence_id}")
        report = _read_json(path, "candidate access audit report")
        if (
            report.get("schema_version") != 1
            or report.get("valid") is not True
            or report.get("study_id") != STUDY_ID
            or report.get("sequence_id") != sequence_id
            or report.get("protocol_manifest_sha256")
            != registration["protocol_manifest_sha256"]
            or report.get("candidate_producer_commit")
            != candidate["producer_commit"]
            or report.get("candidate_executable_sha256")
            != candidate["executable_sha256"]
            or report.get("forbidden_accesses") != []
        ):
            raise ValueError(f"candidate access audit report identity mismatch: {sequence_id}")
        reports[sequence_id] = path
    return reports


def execute_matrix(
    protocol: Mapping[str, object],
    registration: Mapping[str, object],
    run_root: Path,
    registry_path: Path,
    *,
    oracle_runner: ConditionRunner = run_baseline_condition,
    candidate_runner: ConditionRunner = run_ov_condition,
) -> dict[str, object]:
    completed = validate_resume(run_root, registration, protocol)
    completed_count = len(completed)
    vocabulary = Path(str(registration["vocabulary"]["path"]))
    for condition_value in expected_conditions(protocol):
        key = f"{condition_value['implementation']}:{condition_value['block_id']}"
        if key in completed:
            continue
        sequence_id = str(condition_value["sequence_id"])
        repetition_id = int(condition_value["repetition_id"])
        dataset = registration["datasets"][sequence_id]
        run_condition = RunCondition(
            sequence_id,
            repetition_id,
            Path(str(dataset["sequence_root"])),
            Path(str(dataset["settings"])),
            Path(str(dataset["dataset_manifest"])),
            Path(str(registration["experiment_manifest"]["path"])),
        )
        formal_identity = _formal_identity(registration, condition_value)
        implementation = str(condition_value["implementation"])
        if implementation == "oracle":
            result = oracle_runner(
                run_condition,
                executable=Path(str(registration["oracle"]["executable"])),
                vocabulary=vocabulary,
                output_root=Path(run_root) / "oracle",
                compatibility_commit=str(registration["compatibility_commit"]),
                producer_commit=str(registration["oracle"]["producer_commit"]),
                registry=Path(registry_path),
                study=STUDY_ID,
                formal_identity=formal_identity,
            )
        else:
            result = candidate_runner(
                run_condition,
                mode="baseline",
                executable=Path(str(registration["candidate"]["executable"])),
                vocabulary=vocabulary,
                output_root=Path(run_root) / "candidate",
                compatibility_commit=str(registration["compatibility_commit"]),
                producer_commit=str(registration["candidate"]["producer_commit"]),
                registry=Path(registry_path),
                study=STUDY_ID,
                cache_root=None,
                cache_identity=None,
                formal_identity=formal_identity,
            )
        if not result.valid:
            return {
                "expected": 180,
                "completed": completed_count,
                "valid": False,
                "invalid_reason": result.invalid_reason,
                "run_dir": str(result.run_dir),
            }
        completed_count += 1
    return {"expected": 180, "completed": completed_count, "valid": True}


def _run_access_audits(
    registration: Mapping[str, object],
    audit_root: Path,
    registry_path: Path,
    forbidden_roots: list[Path],
    forbidden_files: list[Path],
) -> dict[str, Path]:
    strace = shutil.which("strace")
    if strace is None:
        raise ValueError("strace is required for P05 access audits")
    candidate = registration["candidate"]
    vocabulary = str(registration["vocabulary"]["path"])
    for sequence_id in SEQUENCE_IDS:
        canonical_report = Path(audit_root) / sequence_id / "audit_report.json"
        if canonical_report.is_file():
            continue
        dataset = registration["datasets"][sequence_id]
        attempt_root = canonical_report.parent
        attempts = sorted(attempt_root.glob("attempt-*"))
        run_dir = attempt_root / f"attempt-{len(attempts) + 1:03d}"
        run_dir.mkdir(parents=True, exist_ok=False)
        trace_prefix = run_dir / "trace"
        command = [
            strace, "-ff", "-e", "trace=file", "-o", str(trace_prefix),
            str(candidate["executable"]), vocabulary, str(dataset["settings"]),
            str(dataset["sequence_root"]), str(dataset["association"]),
            "baseline", sequence_id, "23011",
        ]
        registered = {
            "schema_version": 1,
            "kind": "p05_candidate_baseline_access_audit",
            "state": "REGISTERED",
            "study_id": STUDY_ID,
            "sequence_id": sequence_id,
            "candidate_producer_commit": candidate["producer_commit"],
            "candidate_executable_sha256": candidate["executable_sha256"],
            "protocol_manifest_sha256": registration["protocol_manifest_sha256"],
            "command": command,
            "cwd": str(run_dir.resolve()),
            "registered_utc": _utc_now(),
        }
        _write_json_atomic(run_dir / "audit_manifest.json", registered)
        _append_jsonl(Path(registry_path), registered)
        environment = os.environ.copy()
        environment["ORB_SLAM2_RUN_SEED"] = "23011"
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
            run_dir / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command, cwd=run_dir, env=environment, stdout=stdout, stderr=stderr,
                check=False,
            )
        if completed.returncode != 0:
            failed = dict(registered, state="FAILED", valid=False,
                          exit_code=completed.returncode, completed_utc=_utc_now())
            _write_json_atomic(run_dir / "audit_manifest.json", failed)
            _append_jsonl(Path(registry_path), failed)
            raise ValueError(f"candidate access probe failed for {sequence_id}")
        trace_paths = sorted(run_dir.glob("trace*"))
        access = audit_trace(
            trace_paths, forbidden_roots, forbidden_files, cwd=run_dir
        )
        timestamps = _association_timestamps(Path(str(dataset["association"])))
        _validate_ov_telemetry(
            run_dir / "frame_telemetry.csv", int(dataset["expected_frames"]),
            "baseline", timestamps,
        )
        parse_trajectory(run_dir / "CameraTrajectory.txt")
        parse_trajectory(run_dir / "KeyFrameTrajectory.txt")
        report = {
            **registered,
            "state": "COMPLETED",
            "valid": True,
            "completed_utc": _utc_now(),
            "exit_code": 0,
            "forbidden_roots": access["forbidden_roots"],
            "forbidden_files": access["forbidden_files"],
            "forbidden_accesses": [],
            "parsed_file_events": access["parsed_file_events"],
            "trace_files": access["trace_files"],
        }
        _write_json_atomic(run_dir / "audit_manifest.json", report)
        _write_json_atomic(canonical_report, report)
        _append_jsonl(Path(registry_path), report)
    return validate_audit_reports(audit_root, registration)


def _load_and_validate_registration(
    registration_path: Path,
    protocol: Mapping[str, object],
    protocol_path: Path,
    repository_root: Path,
    oracle_build_manifest: Path,
    candidate_executable: Path,
) -> dict[str, object]:
    registration = _read_json(registration_path, "batch registration")
    validate_batch_registration(registration, protocol, sha256_file(protocol_path))
    current = _build_registration(
        protocol,
        protocol_path,
        repository_root,
        oracle_build_manifest,
        candidate_executable,
        registered_utc=str(registration["registered_utc"]),
    )
    if _without_time(current) != _without_time(registration):
        raise ValueError("batch registration differs from current immutable inputs")
    return registration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--oracle-build-manifest", type=Path, required=True)
    parser.add_argument("--candidate-executable", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifests", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--forbidden-root", type=Path, action="append", default=[])
    parser.add_argument("--forbidden-file", type=Path, action="append", default=[])
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--validate-only", action="store_true")
    actions.add_argument("--register-only", action="store_true")
    actions.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    repository = Path.cwd().resolve()
    registration_path = args.run_root / "batch_registration.json"
    try:
        if args.data_root.resolve() != (repository / "data/tum/raw").resolve():
            raise ValueError("data root override differs from frozen repository layout")
        if args.data_manifests.resolve() != (repository / "data/tum/manifests").resolve():
            raise ValueError("data manifest override differs from frozen repository layout")
        protocol = load_protocol(args.protocol, args.experiment_manifest)
        if args.register_only:
            result = register_batch(
                protocol, args.protocol, repository, args.oracle_build_manifest,
                args.candidate_executable, registration_path, args.registry,
            )
        elif not registration_path.is_file():
            if not args.validate_only:
                raise ValueError("register the formal batch before audit or execution")
            result = _build_registration(
                protocol, args.protocol, repository, args.oracle_build_manifest,
                args.candidate_executable, registered_utc="VALIDATION_ONLY",
            )
        else:
            registration = _load_and_validate_registration(
                registration_path, protocol, args.protocol, repository,
                args.oracle_build_manifest, args.candidate_executable,
            )
            audit_root = args.run_root / "audits"
            if args.audit_only:
                result = {
                    "valid": True,
                    "audits": {
                        key: str(value) for key, value in _run_access_audits(
                            registration, audit_root, args.registry,
                            [path.resolve() for path in args.forbidden_root],
                            [path.resolve() for path in args.forbidden_file],
                        ).items()
                    },
                }
            elif args.validate_only:
                audits = (
                    validate_audit_reports(audit_root, registration)
                    if audit_root.is_dir() else {}
                )
                result = {
                    "valid": True,
                    "audits": len(audits),
                    "completed": len(validate_resume(args.run_root, registration, protocol)),
                }
            else:
                validate_audit_reports(audit_root, registration)
                result = execute_matrix(
                    protocol, registration, args.run_root, args.registry
                )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"P05_FORMAL_RUNNER_ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
