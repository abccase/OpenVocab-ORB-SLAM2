#!/usr/bin/env python3
"""Run and validate reproducible ORB-SLAM2 TUM baseline conditions."""

from __future__ import annotations

import argparse
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


def _artifact(path: Path, *, pose_count: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": path.name, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    if pose_count is not None:
        value["pose_count"] = pose_count
    return value


def _completed_attempt(attempt: Path, compatibility_commit: str, expected_frames: int) -> RunResult | None:
    manifest_path = attempt / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state") != "COMPLETED" or manifest.get("valid") is not True:
            return None
        if manifest.get("compatibility_commit") != compatibility_commit:
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
    registry: Path | None = None,
    study: str = "oracle",
) -> RunResult:
    sequence_root = Path(condition.sequence_root).resolve()
    association = sequence_root / "associate.txt"
    expected_frames = _association_count(association)
    condition_root = Path(output_root) / condition.sequence_id / f"seed-{condition.seed}"
    attempts = sorted(condition_root.glob("attempt-*")) if condition_root.is_dir() else []
    for attempt in attempts:
        completed = _completed_attempt(attempt, compatibility_commit, expected_frames)
        if completed is not None:
            return completed
    attempt_number = len(attempts) + 1
    run_dir = condition_root / f"attempt-{attempt_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    executable = Path(executable).resolve()
    vocabulary = Path(vocabulary).resolve()
    settings = Path(condition.settings).resolve()
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


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline",), required=True)
    parser.add_argument("--study", choices=("smoke", "oracle"), required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--all-sequences", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("config/EXPERIMENT_MANIFEST.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/tum/raw"))
    parser.add_argument("--data-manifests", type=Path, default=Path("data/tum/manifests"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("Examples/RGB-D/rgbd_tum"))
    parser.add_argument("--vocabulary", type=Path, default=Path("Vocabulary/ORBvoc.txt"))
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
    output_root = args.output_root or Path("runs") / args.study
    failures = 0
    for item in selected:
        sequence_root = args.data_root / item["archive"][:-4]
        condition = RunCondition(
            item["id"],
            int(seeds[0]),
            sequence_root,
            Path("Examples/RGB-D") / item["settings"],
            args.data_manifests / f"{item['id']}.json",
        )
        for seed in seeds:
            condition = RunCondition(condition.sequence_id, int(seed), condition.sequence_root, condition.settings, condition.dataset_manifest)
            result = run_baseline_condition(
                condition,
                executable=args.executable,
                vocabulary=args.vocabulary,
                output_root=output_root,
                compatibility_commit=compatibility_commit,
                registry=args.registry,
                study=args.study,
            )
            print(f"{'VALID' if result.valid else 'INVALID'} {condition.sequence_id} seed={seed} dir={result.run_dir} reason={result.invalid_reason}")
            failures += int(not result.valid)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
