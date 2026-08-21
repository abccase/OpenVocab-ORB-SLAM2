#!/usr/bin/env python3
"""Verify the frozen five-run P02-to-P05 baseline equivalence envelope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from semantic_py.openvocab_slam.config import FORMAL_BASELINE_PRODUCER_COMMIT


TrajectoryRow = tuple[float, float, float, float, float, float, float, float]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _validate_registered_pair(
    oracle: dict[str, object], candidate: dict[str, object], *,
    sequence_id: str, seed: int, dataset_manifest_sha256: str,
    source_tree_sha256: str, experiment_manifest_sha256: str,
    expected_oracle_producer_commit: str,
    expected_candidate_producer_commit: str,
) -> None:
    expected = (
        (oracle, "oracle", "baseline", "oracle", expected_oracle_producer_commit),
        (candidate, "equivalence", "baseline", "candidate",
         expected_candidate_producer_commit),
    )
    for manifest, study, mode, label, expected_producer in expected:
        if (not isinstance(expected_producer, str) or len(expected_producer) != 40 or
                any(character not in "0123456789abcdef"
                    for character in expected_producer)):
            raise ValueError(f"expected {label} producer identity is not explicit")
        if (manifest.get("sequence_id") != sequence_id or
                manifest.get("seed") != seed or
                manifest.get("study") != study or manifest.get("mode") != mode):
            raise ValueError("run registration sequence/seed/study/mode mismatch")
        producer = manifest.get("producer_commit")
        if (not isinstance(producer, str) or len(producer) != 40 or
                any(character not in "0123456789abcdef" for character in producer)):
            raise ValueError("run producer identity is not explicit")
        if producer != expected_producer:
            raise ValueError(f"{label} producer identity differs from trusted expectation")
    for field in ("compatibility_commit", "vocabulary", "settings",
                  "association_sha256", "dataset_manifest_sha256",
                  "expected_frames"):
        if oracle.get(field) != candidate.get(field):
            raise ValueError(f"oracle/candidate registration differs for {field}")
    if oracle.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("run dataset identity differs from current frozen manifest")
    if candidate.get("cache_identity") is not None or candidate.get("cache_root") is not None:
        raise ValueError("equivalence candidate registered semantic cache assets")
    if candidate.get("pacing") != "dataset_timestamp_paced_relative":
        raise ValueError("candidate pacing identity differs from oracle protocol")
    verified = candidate.get("verified_inputs")
    if (not isinstance(verified, dict) or
            verified.get("source_tree_sha256") != source_tree_sha256):
        raise ValueError("candidate source-tree identity mismatch")
    registration = candidate.get("registration_identity")
    if (not isinstance(registration, dict) or
            registration.get("experiment_manifest_sha256") !=
            experiment_manifest_sha256):
        raise ValueError("candidate experiment/config identity mismatch")


def parse_tum_trajectory(path: Path) -> list[TrajectoryRow]:
    rows: list[TrajectoryRow] = []
    previous = -math.inf
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"{path}:{line_number}: expected eight TUM columns")
        try:
            values = tuple(float(field) for field in fields)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric field") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: non-finite trajectory field")
        if values[0] <= previous:
            raise ValueError(f"{path}:{line_number}: timestamps are not strictly increasing")
        previous = values[0]
        rows.append(values)  # type: ignore[arg-type]
    if not rows:
        raise ValueError(f"trajectory has no rows: {path}")
    return rows


def associate_timestamps(
    first: Sequence[float], second: Sequence[float], max_difference: float,
) -> list[tuple[int, int]]:
    if not math.isfinite(max_difference) or max_difference < 0:
        raise ValueError("maximum timestamp difference must be finite and nonnegative")
    candidates = [
        (abs(a - b), first_index, second_index)
        for first_index, a in enumerate(first)
        for second_index, b in enumerate(second)
        if abs(a - b) <= max_difference
    ]
    candidates.sort()
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, first_index, second_index in candidates:
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        matches.append((first_index, second_index))
    return sorted(matches)


def compute_ate_rmse(
    candidate: Sequence[TrajectoryRow],
    groundtruth: Sequence[TrajectoryRow],
    *,
    max_difference: float = 0.02,
) -> tuple[float, int]:
    pairs = associate_timestamps(
        [row[0] for row in candidate], [row[0] for row in groundtruth],
        max_difference,
    )
    if len(pairs) < 3:
        raise ValueError("ATE requires at least three associated poses")
    source = np.asarray([[candidate[i][1], candidate[i][2], candidate[i][3]]
                         for i, _ in pairs], dtype=np.float64)
    target = np.asarray([[groundtruth[j][1], groundtruth[j][2], groundtruth[j][3]]
                         for _, j in pairs], dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    aligned = (rotation @ source.T).T + translation
    errors = np.linalg.norm(aligned - target, axis=1)
    return float(np.sqrt(np.mean(errors * errors))), len(pairs)


def _validated(rows: Sequence[dict[str, object]], label: str) -> list[dict[str, float | int]]:
    if len(rows) != 5:
        raise ValueError(f"{label} must contain exactly five runs")
    values: list[dict[str, float | int]] = []
    seeds: set[int] = set()
    for row in rows:
        seed = int(row["seed"])
        valid_fraction = float(row["valid_pose_fraction"])
        ate = float(row["ate_rmse_m"])
        if seed in seeds:
            raise ValueError(f"{label} contains duplicate seed {seed}")
        if not (math.isfinite(valid_fraction) and 0.0 <= valid_fraction <= 1.0):
            raise ValueError(f"{label} has invalid valid-pose fraction")
        if not (math.isfinite(ate) and ate >= 0.0):
            raise ValueError(f"{label} has invalid ATE")
        seeds.add(seed)
        values.append({"seed": seed, "valid_pose_fraction": valid_fraction, "ate_rmse_m": ate})
    return values


def verify_equivalence(
    oracle_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    oracle = _validated(oracle_rows, "oracle")
    candidate = _validated(candidate_rows, "candidate")
    oracle_by_seed = {int(row["seed"]): row for row in oracle}
    candidate_by_seed = {int(row["seed"]): row for row in candidate}
    if set(oracle_by_seed) != set(candidate_by_seed):
        raise ValueError("oracle and candidate seeds are not paired")

    oracle_valid = [float(row["valid_pose_fraction"]) for row in oracle]
    candidate_valid = [float(row["valid_pose_fraction"]) for row in candidate]
    valid_difference = abs(statistics.median(candidate_valid) - statistics.median(oracle_valid))

    oracle_ate = [float(row["ate_rmse_m"]) for row in oracle]
    candidate_ate = [float(row["ate_rmse_m"]) for row in candidate]
    oracle_median = statistics.median(oracle_ate)
    candidate_median = statistics.median(candidate_ate)
    pooled_mad = statistics.median(
        [abs(value - oracle_median) for value in oracle_ate] +
        [abs(value - candidate_median) for value in candidate_ate]
    )
    ate_difference = abs(candidate_median - oracle_median)
    ate_tolerance = max(1e-4, 2.0 * pooled_mad)
    valid = valid_difference <= 0.005 and ate_difference <= ate_tolerance
    return {
        "schema_version": 1,
        "valid": valid,
        "paired_seeds": sorted(oracle_by_seed),
        "valid_pose_fraction_difference": valid_difference,
        "valid_pose_fraction_tolerance": 0.005,
        "median_ate_difference_m": ate_difference,
        "pooled_ate_mad_m": pooled_mad,
        "ate_tolerance_m": ate_tolerance,
    }


def _telemetry_pose_fraction(path: Path, *, require_no_semantic_access: bool) -> tuple[float, int]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        pose_valid = [bool(row["pose_valid"]) for row in rows]
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        pose_valid = [int(row["pose_valid"]) == 1 for row in rows]
        if require_no_semantic_access:
            for row in rows:
                if (int(row["semantic_accessed"]) != 0 or
                        row["semantic_state"] != "BASELINE" or
                        int(row["raw_keypoints"]) != int(row["used_keypoints"]) or
                        any(int(row[name]) != 0 for name in (
                            "removed_dynamic", "retained_uncertain", "removed_uncertain"))):
                    raise ValueError(f"baseline telemetry accessed semantics: {path}")
    else:
        raise ValueError(f"unsupported telemetry format: {path}")
    if not pose_valid:
        raise ValueError(f"telemetry has no frames: {path}")
    return sum(pose_valid) / len(pose_valid), len(pose_valid)


def _completed_attempt(condition_root: Path) -> Path:
    valid: list[Path] = []
    for attempt in sorted(condition_root.glob("attempt-*")):
        manifest_path = attempt / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state") == "COMPLETED" and manifest.get("valid") is True:
            valid.append(attempt)
    if len(valid) != 1:
        raise ValueError(f"expected exactly one valid attempt under {condition_root}, got {len(valid)}")
    return valid[0]


def measure_run(
    run_dir: Path, groundtruth_path: Path, *, require_no_semantic_access: bool,
) -> dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETED" or manifest.get("valid") is not True:
        raise ValueError(f"run is not valid and complete: {run_dir}")
    if require_no_semantic_access and manifest.get("mode") != "baseline":
        raise ValueError(f"equivalence candidate is not baseline mode: {run_dir}")
    trajectory_path = run_dir / "CameraTrajectory.txt"
    telemetry_path = run_dir / (
        "frame_telemetry.csv" if (run_dir / "frame_telemetry.csv").is_file()
        else "frame_telemetry.jsonl"
    )
    trajectory_sha256 = _sha256_file(trajectory_path)
    telemetry_sha256 = _sha256_file(telemetry_path)
    if (trajectory_sha256 != manifest.get("trajectory", {}).get("sha256") or
            telemetry_sha256 != manifest.get("telemetry", {}).get("sha256")):
        raise ValueError(f"run artifact hash differs from manifest: {run_dir}")
    trajectory = parse_tum_trajectory(trajectory_path)
    groundtruth = parse_tum_trajectory(groundtruth_path)
    ate_rmse, associated = compute_ate_rmse(trajectory, groundtruth)
    valid_fraction, telemetry_frames = _telemetry_pose_fraction(
        telemetry_path, require_no_semantic_access=require_no_semantic_access)
    if telemetry_frames != int(manifest["expected_frames"]):
        raise ValueError(f"telemetry coverage differs from run manifest: {run_dir}")
    return {
        "seed": int(manifest["seed"]),
        "producer_commit": manifest.get("producer_commit"),
        "compatibility_commit": manifest.get("compatibility_commit"),
        "executable_sha256": manifest.get("executable", {}).get("sha256"),
        "vocabulary_sha256": manifest.get("vocabulary", {}).get("sha256"),
        "settings_sha256": manifest.get("settings", {}).get("sha256"),
        "association_sha256": manifest.get("association_sha256"),
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "valid_pose_fraction": valid_fraction,
        "ate_rmse_m": ate_rmse,
        "associated_pose_count": associated,
        "trajectory_pose_count": len(trajectory),
        "trajectory_sha256": trajectory_sha256,
        "groundtruth_sha256": _sha256_file(groundtruth_path),
        "telemetry_sha256": telemetry_sha256,
        "run_manifest_sha256": _sha256_file(manifest_path),
        "run_dir": str(run_dir.resolve()),
    }


def build_equivalence_report(
    *, oracle_root: Path, candidate_root: Path, data_root: Path,
    experiment_manifest: Path,
    expected_oracle_producer_commit: str = FORMAL_BASELINE_PRODUCER_COMMIT,
) -> dict[str, object]:
    manifest = json.loads(experiment_manifest.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in manifest["random_seeds"]]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("experiment manifest must freeze five unique seeds")
    sequences: dict[str, object] = {}
    all_valid = True
    experiment_sha = _sha256_file(experiment_manifest)
    candidate_producer_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    for dataset in manifest["datasets"]:
        sequence_id = str(dataset["id"])
        sequence_root = data_root / str(dataset["archive"])[:-4]
        groundtruth_path = sequence_root / "groundtruth.txt"
        dataset_manifest_path = data_root.parent / "manifests" / f"{sequence_id}.json"
        dataset_identity = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        dataset_sha = _sha256_file(dataset_manifest_path)
        source_tree = str(dataset_identity["extracted_tree_sha256"])
        oracle_rows = []
        candidate_rows = []
        for seed in seeds:
            oracle_dir = _completed_attempt(oracle_root / sequence_id / f"seed-{seed}")
            candidate_dir = _completed_attempt(candidate_root / sequence_id / f"seed-{seed}")
            oracle_manifest = json.loads(
                (oracle_dir / "run_manifest.json").read_text(encoding="utf-8"))
            candidate_manifest = json.loads(
                (candidate_dir / "run_manifest.json").read_text(encoding="utf-8"))
            _validate_registered_pair(
                oracle_manifest, candidate_manifest, sequence_id=sequence_id,
                seed=seed, dataset_manifest_sha256=dataset_sha,
                source_tree_sha256=source_tree,
                experiment_manifest_sha256=experiment_sha,
                expected_oracle_producer_commit=expected_oracle_producer_commit,
                expected_candidate_producer_commit=candidate_producer_commit)
            oracle_rows.append(measure_run(
                oracle_dir, groundtruth_path, require_no_semantic_access=False))
            candidate_rows.append(measure_run(
                candidate_dir, groundtruth_path, require_no_semantic_access=True))
        result = verify_equivalence(oracle_rows, candidate_rows)
        all_valid = all_valid and bool(result["valid"])
        sequences[sequence_id] = {
            "oracle_runs": oracle_rows,
            "candidate_runs": candidate_rows,
            "equivalence": result,
        }
    return {
        "schema_version": 2,
        "study_id": manifest["study_id"],
        "valid": all_valid,
        "trajectory_alignment": "SE3",
        "association_max_difference_seconds": 0.02,
        "tool_sha256": _sha256_file(Path(__file__).resolve()),
        "code_identity": candidate_producer_commit,
        "producer_identity": {
            "oracle_expected_commit": expected_oracle_producer_commit,
            "candidate_expected_commit": candidate_producer_commit,
        },
        "numpy_version": np.__version__,
        "parameters": {
            "oracle_root": str(oracle_root.resolve()),
            "candidate_root": str(candidate_root.resolve()),
            "data_root": str(data_root.resolve()),
            "experiment_manifest": str(experiment_manifest.resolve()),
            "experiment_manifest_sha256": experiment_sha,
            "required_runs_per_sequence": 5,
            "study_id": manifest["study_id"],
            "sequence_ids": [str(item["id"]) for item in manifest["datasets"]],
            "seeds": seeds,
            "playback": manifest.get("playback"),
            "association_max_difference_seconds": 0.02,
            "trajectory_alignment": "SE3_no_scale",
            "valid_pose_fraction_tolerance": 0.005,
            "ate_tolerance_floor_m": 1e-4,
            "ate_tolerance_pooled_mad_multiplier": 2.0,
        },
        "sequences": sequences,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, default=Path("runs/oracle"))
    parser.add_argument("--candidate-root", type=Path,
                        default=Path("runs/equivalence/baseline"))
    parser.add_argument("--data-root", type=Path, default=Path("data/tum/raw"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("config/EXPERIMENT_MANIFEST.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_equivalence_report(
        oracle_root=args.oracle_root, candidate_root=args.candidate_root,
        data_root=args.data_root, experiment_manifest=args.manifest,
    )
    _write_json_atomic(args.output, result)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
