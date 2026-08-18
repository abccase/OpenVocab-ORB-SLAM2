#!/usr/bin/env python3
"""Build, generate, and validate formal causal dynamic-score caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_py.openvocab_slam.dynamic_cache import (
    DynamicCacheJob,
    DynamicCacheManifest,
    build_dynamic_job,
    generate_dynamic_cache,
    hash_dataset_tree,
    validate_dynamic_cache,
)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _require_clean_product_tree(root: Path) -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root, text=True
    )
    if status:
        raise ValueError("product tree must be clean before dynamic cache generation")


def _load_intrinsics(path: Path) -> np.ndarray:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"unable to open camera settings: {path}")
    try:
        matrix = np.array(
            [
                [storage.getNode("Camera.fx").real(), 0.0, storage.getNode("Camera.cx").real()],
                [0.0, storage.getNode("Camera.fy").real(), storage.getNode("Camera.cy").real()],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    finally:
        storage.release()
    if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError(f"invalid camera intrinsics: {path}")
    return matrix


def _build_formal_jobs(
    root: Path,
    selected: set[str] | None = None,
    *,
    enforce_producer_commit: bool = True,
) -> list[DynamicCacheJob]:
    experiment = _load_json(root / "config/EXPERIMENT_MANIFEST.yaml")
    producer = _git_head(root)
    jobs: list[DynamicCacheJob] = []
    for dataset in experiment["datasets"]:
        sequence_id = str(dataset["id"])
        if selected is not None and sequence_id not in selected:
            continue
        dataset_root = root / "data/tum/raw" / Path(str(dataset["archive"])).stem
        sequence_manifest = _load_json(root / "data/tum/manifests" / f"{sequence_id}.json")
        if hash_dataset_tree(dataset_root) != sequence_manifest["extracted_tree_sha256"]:
            raise ValueError(f"source tree hash mismatch for {sequence_id}")
        selected_producer = producer
        existing_manifest = root / "cache/dynamic/v1" / sequence_id / "cache_manifest.json"
        if existing_manifest.is_file():
            observed = DynamicCacheManifest.from_primitive(_load_json(existing_manifest))
            if enforce_producer_commit and observed.producer_commit != producer:
                raise ValueError(f"producer commit mismatch for {sequence_id}")
            selected_producer = observed.producer_commit
        run_root = root / "runs/oracle" / sequence_id / "seed-23011/attempt-001"
        jobs.append(
            build_dynamic_job(
                root,
                sequence_id,
                dataset_root,
                root / "cache/semantic/v1" / sequence_id,
                run_root / "CameraTrajectory.txt",
                run_root / "run_manifest.json",
                _load_intrinsics(root / "Examples/RGB-D" / str(dataset["settings"])),
                producer_commit=selected_producer,
                dataset_manifest_path=(
                    root / "data/tum/manifests" / f"{sequence_id}.json"
                ),
            )
        )
    if selected and {job.sequence_id for job in jobs} != selected:
        raise ValueError("requested sequence is not in the experiment manifest")
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-sequences", action="store_true")
    group.add_argument("--sequence", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    selected = None if args.all_sequences else set(args.sequence or [])
    jobs = _build_formal_jobs(
        PROJECT_ROOT,
        selected,
        enforce_producer_commit=not args.validate_only,
    )
    if args.validate_only:
        failed = False
        for job in jobs:
            result = validate_dynamic_cache(job)
            state = "PASS" if result.valid else "FAIL"
            print(f"DYNAMIC_CACHE_VALIDATE {job.sequence_id} {state} frames={result.frame_count}")
            for error in result.errors:
                print(f"  {error}")
            failed = failed or not result.valid
        return int(failed)
    _require_clean_product_tree(PROJECT_ROOT)
    if not args.resume and any(job.cache_root.exists() for job in jobs):
        raise ValueError("dynamic cache already exists; pass --resume to validate and continue")
    for job in jobs:
        result = generate_dynamic_cache(job)
        print(f"DYNAMIC_CACHE_COMPLETE {job.sequence_id} frames={len(result.frame_index)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DYNAMIC_CACHE_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
