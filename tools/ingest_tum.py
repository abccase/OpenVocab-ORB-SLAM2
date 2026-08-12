#!/usr/bin/env python3
"""Validate and safely ingest user-provided TUM RGB-D archives."""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import math
import shutil
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import cv2
import numpy as np


EXPECTED_ARCHIVES = (
    "rgbd_dataset_freiburg1_desk.tgz",
    "rgbd_dataset_freiburg1_room.tgz",
    "rgbd_dataset_freiburg3_sitting_xyz.tgz",
    "rgbd_dataset_freiburg3_sitting_halfsphere.tgz",
    "rgbd_dataset_freiburg3_walking_xyz.tgz",
    "rgbd_dataset_freiburg3_walking_halfsphere.tgz",
)


class UnsafeArchive(ValueError):
    """Raised before extraction when an archive violates the path contract."""


@dataclass(frozen=True)
class InboxValidation:
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected


@dataclass(frozen=True)
class DatasetSpec:
    sequence_id: str
    archive: str
    settings: str


def validate_inbox(inbox: Path) -> InboxValidation:
    """Require exactly the frozen six regular archive files."""
    inbox = Path(inbox)
    actual = {entry.name for entry in inbox.iterdir()} if inbox.is_dir() else set()
    expected = set(EXPECTED_ARCHIVES)
    missing = tuple(name for name in EXPECTED_ARCHIVES if name not in actual)
    unexpected = tuple(sorted(actual - expected))
    for name in EXPECTED_ARCHIVES:
        path = inbox / name
        if name in actual and (not path.is_file() or path.is_symlink()):
            missing += (name,)
    return InboxValidation(missing=missing, unexpected=unexpected)


def _validated_member_path(name: str, allowed_top_level: str) -> tuple[str, ...]:
    if not name or name.startswith("/") or "\x00" in name:
        raise UnsafeArchive(f"unsafe absolute or empty member path: {name!r}")
    raw = PurePosixPath(name)
    if any(part == ".." for part in raw.parts):
        raise UnsafeArchive(f"parent traversal in member path: {name!r}")
    normalized = tuple(part for part in raw.parts if part not in ("", "."))
    if not normalized or normalized[0] != allowed_top_level:
        raise UnsafeArchive(f"unexpected top-level path: {name!r}")
    return normalized


def inspect_archive(archive_path: Path, allowed_top_level: str) -> tuple[tarfile.TarInfo, ...]:
    """Read every tar header and reject unsafe member types and destinations."""
    seen: set[tuple[str, ...]] = set()
    members: list[tarfile.TarInfo] = []
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                normalized = _validated_member_path(member.name, allowed_top_level)
                if normalized in seen:
                    raise UnsafeArchive(f"duplicate normalized member path: {member.name!r}")
                seen.add(normalized)
                if member.issym() or member.islnk():
                    raise UnsafeArchive(f"links are forbidden: {member.name!r}")
                if member.isdev() or member.isfifo():
                    raise UnsafeArchive(f"device and FIFO members are forbidden: {member.name!r}")
                if not (member.isdir() or member.isreg()):
                    raise UnsafeArchive(f"unsupported member type: {member.name!r}")
                members.append(member)
    except (tarfile.TarError, OSError) as exc:
        raise UnsafeArchive(f"cannot read gzip tar archive {archive_path}: {exc}") from exc
    if not members:
        raise UnsafeArchive(f"archive is empty: {archive_path}")
    return tuple(members)


def safe_extract_archive(archive_path: Path, output_root: Path, top_level: str) -> Path:
    """Extract a validated archive into staging and atomically promote its tree."""
    members = inspect_archive(archive_path, top_level)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / top_level
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {target}")
    staging = output_root / f".{top_level}.partial-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in members:
                normalized = _validated_member_path(member.name, top_level)
                relative = normalized[1:]
                if not relative:
                    continue
                destination = staging.joinpath(*relative)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise UnsafeArchive(f"regular member has no payload: {member.name!r}")
                with source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
        staging.rename(target)
        return target
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _require_unique_timestamps(stream: Sequence[tuple[float, str]], name: str) -> None:
    timestamps = [timestamp for timestamp, _ in stream]
    if len(set(timestamps)) != len(timestamps):
        raise ValueError(f"duplicate {name} timestamp")


def associate_streams(
    rgb: Iterable[tuple[float, str]],
    depth: Iterable[tuple[float, str]],
    max_difference: float = 0.02,
) -> list[tuple[float, str, float, str]]:
    """Greedily pair nearest RGB/depth times once, with deterministic ties."""
    if max_difference < 0:
        raise ValueError("max_difference must be nonnegative")
    rgb_rows = sorted(rgb, key=lambda row: (row[0], row[1]))
    depth_rows = sorted(depth, key=lambda row: (row[0], row[1]))
    _require_unique_timestamps(rgb_rows, "RGB")
    _require_unique_timestamps(depth_rows, "depth")
    result: list[tuple[float, str, float, str]] = []
    epsilon = 1e-12
    candidates = sorted(
        (
            abs(rgb_timestamp - depth_timestamp),
            rgb_timestamp,
            depth_timestamp,
            rgb_path,
            depth_path,
            rgb_index,
            depth_index,
        )
        for rgb_index, (rgb_timestamp, rgb_path) in enumerate(rgb_rows)
        for depth_index, (depth_timestamp, depth_path) in enumerate(depth_rows)
        if abs(rgb_timestamp - depth_timestamp) <= max_difference + epsilon
    )
    used_rgb: set[int] = set()
    used_depth: set[int] = set()
    for _, rgb_timestamp, depth_timestamp, rgb_path, depth_path, rgb_index, depth_index in candidates:
        if rgb_index in used_rgb or depth_index in used_depth:
            continue
        used_rgb.add(rgb_index)
        used_depth.add(depth_index)
        result.append((rgb_timestamp, rgb_path, depth_timestamp, depth_path))
    return sorted(result, key=lambda row: (row[0], row[1], row[2], row[3]))


def _safe_relative_path(value: str, source: Path) -> str:
    path = PurePosixPath(value)
    if not value or value.startswith("/") or any(part in ("", "..") for part in path.parts):
        raise ValueError(f"unsafe relative path in {source}: {value!r}")
    return path.as_posix()


def parse_tum_index(path: Path, expected_columns: int) -> list[tuple[float, tuple[str, ...]]]:
    """Parse a TUM timestamp file with finite, unique timestamps."""
    if expected_columns < 2:
        raise ValueError("expected_columns must be at least two")
    rows: list[tuple[float, tuple[str, ...]]] = []
    seen: set[float] = set()
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != expected_columns:
            raise ValueError(f"{path}:{line_number}: expected {expected_columns} columns")
        try:
            timestamp = float(fields[0])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid timestamp") from exc
        if not math.isfinite(timestamp) or timestamp in seen:
            raise ValueError(f"{path}:{line_number}: duplicate or non-finite timestamp")
        seen.add(timestamp)
        values = tuple(fields[1:])
        if expected_columns == 2:
            values = (_safe_relative_path(values[0], Path(path)),)
        else:
            try:
                numeric = tuple(float(value) for value in values)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric trajectory value") from exc
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"{path}:{line_number}: non-finite trajectory value")
        rows.append((timestamp, values))
    if not rows:
        raise ValueError(f"no data rows in {path}")
    return sorted(rows, key=lambda row: row[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_extracted_tree(root: Path) -> str:
    """Hash original extracted regular files, excluding generated association data."""
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.is_symlink() or path.name == "associate.txt":
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validate_images(
    dataset_root: Path,
    rgb_rows: Sequence[tuple[float, tuple[str, ...]]],
    depth_rows: Sequence[tuple[float, tuple[str, ...]]],
) -> tuple[dict[str, int], dict[str, int | str]]:
    dimensions: set[tuple[int, int]] = set()
    for _, (relative,) in rgb_rows:
        path = dataset_root / relative
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError(f"invalid RGB image: {relative}")
        dimensions.add((int(image.shape[1]), int(image.shape[0])))
    depth_dimensions: set[tuple[int, int]] = set()
    minimum_positive: int | None = None
    maximum = 0
    for _, (relative,) in depth_rows:
        path = dataset_root / relative
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint16 or image.ndim != 2:
            raise ValueError(f"invalid uint16 depth image: {relative}")
        depth_dimensions.add((int(image.shape[1]), int(image.shape[0])))
        positive = image[image > 0]
        if positive.size:
            current_minimum = int(positive.min())
            minimum_positive = current_minimum if minimum_positive is None else min(minimum_positive, current_minimum)
            maximum = max(maximum, int(positive.max()))
    if len(dimensions) != 1 or dimensions != depth_dimensions:
        raise ValueError(f"inconsistent RGB/depth dimensions: rgb={dimensions}, depth={depth_dimensions}")
    if minimum_positive is None:
        raise ValueError("all depth images contain only invalid zero values")
    width, height = next(iter(dimensions))
    return (
        {"width": width, "height": height},
        {"dtype": "uint16", "min_positive": minimum_positive, "max": maximum},
    )


def build_sequence_manifest(
    *,
    sequence_id: str,
    dataset_root: Path,
    archive_path: Path,
    settings: str,
    manifest_path: Path,
) -> dict[str, object]:
    """Validate one extracted TUM tree and atomically write its manifest."""
    dataset_root = Path(dataset_root)
    required = {name: dataset_root / name for name in ("rgb.txt", "depth.txt", "groundtruth.txt")}
    for name, path in required.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing required regular file: {name}")
    rgb_index = parse_tum_index(required["rgb.txt"], 2)
    depth_index = parse_tum_index(required["depth.txt"], 2)
    groundtruth = parse_tum_index(required["groundtruth.txt"], 8)
    for _, (relative,) in (*rgb_index, *depth_index):
        path = dataset_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"indexed image is not a regular file: {relative}")
    rgb_rows = [(timestamp, values[0]) for timestamp, values in rgb_index]
    depth_rows = [(timestamp, values[0]) for timestamp, values in depth_index]
    associations = associate_streams(rgb_rows, depth_rows, 0.02)
    if not associations:
        raise ValueError("no RGB-depth associations within 0.02 seconds")
    association_text = "".join(
        f"{rgb_time:.9f} {rgb_path} {depth_time:.9f} {depth_path}\n"
        for rgb_time, rgb_path, depth_time, depth_path in associations
    )
    association_path = dataset_root / "associate.txt"
    original_tree_sha256 = hash_extracted_tree(dataset_root)
    image_dimensions, depth_facts = _validate_images(dataset_root, rgb_index, depth_index)
    _write_text_atomic(association_path, association_text)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "sequence_id": sequence_id,
        "archive": Path(archive_path).name,
        "archive_sha256": _sha256_file(Path(archive_path)),
        "extracted_tree_sha256": original_tree_sha256,
        "required_file_sha256": {name: _sha256_file(path) for name, path in required.items()},
        "association_sha256": _sha256_file(association_path),
        "counts": {
            "rgb": len(rgb_index),
            "depth": len(depth_index),
            "groundtruth": len(groundtruth),
            "associations": len(associations),
        },
        "timestamp_range": {
            "rgb": [rgb_index[0][0], rgb_index[-1][0]],
            "depth": [depth_index[0][0], depth_index[-1][0]],
            "groundtruth": [groundtruth[0][0], groundtruth[-1][0]],
        },
        "image_dimensions": image_dimensions,
        "depth": depth_facts,
        "settings": settings,
        "max_association_difference_seconds": 0.02,
        "validation_status": "VALID",
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _write_text_atomic(Path(manifest_path), encoded)
    return manifest


def _top_level_from_archive(name: str) -> str:
    if not name.endswith(".tgz"):
        raise ValueError(f"archive must keep .tgz basename: {name}")
    top_level = name[:-4]
    if not top_level or "/" in top_level or "\\" in top_level:
        raise ValueError(f"unsafe archive basename: {name}")
    return top_level


def ingest_datasets(
    datasets: Sequence[DatasetSpec],
    inbox: Path,
    output_root: Path,
    manifests_root: Path,
    *,
    required_free_bytes: int = 20 * 1024**3,
    validate_only: bool = False,
) -> list[dict[str, object]]:
    """Inspect every archive first, then extract/resume and validate each dataset."""
    inbox = Path(inbox)
    output_root = Path(output_root)
    manifests_root = Path(manifests_root)
    expected_names = {spec.archive for spec in datasets}
    actual_names = {entry.name for entry in inbox.iterdir()} if inbox.is_dir() else set()
    if actual_names != expected_names:
        raise ValueError(
            f"inbox allowlist mismatch: missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    free_bytes = shutil.disk_usage(inbox).free
    if free_bytes < required_free_bytes:
        raise ValueError(f"insufficient free disk: {free_bytes} < {required_free_bytes}")
    inspected: list[tuple[DatasetSpec, Path, str]] = []
    for spec in datasets:
        archive_path = inbox / spec.archive
        if not archive_path.is_file() or archive_path.is_symlink():
            raise ValueError(f"archive is not a regular file: {archive_path}")
        top_level = _top_level_from_archive(spec.archive)
        inspect_archive(archive_path, top_level)
        inspected.append((spec, archive_path, top_level))
    results: list[dict[str, object]] = []
    for spec, archive_path, top_level in inspected:
        dataset_root = output_root / top_level
        if validate_only and not dataset_root.is_dir():
            raise ValueError(f"dataset not extracted for validation: {dataset_root}")
        if not dataset_root.exists():
            safe_extract_archive(archive_path, output_root, top_level)
        results.append(
            build_sequence_manifest(
                sequence_id=spec.sequence_id,
                dataset_root=dataset_root,
                archive_path=archive_path,
                settings=spec.settings,
                manifest_path=manifests_root / f"{spec.sequence_id}.json",
            )
        )
    return results


def _load_specs(manifest_path: Path) -> list[DatasetSpec]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    specs = [
        DatasetSpec(item["id"], item["archive"], item["settings"])
        for item in manifest.get("datasets", [])
    ]
    if tuple(spec.archive for spec in specs) != EXPECTED_ARCHIVES:
        raise ValueError("experiment manifest does not contain the frozen six archive allowlist")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("config/EXPERIMENT_MANIFEST.yaml"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifests = ingest_datasets(
        _load_specs(args.manifest),
        args.inbox,
        args.output,
        args.manifests,
        validate_only=args.validate_only,
    )
    for manifest in manifests:
        print(
            f"VALID {manifest['sequence_id']} associations={manifest['counts']['associations']} "
            f"tree_sha256={manifest['extracted_tree_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
