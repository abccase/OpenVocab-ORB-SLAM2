from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import msgpack
import zstandard

from .schemas import CacheManifest, SemanticFramePacket


@dataclass(frozen=True)
class CacheValidation:
    valid: bool
    errors: tuple[str, ...]
    frame_count: int


class CacheWriter:
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.manifest = manifest
        self.frames = self.root / "frames"
        self.index_path = self.root / "cache_index.jsonl"
        self.manifest_path = self.root / "cache_manifest.json"
        self.frames.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            observed = CacheManifest.from_primitive(json.loads(self.manifest_path.read_text()))
            if observed != manifest:
                raise ValueError("manifest identity mismatch")
        else:
            _write_json_atomic(self.manifest_path, manifest.to_primitive())
        self.entries = _read_index(self.index_path)
        _require_unique_index(self.entries)

    def has_valid_frame(self, frame_id: int, timestamp: float, source_image_sha256: str) -> bool:
        matches = [entry for entry in self.entries if int(entry["frame_id"]) == frame_id]
        if not matches:
            return False
        entry = matches[0]
        if float(entry["timestamp"]) != timestamp or entry["source_image_sha256"] != source_image_sha256:
            raise ValueError("existing frame identity mismatch")
        path = self.root / str(entry["path"])
        if _sha256_file(path) != entry["sha256"]:
            raise ValueError("existing frame hash mismatch")
        packet = read_cache_frame(path)
        _require_packet_identity(packet, self.manifest)
        if (
            packet.frame_id != frame_id
            or packet.timestamp != timestamp
            or packet.source_image_sha256 != source_image_sha256
        ):
            raise ValueError("existing packet identity mismatch")
        return True

    def add(self, packet: SemanticFramePacket) -> str:
        _require_packet_identity(packet, self.manifest)
        by_frame = {int(entry["frame_id"]): entry for entry in self.entries}
        if packet.frame_id in by_frame:
            entry = by_frame[packet.frame_id]
            path = self.root / str(entry["path"])
            observed = read_cache_frame(path)
            digest = _sha256_file(path)
            if observed != packet or digest != entry["sha256"]:
                raise ValueError("existing packet identity or hash mismatch")
            return digest
        if any(float(entry["timestamp"]) == packet.timestamp for entry in self.entries):
            raise ValueError("duplicate timestamp")
        path = self.frames / f"{packet.frame_id:06d}.msgpack.zst"
        if path.exists():
            observed = read_cache_frame(path)
            if observed != packet:
                raise ValueError("orphan packet identity mismatch")
            digest = _sha256_file(path)
        else:
            digest = write_cache_frame(path, packet)
        entry = {
            "frame_id": packet.frame_id,
            "timestamp": packet.timestamp,
            "path": str(path.relative_to(self.root)),
            "sha256": digest,
            "source_image_sha256": packet.source_image_sha256,
        }
        _append_jsonl(self.index_path, entry)
        self.entries.append(entry)
        return digest

    def finalize(self) -> "CacheValidation":
        validation = validate_cache(self.root, self.manifest)
        if not validation.valid:
            raise ValueError("cache is incomplete or invalid: " + "; ".join(validation.errors))
        _write_json_atomic(
            self.root / "cache_complete.json",
            {
                "manifest_sha256": _sha256_file(self.manifest_path),
                "index_sha256": _sha256_file(self.index_path),
                "frame_count": validation.frame_count,
            },
        )
        return validation


def write_cache_frame(path, packet):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = msgpack.packb(packet.to_primitive(), use_bin_type=True)
    compressed = zstandard.ZstdCompressor(level=9, threads=0).compress(packed)
    temporary = path.parent / f".{path.name}.partial"
    try:
        with temporary.open("wb") as stream:
            stream.write(compressed)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_file(path)


def read_cache_frame(path):
    compressed = Path(path).read_bytes()
    try:
        packed = zstandard.ZstdDecompressor().decompress(compressed)
        primitive = msgpack.unpackb(packed, raw=False, strict_map_key=True)
    except (zstandard.ZstdError, msgpack.exceptions.UnpackException, ValueError, TypeError) as exc:
        raise ValueError(f"invalid cache frame encoding: {exc}") from exc
    if not isinstance(primitive, dict):
        raise ValueError("cache packet root must be a map")
    return SemanticFramePacket.from_primitive(primitive)


def validate_cache(root, expected):
    root = Path(root)
    errors: list[str] = []
    manifest_path = root / "cache_manifest.json"
    try:
        observed_manifest = CacheManifest.from_primitive(json.loads(manifest_path.read_text()))
        if observed_manifest != expected:
            errors.append("manifest identity mismatch")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid cache manifest: {exc}")
    index_path = root / "cache_index.jsonl"
    try:
        entries = _read_index(index_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return CacheValidation(False, tuple(errors + [f"invalid cache index: {exc}"]), 0)
    seen_frames: set[int] = set()
    seen_timestamps: set[float] = set()
    referenced: set[Path] = set()
    for entry in entries:
        try:
            frame_id = int(entry["frame_id"])
            timestamp = float(entry["timestamp"])
            if frame_id in seen_frames:
                errors.append(f"duplicate frame_id: {frame_id}")
            if timestamp in seen_timestamps:
                errors.append(f"duplicate timestamp: {timestamp}")
            seen_frames.add(frame_id)
            seen_timestamps.add(timestamp)
            relative = Path(str(entry["path"]))
            if relative.is_absolute() or ".." in relative.parts or relative.parent != Path("frames"):
                raise ValueError("unsafe packet path")
            packet_path = root / relative
            referenced.add(packet_path)
            if _sha256_file(packet_path) != entry["sha256"]:
                raise ValueError("packet hash mismatch")
            packet = read_cache_frame(packet_path)
            _require_packet_identity(packet, expected)
            if packet.frame_id != frame_id or packet.timestamp != timestamp:
                raise ValueError("packet index identity mismatch")
            if packet.source_image_sha256 != entry["source_image_sha256"]:
                raise ValueError("packet source image mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"invalid index entry: {exc}")
    expected_frames = set(range(expected.expected_frame_count))
    missing = sorted(expected_frames - seen_frames)
    extra_ids = sorted(seen_frames - expected_frames)
    if missing:
        errors.append(f"missing frame_ids: {_format_id_ranges(missing)}")
    if extra_ids:
        errors.append(f"extra frame_ids: {_format_id_ranges(extra_ids)}")
    observed_packets = set((root / "frames").glob("*.msgpack.zst")) if (root / "frames").is_dir() else set()
    for extra in sorted(observed_packets - referenced):
        errors.append(f"extra packet: {extra.name}")
    return CacheValidation(not errors, tuple(errors), len(entries))


def _require_packet_identity(packet: SemanticFramePacket, manifest: CacheManifest) -> None:
    observed = (
        packet.schema,
        packet.study_id,
        packet.sequence_id,
        packet.prompt_sha256,
        packet.model_manifest_sha256,
        packet.inference_config_sha256,
    )
    expected = (
        manifest.schema,
        manifest.study_id,
        manifest.sequence_id,
        manifest.prompt_sha256,
        manifest.model_manifest_sha256,
        manifest.inference_config_sha256,
    )
    if observed != expected:
        raise ValueError("packet identity mismatch")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_bytes() if path.exists() else b""
    row = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    temporary = path.parent / f".{path.name}.partial"
    try:
        with temporary.open("wb") as stream:
            stream.write(previous)
            stream.write(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank index row {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"index row {line_number} is not an object")
        entries.append(value)
    return entries


def _require_unique_index(entries: list[dict[str, Any]]) -> None:
    frames: set[int] = set()
    timestamps: set[float] = set()
    for entry in entries:
        frame_id = int(entry["frame_id"])
        timestamp = float(entry["timestamp"])
        if frame_id in frames:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        if timestamp in timestamps:
            raise ValueError(f"duplicate timestamp: {timestamp}")
        frames.add(frame_id)
        timestamps.add(timestamp)


def _format_id_ranges(values: list[int]) -> str:
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)
