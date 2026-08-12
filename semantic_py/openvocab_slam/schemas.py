from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _require_sha256(name: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _validate_rle(rle: dict[str, object]) -> tuple[int, int, list[int]]:
    if set(rle) != {"size", "counts"}:
        raise ValueError("mask RLE must contain exactly size and counts")
    size = rle["size"]
    counts = rle["counts"]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("mask RLE size must be [height, width]")
    height, width = (int(size[0]), int(size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("mask RLE dimensions must be positive")
    if not isinstance(counts, (list, tuple)) or not counts:
        raise ValueError("mask RLE counts must be a nonempty sequence")
    normalized = [int(count) for count in counts]
    if any(count < 0 for count in normalized) or sum(normalized) != height * width:
        raise ValueError("mask RLE counts do not cover the image")
    return height, width, normalized


@dataclass(frozen=True)
class InstanceObservation:
    local_id: int
    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask_rle: dict[str, object]

    def __post_init__(self) -> None:
        if self.local_id < 0:
            raise ValueError("local_id must be nonnegative")
        if not self.label or self.label != self.label.strip() or self.label != self.label.lower():
            raise ValueError("label must be normalized")
        if not math.isfinite(float(self.score)) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and in [0, 1]")
        if len(self.box_xyxy) != 4 or not all(math.isfinite(float(value)) for value in self.box_xyxy):
            raise ValueError("box must contain four finite coordinates")
        if self.box_xyxy[2] <= self.box_xyxy[0] or self.box_xyxy[3] <= self.box_xyxy[1]:
            raise ValueError("box must have positive area")
        _validate_rle(self.mask_rle)

    def to_primitive(self) -> dict[str, object]:
        return {
            "local_id": self.local_id,
            "label": self.label,
            "score": self.score,
            "box_xyxy": list(self.box_xyxy),
            "mask_rle": {
                "size": list(self.mask_rle["size"]),
                "counts": list(self.mask_rle["counts"]),
            },
        }

    @classmethod
    def from_primitive(cls, value: dict[str, Any]) -> "InstanceObservation":
        return cls(
            local_id=int(value["local_id"]),
            label=str(value["label"]),
            score=float(value["score"]),
            box_xyxy=tuple(float(item) for item in value["box_xyxy"]),
            mask_rle={
                "size": [int(item) for item in value["mask_rle"]["size"]],
                "counts": [int(item) for item in value["mask_rle"]["counts"]],
            },
        )


@dataclass(frozen=True)
class SemanticFramePacket:
    schema: str
    study_id: str
    sequence_id: str
    frame_id: int
    timestamp: float
    source_image_sha256: str
    image_width: int
    image_height: int
    prompt_sha256: str
    model_manifest_sha256: str
    inference_config_sha256: str
    inference_time_seconds: float
    instances: tuple[InstanceObservation, ...]

    def __post_init__(self) -> None:
        if not self.schema or not self.study_id or not self.sequence_id:
            raise ValueError("packet identity fields must not be empty")
        if self.frame_id < 0:
            raise ValueError("frame_id must be nonnegative")
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if not math.isfinite(float(self.inference_time_seconds)) or self.inference_time_seconds < 0:
            raise ValueError("inference time must be finite and nonnegative")
        for name in (
            "source_image_sha256",
            "prompt_sha256",
            "model_manifest_sha256",
            "inference_config_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if tuple(instance.local_id for instance in self.instances) != tuple(range(len(self.instances))):
            raise ValueError("instance local_ids must be contiguous from zero")
        for instance in self.instances:
            height, width, _ = _validate_rle(instance.mask_rle)
            if (width, height) != (self.image_width, self.image_height):
                raise ValueError("instance mask dimensions do not match packet image")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "study_id": self.study_id,
            "sequence_id": self.sequence_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "source_image_sha256": self.source_image_sha256,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "prompt_sha256": self.prompt_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "inference_config_sha256": self.inference_config_sha256,
            "inference_time_seconds": self.inference_time_seconds,
            "instances": [instance.to_primitive() for instance in self.instances],
        }

    @classmethod
    def from_primitive(cls, value: dict[str, Any]) -> "SemanticFramePacket":
        required = {
            "schema", "study_id", "sequence_id", "frame_id", "timestamp",
            "source_image_sha256", "image_width", "image_height", "prompt_sha256",
            "model_manifest_sha256", "inference_config_sha256",
            "inference_time_seconds", "instances",
        }
        if set(value) != required:
            raise ValueError("packet fields do not match schema")
        return cls(
            schema=str(value["schema"]),
            study_id=str(value["study_id"]),
            sequence_id=str(value["sequence_id"]),
            frame_id=int(value["frame_id"]),
            timestamp=float(value["timestamp"]),
            source_image_sha256=str(value["source_image_sha256"]),
            image_width=int(value["image_width"]),
            image_height=int(value["image_height"]),
            prompt_sha256=str(value["prompt_sha256"]),
            model_manifest_sha256=str(value["model_manifest_sha256"]),
            inference_config_sha256=str(value["inference_config_sha256"]),
            inference_time_seconds=float(value["inference_time_seconds"]),
            instances=tuple(InstanceObservation.from_primitive(item) for item in value["instances"]),
        )


@dataclass(frozen=True)
class CacheManifest:
    schema: str
    study_id: str
    sequence_id: str
    source_tree_sha256: str
    association_sha256: str
    prompt_sha256: str
    model_manifest_sha256: str
    inference_config_sha256: str
    producer_commit: str
    image_long_side: int
    expected_frame_count: int
    resolution_fallback: str | None

    def __post_init__(self) -> None:
        if not self.schema or not self.study_id or not self.sequence_id:
            raise ValueError("cache identity fields must not be empty")
        for name in (
            "source_tree_sha256", "association_sha256", "prompt_sha256",
            "model_manifest_sha256", "inference_config_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not _GIT_SHA.fullmatch(self.producer_commit):
            raise ValueError("producer_commit must be a lowercase 40-character Git SHA")
        if self.image_long_side <= 0 or self.expected_frame_count <= 0:
            raise ValueError("cache dimensions and frame count must be positive")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "study_id": self.study_id,
            "sequence_id": self.sequence_id,
            "source_tree_sha256": self.source_tree_sha256,
            "association_sha256": self.association_sha256,
            "prompt_sha256": self.prompt_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "inference_config_sha256": self.inference_config_sha256,
            "producer_commit": self.producer_commit,
            "image_long_side": self.image_long_side,
            "expected_frame_count": self.expected_frame_count,
            "resolution_fallback": self.resolution_fallback,
        }

    @classmethod
    def from_primitive(cls, value: dict[str, Any]) -> "CacheManifest":
        required = {
            "schema", "study_id", "sequence_id", "source_tree_sha256",
            "association_sha256", "prompt_sha256", "model_manifest_sha256",
            "inference_config_sha256", "producer_commit", "image_long_side",
            "expected_frame_count", "resolution_fallback",
        }
        if set(value) != required:
            raise ValueError("cache manifest fields do not match schema")
        return cls(
            schema=str(value["schema"]),
            study_id=str(value["study_id"]),
            sequence_id=str(value["sequence_id"]),
            source_tree_sha256=str(value["source_tree_sha256"]),
            association_sha256=str(value["association_sha256"]),
            prompt_sha256=str(value["prompt_sha256"]),
            model_manifest_sha256=str(value["model_manifest_sha256"]),
            inference_config_sha256=str(value["inference_config_sha256"]),
            producer_commit=str(value["producer_commit"]),
            image_long_side=int(value["image_long_side"]),
            expected_frame_count=int(value["expected_frame_count"]),
            resolution_fallback=value["resolution_fallback"],
        )


def encode_binary_mask_rle(mask):
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    height, width = array.shape
    if height <= 0 or width <= 0:
        raise ValueError("mask must not be empty")
    flat = array.ravel(order="F")
    counts: list[int] = []
    current = False
    run = 0
    for value in flat:
        bit = bool(value)
        if bit == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = bit
    counts.append(run)
    return {"size": [height, width], "counts": counts}


def decode_binary_mask_rle(rle):
    height, width, counts = _validate_rle(rle)
    flat = np.empty(height * width, dtype=np.bool_)
    offset = 0
    value = False
    for count in counts:
        flat[offset : offset + count] = value
        offset += count
        value = not value
    return flat.reshape((height, width), order="F")
