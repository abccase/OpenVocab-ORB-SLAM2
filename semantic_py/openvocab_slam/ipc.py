from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import cv2
import msgpack
import numpy as np

from .schemas import InstanceObservation


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_INT_MAX = 2**31 - 1


def create_service_sockets(context, *, request_endpoint: str, result_endpoint: str):
    import zmq

    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.RCVHWM, 1)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.connect(request_endpoint)
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.setsockopt(zmq.SNDHWM, 1)
    publisher.setsockopt(zmq.CONFLATE, 1)
    publisher.bind(result_endpoint)
    return subscriber, publisher


class ProtocolError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class FrameRequest:
    run_id: str
    frame_id: int
    source_timestamp_ns: int
    prompt_sha256: str
    image_width: int
    image_height: int
    jpeg_bytes: bytes


@dataclass(frozen=True)
class PacketExpectations:
    run_id: str
    prompt_sha256: str
    model_manifest_sha256: str
    image_width: int
    image_height: int
    current_timestamp_ns: int
    max_age_ns: int


@dataclass(frozen=True)
class SemanticPacket:
    run_id: str
    frame_id: int
    source_timestamp_ns: int
    produced_timestamp_ns: int
    prompt_sha256: str
    model_manifest_sha256: str
    image_width: int
    image_height: int
    inference_ms: float
    age_ns: int
    instances: tuple[InstanceObservation, ...]


class LatestFrameService:
    def __init__(
        self,
        *,
        run_id: str,
        prompt_sha256: str,
        model_manifest_sha256: str,
        infer,
        event_log: Path,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must not be empty")
        _require_sha("prompt_sha256", prompt_sha256)
        _require_sha("model_manifest_sha256", model_manifest_sha256)
        self.run_id = run_id
        self.prompt_sha256 = prompt_sha256
        self.model_manifest_sha256 = model_manifest_sha256
        self.infer = infer
        self.event_log = Path(event_log)

    def _record(self, event: dict[str, Any]) -> None:
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _prepare_payload(
        self, payload: bytes, *, produced_timestamp_ns: int | None = None
    ) -> tuple[bytes, dict[str, Any]]:
        receive_ns = time.monotonic_ns()
        try:
            request = unpack_frame_request(
                payload,
                expected_run_id=self.run_id,
                expected_prompt_sha256=self.prompt_sha256,
            )
            encoded = np.frombuffer(request.jpeg_bytes, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (request.image_height, request.image_width):
                raise ProtocolError("INVALID_JPEG", "decoded dimensions do not match request")
        except ProtocolError as error:
            self._record({
                "state": "REJECTED",
                "reason": error.code,
                "receive_monotonic_ns": receive_ns,
                "publish_monotonic_ns": time.monotonic_ns(),
            })
            raise

        inference_start_ns = time.monotonic_ns()
        try:
            instances = tuple(self.infer(image))
            inference_end_ns = time.monotonic_ns()
            for local_id, instance in enumerate(instances):
                if instance.local_id != local_id:
                    raise ProtocolError(
                        "INVALID_INSTANCE", "inference local IDs must be contiguous"
                    )
                size = instance.mask_rle.get("size")
                if size != [request.image_height, request.image_width]:
                    raise ProtocolError(
                        "MALFORMED_RLE", "inference mask dimensions mismatch"
                    )
            inference_ms = (inference_end_ns - inference_start_ns) / 1_000_000.0
            produced_ns = time.time_ns() if produced_timestamp_ns is None else int(
                produced_timestamp_ns
            )
            value = {
                "protocol_version": 1,
                "kind": "semantic_packet",
                "run_id": self.run_id,
                "frame_id": request.frame_id,
                "source_timestamp_ns": request.source_timestamp_ns,
                "produced_timestamp_ns": produced_ns,
                "prompt_sha256": self.prompt_sha256,
                "model_manifest_sha256": self.model_manifest_sha256,
                "image_width": request.image_width,
                "image_height": request.image_height,
                "inference_ms": inference_ms,
                "instances": [instance.to_primitive() for instance in instances],
            }
            result = msgpack.packb(value, use_bin_type=True)
        except Exception as error:
            self._record({
                "state": "INFERENCE_FAILED",
                "frame_id": request.frame_id,
                "source_timestamp_ns": request.source_timestamp_ns,
                "receive_monotonic_ns": receive_ns,
                "inference_start_monotonic_ns": inference_start_ns,
                "publish_monotonic_ns": time.monotonic_ns(),
                "error": str(error),
            })
            raise
        completed = {
            "state": "INFERENCE_COMPLETED",
            "frame_id": request.frame_id,
            "source_timestamp_ns": request.source_timestamp_ns,
            "receive_monotonic_ns": receive_ns,
            "inference_start_monotonic_ns": inference_start_ns,
            "inference_end_monotonic_ns": inference_end_ns,
            "inference_ms": inference_ms,
            "instance_count": len(instances),
        }
        self._record(completed)
        return result, completed

    def handle_payload(
        self, payload: bytes, *, produced_timestamp_ns: int | None = None
    ) -> bytes:
        result, _ = self._prepare_payload(
            payload, produced_timestamp_ns=produced_timestamp_ns
        )
        return result

    def serve_once(self, request_socket, result_socket) -> str:
        import zmq

        try:
            payload = request_socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return "NO_REQUEST"
        try:
            result, metadata = self._prepare_payload(payload)
        except ProtocolError:
            return "REJECTED"
        try:
            result_socket.send(result, flags=zmq.NOBLOCK)
        except zmq.Again:
            self._record({
                **metadata,
                "state": "RESULT_DROPPED",
                "publish_monotonic_ns": time.monotonic_ns(),
            })
            return "RESULT_DROPPED"
        self._record({
            **metadata,
            "state": "PUBLISHED",
            "publish_monotonic_ns": time.monotonic_ns(),
        })
        return "PUBLISHED"


def _unpack(payload: bytes) -> dict[str, Any]:
    try:
        value = msgpack.unpackb(payload, raw=False, strict_map_key=True)
    except (ValueError, TypeError, msgpack.ExtraData, msgpack.FormatError,
            msgpack.StackError, msgpack.OutOfData) as error:
        raise ProtocolError("CORRUPT_MESSAGEPACK", str(error)) from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProtocolError("CORRUPT_MESSAGEPACK", "top-level value must be a string-keyed map")
    return value


def _exact_fields(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ProtocolError("WRONG_FIELDS", "wire fields do not match protocol")


def _identity(value: dict[str, Any], *, kind: str) -> None:
    if type(value.get("protocol_version")) is not int:
        raise ProtocolError("INVALID_PACKET", "protocol_version must be an integer")
    if value.get("protocol_version") != 1:
        raise ProtocolError("WRONG_PROTOCOL_VERSION")
    if not isinstance(value.get("kind"), str):
        raise ProtocolError("INVALID_PACKET", "kind must be a string")
    if value.get("kind") != kind:
        raise ProtocolError("WRONG_KIND")


def _require_sha(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProtocolError("INVALID_IDENTITY", f"{name} is not a lowercase SHA256")
    return value


def _require_integer(
    name: str,
    value: object,
    *,
    minimum: int = 0,
    maximum: int = _UINT64_MAX,
    error_code: str = "INVALID_PACKET",
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ProtocolError(
            error_code,
            f"{name} must be an integer in [{minimum}, {maximum}]",
        )
    return value


def _require_number(name: str, value: object) -> float:
    if type(value) not in (int, float):
        raise ProtocolError("INVALID_PACKET", f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProtocolError("INVALID_PACKET", f"{name} must be finite")
    return parsed


def unpack_frame_request(
    payload: bytes, *, expected_run_id: str, expected_prompt_sha256: str
) -> FrameRequest:
    value = _unpack(payload)
    _identity(value, kind="frame_request")
    _exact_fields(value, {
        "protocol_version", "kind", "run_id", "frame_id",
        "source_timestamp_ns", "prompt_sha256", "image_width",
        "image_height", "jpeg_bytes",
    })
    if not isinstance(value["run_id"], str):
        raise ProtocolError("INVALID_PACKET", "run_id must be a string")
    if value["run_id"] != expected_run_id:
        raise ProtocolError("WRONG_RUN_ID")
    if value["prompt_sha256"] != expected_prompt_sha256:
        raise ProtocolError("WRONG_PROMPT")
    _require_sha("prompt_sha256", value["prompt_sha256"])
    if not isinstance(value["jpeg_bytes"], bytes) or not value["jpeg_bytes"]:
        raise ProtocolError("INVALID_JPEG", "jpeg_bytes must be nonempty MessagePack bin")
    if not value["jpeg_bytes"].startswith(b"\xff\xd8"):
        raise ProtocolError("INVALID_JPEG", "JPEG start marker missing")
    frame_id = _require_integer("frame_id", value["frame_id"])
    timestamp = _require_integer("source_timestamp_ns", value["source_timestamp_ns"])
    width = _require_integer(
        "image_width", value["image_width"], minimum=1, maximum=_INT_MAX
    )
    height = _require_integer(
        "image_height", value["image_height"], minimum=1, maximum=_INT_MAX
    )
    return FrameRequest(
        str(value["run_id"]), frame_id, timestamp, str(value["prompt_sha256"]),
        width, height, value["jpeg_bytes"],
    )


def _instance(value: object, width: int, height: int, expected_id: int) -> InstanceObservation:
    if not isinstance(value, dict) or set(value) != {
        "local_id", "label", "score", "box_xyxy", "mask_rle"
    }:
        raise ProtocolError("INVALID_INSTANCE", "instance fields do not match protocol")
    if _require_integer(
        "local_id", value["local_id"], maximum=_UINT32_MAX,
        error_code="INVALID_INSTANCE",
    ) != expected_id:
        raise ProtocolError("INVALID_INSTANCE", "local IDs must be contiguous")
    rle = value["mask_rle"]
    if not isinstance(rle, dict) or set(rle) != {"size", "counts"}:
        raise ProtocolError("MALFORMED_RLE")
    size = rle["size"]
    counts = rle["counts"]
    if not isinstance(size, list) or len(size) != 2 or any(
        type(item) is not int or item < 1 or item > _INT_MAX for item in size
    ):
        raise ProtocolError("MALFORMED_RLE")
    if (
        size != [height, width]
        or not isinstance(counts, list) or not counts
        or any(
            type(item) is not int or item < 0 or item > _UINT32_MAX
            for item in counts
        )
        or sum(counts) != width * height
    ):
        raise ProtocolError("MALFORMED_RLE")
    if not isinstance(value["label"], str) or not value["label"]:
        raise ProtocolError("INVALID_INSTANCE", "label must be a nonempty string")
    score = _require_number("score", value["score"])
    if score < 0.0 or score > 1.0:
        raise ProtocolError("INVALID_INSTANCE", "score outside [0, 1]")
    box = value["box_xyxy"]
    if not isinstance(box, list) or len(box) != 4:
        raise ProtocolError("INVALID_INSTANCE", "box_xyxy must contain four numbers")
    parsed_box = tuple(_require_number("box_xyxy", item) for item in box)
    if parsed_box[2] <= parsed_box[0] or parsed_box[3] <= parsed_box[1]:
        raise ProtocolError("INVALID_INSTANCE", "box_xyxy has no area")
    try:
        return InstanceObservation.from_primitive(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("INVALID_INSTANCE", str(error)) from error


def unpack_semantic_packet(payload: bytes, expected: PacketExpectations) -> SemanticPacket:
    value = _unpack(payload)
    _identity(value, kind="semantic_packet")
    _exact_fields(value, {
        "protocol_version", "kind", "run_id", "frame_id",
        "source_timestamp_ns", "produced_timestamp_ns", "prompt_sha256",
        "model_manifest_sha256", "image_width", "image_height",
        "inference_ms", "instances",
    })
    if not isinstance(value["run_id"], str):
        raise ProtocolError("INVALID_PACKET", "run_id must be a string")
    if value["run_id"] != expected.run_id:
        raise ProtocolError("WRONG_RUN_ID")
    if value["prompt_sha256"] != expected.prompt_sha256:
        raise ProtocolError("WRONG_PROMPT")
    if value["model_manifest_sha256"] != expected.model_manifest_sha256:
        raise ProtocolError("WRONG_MODEL_MANIFEST")
    _require_sha("prompt_sha256", value["prompt_sha256"])
    _require_sha("model_manifest_sha256", value["model_manifest_sha256"])
    width = _require_integer(
        "image_width", value["image_width"], minimum=1, maximum=_INT_MAX
    )
    height = _require_integer(
        "image_height", value["image_height"], minimum=1, maximum=_INT_MAX
    )
    if (width, height) != (expected.image_width, expected.image_height):
        raise ProtocolError("WRONG_DIMENSIONS")
    frame_id = _require_integer("frame_id", value["frame_id"])
    source_timestamp_ns = _require_integer(
        "source_timestamp_ns", value["source_timestamp_ns"]
    )
    produced_timestamp_ns = _require_integer(
        "produced_timestamp_ns", value["produced_timestamp_ns"]
    )
    if produced_timestamp_ns < source_timestamp_ns:
        raise ProtocolError("INVALID_TIMESTAMP")
    if source_timestamp_ns > expected.current_timestamp_ns:
        raise ProtocolError("FUTURE_PACKET")
    age_ns = expected.current_timestamp_ns - source_timestamp_ns
    if age_ns > expected.max_age_ns:
        raise ProtocolError("STALE_PACKET")
    inference_ms = _require_number("inference_ms", value["inference_ms"])
    if inference_ms < 0.0:
        raise ProtocolError("INVALID_INFERENCE_TIME")
    raw_instances = value["instances"]
    if not isinstance(raw_instances, list):
        raise ProtocolError("INVALID_INSTANCE", "instances must be an array")
    instances = tuple(
        _instance(item, width, height, local_id)
        for local_id, item in enumerate(raw_instances)
    )
    return SemanticPacket(
        str(value["run_id"]), frame_id, source_timestamp_ns,
        produced_timestamp_ns, str(value["prompt_sha256"]),
        str(value["model_manifest_sha256"]), width, height, inference_ms,
        age_ns, instances,
    )
