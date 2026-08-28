#!/usr/bin/env python3
"""Generate the byte-identical P06 cross-language MessagePack fixtures."""

import argparse
from pathlib import Path

import cv2
import msgpack
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN_ID = "p06-fixture-run"
PROMPT_SHA256 = "1" * 64
MODEL_SHA256 = "2" * 64


def request() -> dict[str, object]:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 1] = (10, 20, 30)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("fixture JPEG encoding failed")
    return {
        "protocol_version": 1,
        "kind": "frame_request",
        "run_id": RUN_ID,
        "frame_id": 7,
        "source_timestamp_ns": 1_000_000_000,
        "prompt_sha256": PROMPT_SHA256,
        "image_width": 3,
        "image_height": 2,
        "jpeg_bytes": encoded.tobytes(),
    }


def packet() -> dict[str, object]:
    return {
        "protocol_version": 1,
        "kind": "semantic_packet",
        "run_id": RUN_ID,
        "frame_id": 7,
        "source_timestamp_ns": 1_000_000_000,
        "produced_timestamp_ns": 1_100_000_000,
        "prompt_sha256": PROMPT_SHA256,
        "model_manifest_sha256": MODEL_SHA256,
        "image_width": 3,
        "image_height": 2,
        "inference_ms": 45.0,
        "instances": [
            {
                "local_id": 0,
                "label": "person",
                "score": 0.9,
                "box_xyxy": [0.0, 0.0, 2.0, 1.0],
                "mask_rle": {"size": [2, 3], "counts": [1, 2, 3]},
            }
        ],
    }


def write(root: Path, name: str, value: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(msgpack.packb(value, use_bin_type=True))


def generate(root: Path = ROOT) -> None:
    valid_request = request()
    valid_packet = packet()
    write(root, "valid_frame_request.msgpack", valid_request)
    write(root, "valid_semantic_packet.msgpack", valid_packet)
    write(root, "wrong_version.msgpack", {**valid_packet, "protocol_version": 2})
    write(root, "wrong_run_id.msgpack", {**valid_packet, "run_id": "another-run"})
    write(root, "stale_timestamp.msgpack", {**valid_packet, "source_timestamp_ns": 900_000_000})
    write(
        root,
        "malformed_rle.msgpack",
        {
            **valid_packet,
            "instances": [
                {
                    **valid_packet["instances"][0],
                    "mask_rle": {"size": [2, 3], "counts": [1, 2]},
                }
            ],
        },
    )
    write(root, "wrong_dimensions.msgpack", {**valid_packet, "image_width": 4})
    write(root, "wrong_field_type.msgpack", {**valid_packet, "frame_id": "7"})
    write(root, "oversized_dimensions.msgpack", {
        **valid_packet, "image_width": 2**32 + 3,
    })
    write(root, "oversized_local_id.msgpack", {
        **valid_packet,
        "instances": [{**valid_packet["instances"][0], "local_id": 2**32}],
    })
    write(root, "oversized_rle_size.msgpack", {
        **valid_packet,
        "instances": [{
            **valid_packet["instances"][0],
            "mask_rle": {"size": [2**32 + 2, 3], "counts": [1, 2, 3]},
        }],
    })
    write(root, "invalid_jpeg_type.msgpack", {**valid_request, "jpeg_bytes": [1, 2]})
    write(
        root,
        "mismatched_source_timestamp.msgpack",
        {**valid_packet, "source_timestamp_ns": 1_010_000_000},
    )
    write(root, "mismatched_frame_id.msgpack", {**valid_packet, "frame_id": 8})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    generate(args.output_root)


if __name__ == "__main__":
    main()
