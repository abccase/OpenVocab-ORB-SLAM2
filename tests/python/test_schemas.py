from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from semantic_py.openvocab_slam.schemas import (
    InstanceObservation,
    SemanticFramePacket,
    decode_binary_mask_rle,
    encode_binary_mask_rle,
)


class SemanticSchemaTests(unittest.TestCase):
    def test_versioned_packet_fixture_is_schema_valid(self) -> None:
        fixture = Path(__file__).parents[1] / "fixtures/semantic_packets/minimal_v1.json"
        packet = SemanticFramePacket.from_primitive(json.loads(fixture.read_text()))

        self.assertEqual(packet.schema, "ovorb.semantic-cache.v1")
        self.assertEqual(packet.instances[0].label, "person")
        self.assertEqual(int(decode_binary_mask_rle(packet.instances[0].mask_rle).sum()), 2)

    def test_binary_mask_rle_round_trip_preserves_fortran_order(self) -> None:
        mask = np.array(
            [
                [False, True, True, False],
                [True, True, False, False],
                [False, False, True, False],
            ],
            dtype=np.bool_,
        )

        encoded = encode_binary_mask_rle(mask)

        self.assertEqual(encoded, {"size": [3, 4], "counts": [1, 1, 1, 2, 1, 1, 1, 1, 3]})
        np.testing.assert_array_equal(decode_binary_mask_rle(encoded), mask)

    def test_packet_round_trip_preserves_identity_and_mask(self) -> None:
        instance = InstanceObservation(
            local_id=0,
            label="person",
            score=0.875,
            box_xyxy=(0.0, 1.0, 3.0, 3.0),
            mask_rle={"size": [3, 4], "counts": [1, 1, 1, 2, 1, 1, 1, 1, 3]},
        )
        packet = SemanticFramePacket(
            schema="ovorb.semantic-cache.v1",
            study_id="ovorb2_tum_v1",
            sequence_id="tiny",
            frame_id=7,
            timestamp=1305031452.79172,
            source_image_sha256="1" * 64,
            image_width=4,
            image_height=3,
            prompt_sha256="2" * 64,
            model_manifest_sha256="3" * 64,
            inference_config_sha256="4" * 64,
            inference_time_seconds=0.125,
            instances=(instance,),
        )

        restored = SemanticFramePacket.from_primitive(packet.to_primitive())

        self.assertEqual(restored, packet)

    def test_packet_rejects_wrong_hash_dimensions_or_instance_ids(self) -> None:
        common = dict(
            schema="ovorb.semantic-cache.v1",
            study_id="ovorb2_tum_v1",
            sequence_id="tiny",
            frame_id=0,
            timestamp=1.0,
            source_image_sha256="1" * 64,
            image_width=4,
            image_height=3,
            prompt_sha256="2" * 64,
            model_manifest_sha256="3" * 64,
            inference_config_sha256="4" * 64,
            inference_time_seconds=0.1,
        )
        valid = InstanceObservation(0, "person", 0.9, (0.0, 0.0, 1.0, 1.0), {"size": [3, 4], "counts": [12]})
        invalid_id = InstanceObservation(2, "chair", 0.8, (1.0, 1.0, 2.0, 2.0), {"size": [3, 4], "counts": [12]})

        with self.assertRaises(ValueError):
            SemanticFramePacket(**{**common, "source_image_sha256": "bad", "instances": ()})
        with self.assertRaises(ValueError):
            SemanticFramePacket(**{**common, "image_width": 0, "instances": ()})
        with self.assertRaises(ValueError):
            SemanticFramePacket(**{**common, "instances": (valid, invalid_id)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
