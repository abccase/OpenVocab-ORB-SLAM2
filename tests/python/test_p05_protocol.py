import json
import tempfile
import unittest
from pathlib import Path

from semantic_py.openvocab_slam.p05_protocol import (
    STUDY_ID,
    expected_blocks,
    load_protocol,
    sha256_file,
    validate_batch_registration,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "config/P05_BASELINE_NONINFERIORITY_V3.json"
EXPERIMENT_PATH = ROOT / "config/EXPERIMENT_MANIFEST.yaml"
ORACLE_COMMIT = "58014b7c1f2b73427b67b4e80a8cf334127f48ea"

EXPECTED_FIRST_POSITIONS = {
    "fr1_desk": (
        "oracle", "oracle", "oracle", "candidate", "candidate",
        "oracle", "oracle", "candidate", "candidate", "oracle",
        "candidate", "oracle", "candidate", "candidate", "candidate",
    ),
    "fr1_room": (
        "oracle", "candidate", "candidate", "oracle", "oracle",
        "oracle", "candidate", "candidate", "oracle", "candidate",
        "oracle", "oracle", "oracle", "candidate", "candidate",
    ),
    "fr3_sitting_xyz": (
        "candidate", "oracle", "oracle", "oracle", "candidate",
        "candidate", "candidate", "candidate", "oracle", "candidate",
        "oracle", "candidate", "candidate", "oracle", "oracle",
    ),
    "fr3_sitting_halfsphere": (
        "candidate", "candidate", "oracle", "oracle", "candidate",
        "candidate", "oracle", "candidate", "oracle", "candidate",
        "oracle", "oracle", "oracle", "candidate", "candidate",
    ),
    "fr3_walking_xyz": (
        "oracle", "candidate", "oracle", "oracle", "oracle",
        "candidate", "candidate", "oracle", "candidate", "candidate",
        "candidate", "oracle", "candidate", "oracle", "oracle",
    ),
    "fr3_walking_halfsphere": (
        "oracle", "oracle", "candidate", "candidate", "oracle",
        "candidate", "candidate", "oracle", "oracle", "candidate",
        "oracle", "oracle", "candidate", "candidate", "candidate",
    ),
}


class P05ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _mutated_protocol(self, mutation) -> Path:
        value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        mutation(value)
        path = self.temp_path / "protocol.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_tracked_protocol_has_exact_hand_checked_order(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH, EXPERIMENT_PATH)
        blocks = protocol["blocks"]
        self.assertEqual(len(blocks), 90)
        self.assertEqual(sum(len(block["execution_order"]) for block in blocks), 180)
        self.assertEqual(blocks, expected_blocks())
        for sequence_id, expected_first in EXPECTED_FIRST_POSITIONS.items():
            sequence_blocks = [
                block for block in blocks if block["sequence_id"] == sequence_id
            ]
            self.assertEqual(
                tuple(block["execution_order"][0] for block in sequence_blocks),
                expected_first,
            )
            self.assertEqual(
                tuple(block["repetition_id"] for block in sequence_blocks),
                tuple(range(23011, 23026)),
            )
            self.assertEqual(
                sorted(expected_first.count(name) for name in ("oracle", "candidate")),
                [7, 8],
            )

    def test_rejects_each_mutated_frozen_scalar(self) -> None:
        cases = (
            (("study_id",), "wrong", "study"),
            (("oracle", "producer_commit"), "0" * 40, "oracle"),
            (("candidate", "producer_policy"), "CURRENT_HEAD", "candidate policy"),
            (("statistics", "resamples"), 99999, "statistics"),
            (("statistics", "seed"), 23011, "statistics"),
            (("statistics", "quantile_method"), "nearest", "statistics"),
            (("statistics", "pose_delta_lower_margin"), -0.09, "statistics"),
            (("statistics", "ate_geometric_ratio_upper_margin"), 1.24, "statistics"),
            (("metrics", "oracle_telemetry_timestamp_tolerance_seconds"), 0.0, "metrics"),
        )
        for keys, replacement, message in cases:
            with self.subTest(keys=keys):
                def mutate(value, keys=keys, replacement=replacement):
                    target = value
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = replacement

                path = self._mutated_protocol(mutate)
                with self.assertRaisesRegex(ValueError, message):
                    load_protocol(path, EXPERIMENT_PATH)

    def test_rejects_changed_repetitions_blocks_and_dataset_order(self) -> None:
        mutations = (
            (lambda value: value["repetition_ids"].pop(), "repetition"),
            (lambda value: value["blocks"].append(value["blocks"][0]), "blocks"),
            (lambda value: value["blocks"][0]["execution_order"].reverse(), "blocks"),
            (lambda value: value["sequence_ids"].reverse(), "sequence"),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                path = self._mutated_protocol(mutation)
                with self.assertRaisesRegex(ValueError, message):
                    load_protocol(path, EXPERIMENT_PATH)

    def test_rejects_experiment_dataset_mismatch(self) -> None:
        experiment = json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8"))
        experiment["datasets"][0]["id"] = "wrong_sequence"
        experiment_path = self.temp_path / "experiment.json"
        experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "experiment dataset"):
            load_protocol(PROTOCOL_PATH, experiment_path)

    def test_sha256_file_uses_file_bytes(self) -> None:
        path = self.temp_path / "value.bin"
        path.write_bytes(b"p05-v2\n")
        self.assertEqual(
            sha256_file(path),
            "45077a63467fdd2af82c4419c236044cbc2978de35c35718de10d25b4e4b95cb",
        )

    def test_batch_registration_requires_frozen_identities(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH, EXPERIMENT_PATH)
        protocol_sha256 = sha256_file(PROTOCOL_PATH)
        registration = {
            "schema_version": 1,
            "state": "REGISTERED",
            "study_id": STUDY_ID,
            "protocol_manifest_sha256": protocol_sha256,
            "candidate_policy": "HEAD_AT_REGISTRATION",
            "oracle": {
                "producer_commit": ORACLE_COMMIT,
                "executable_sha256": "1" * 64,
                "build_manifest_sha256": "2" * 64,
            },
            "candidate": {
                "producer_commit": "3" * 40,
                "executable_sha256": "4" * 64,
            },
        }
        validate_batch_registration(registration, protocol, protocol_sha256)
        for field_path, replacement in (
            (("state",), "RUNNING"),
            (("protocol_manifest_sha256",), "5" * 64),
            (("oracle", "producer_commit"), "6" * 40),
            (("candidate", "producer_commit"), "not-a-commit"),
            (("candidate", "executable_sha256"), "not-a-hash"),
        ):
            with self.subTest(field_path=field_path):
                changed = json.loads(json.dumps(registration))
                target = changed
                for key in field_path[:-1]:
                    target = target[key]
                target[field_path[-1]] = replacement
                with self.assertRaises(ValueError):
                    validate_batch_registration(changed, protocol, protocol_sha256)


if __name__ == "__main__":
    unittest.main()
