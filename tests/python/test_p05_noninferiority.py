import math
import unittest
from unittest import mock

import numpy as np

from semantic_py.openvocab_slam.p05_noninferiority import (
    bootstrap_indices,
    evaluate_sequence,
    evaluate_study,
)
from semantic_py.openvocab_slam.p05_protocol import EXPECTED_STATISTICS, SEQUENCE_IDS


def paired_rows(
    pose_delta: float = 0.0,
    ate_ratio: float = 1.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    oracle = [
        {
            "repetition_id": repetition_id,
            "valid_pose_fraction": 0.8,
            "ate_rmse_m": 0.1,
        }
        for repetition_id in range(23011, 23026)
    ]
    candidate = [
        {
            "repetition_id": repetition_id,
            "valid_pose_fraction": 0.8 + pose_delta,
            "ate_rmse_m": 0.1 * ate_ratio,
        }
        for repetition_id in range(23011, 23026)
    ]
    return oracle, candidate


class P05NoninferiorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.statistics = dict(EXPECTED_STATISTICS)

    def test_pcg64_indices_match_frozen_literal_and_are_reproducible(self) -> None:
        first = bootstrap_indices(15, 100000, 23010)
        second = bootstrap_indices(15, 100000, 23010)
        self.assertEqual(first.dtype, np.dtype(np.int16))
        self.assertEqual(first.shape, (100000, 15))
        self.assertEqual(
            first[:2].tolist(),
            [
                [4, 14, 13, 5, 3, 2, 14, 9, 10, 14, 2, 3, 9, 13, 11],
                [5, 0, 4, 0, 11, 3, 13, 6, 0, 1, 7, 9, 12, 10, 12],
            ],
        )
        np.testing.assert_array_equal(first, second)

    def test_exact_noninferiority_boundaries_pass(self) -> None:
        oracle, candidate = paired_rows(-0.10, 1.25)
        result = evaluate_sequence(oracle, candidate, self.statistics)
        self.assertTrue(result["valid"])
        self.assertTrue(result["pose_pass"])
        self.assertTrue(result["ate_pass"])
        self.assertAlmostEqual(result["pose_lower_95"], -0.10, places=14)
        self.assertAlmostEqual(result["ate_ratio_upper_95"], 1.25, places=14)

    def test_values_immediately_outside_boundaries_fail(self) -> None:
        oracle, candidate = paired_rows(-0.100001, 1.250001)
        result = evaluate_sequence(oracle, candidate, self.statistics)
        self.assertFalse(result["valid"])
        self.assertFalse(result["pose_pass"])
        self.assertFalse(result["ate_pass"])

    def test_bootstrap_index_matrix_is_created_once_and_shared_by_metrics(self) -> None:
        oracle, candidate = paired_rows()
        for index, row in enumerate(candidate):
            row["valid_pose_fraction"] = 0.5 + index / 100.0
            row["ate_rmse_m"] = 0.1 * math.exp(index / 100.0)
        for row in oracle:
            row["valid_pose_fraction"] = 0.5
        fixed = np.asarray([[0] * 15, [14] * 15], dtype=np.int16)
        with mock.patch(
            "semantic_py.openvocab_slam.p05_noninferiority.bootstrap_indices",
            return_value=fixed,
        ) as indices:
            result = evaluate_sequence(oracle, candidate, self.statistics)
        indices.assert_called_once_with(15, 100000, 23010)
        self.assertAlmostEqual(result["pose_lower_95"], 0.007, places=14)
        self.assertAlmostEqual(
            result["ate_ratio_upper_95"], math.exp(0.133), places=14
        )

    def test_requires_exact_unique_paired_repetitions(self) -> None:
        oracle, candidate = paired_rows()
        cases = []
        cases.append((oracle[:-1], candidate, "exactly 15"))
        duplicate = [dict(row) for row in oracle]
        duplicate[-1]["repetition_id"] = 23011
        cases.append((duplicate, candidate, "duplicate repetition"))
        mismatched = [dict(row) for row in candidate]
        mismatched[-1]["repetition_id"] = 23026
        cases.append((oracle, mismatched, "repetition IDs"))
        for oracle_rows, candidate_rows, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_sequence(oracle_rows, candidate_rows, self.statistics)

    def test_rejects_invalid_pose_ate_and_statistics_domains(self) -> None:
        mutations = (
            ("valid_pose_fraction", -0.01, "valid-pose"),
            ("valid_pose_fraction", 1.01, "valid-pose"),
            ("valid_pose_fraction", math.nan, "valid-pose"),
            ("ate_rmse_m", 0.0, "ATE"),
            ("ate_rmse_m", -0.1, "ATE"),
            ("ate_rmse_m", math.inf, "ATE"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field, value=value):
                oracle, candidate = paired_rows()
                candidate[0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_sequence(oracle, candidate, self.statistics)
        oracle, candidate = paired_rows()
        changed_statistics = dict(self.statistics, resamples=99999)
        with self.assertRaisesRegex(ValueError, "statistics"):
            evaluate_sequence(oracle, candidate, changed_statistics)

    def test_report_contains_all_pairs_descriptive_summaries_and_metadata(self) -> None:
        oracle, candidate = paired_rows(0.01, 1.1)
        result = evaluate_sequence(oracle, candidate, self.statistics)
        self.assertEqual(len(result["paired_values"]), 15)
        self.assertEqual(result["paired_values"][0]["repetition_id"], 23011)
        self.assertEqual(result["bootstrap"]["generator"], "PCG64")
        self.assertEqual(result["bootstrap"]["resamples"], 100000)
        self.assertEqual(result["margins"], {
            "pose_delta_lower": -0.10,
            "ate_ratio_upper": 1.25,
        })
        self.assertAlmostEqual(
            result["unpaired_summaries"]["candidate"]["ate_rmse_m"]["mean"],
            0.11,
        )

    def test_study_requires_all_six_frozen_sequences(self) -> None:
        rows = paired_rows()
        complete = {sequence_id: rows for sequence_id in SEQUENCE_IDS}
        result = evaluate_study(complete, SEQUENCE_IDS, self.statistics)
        self.assertTrue(result["valid"])
        self.assertEqual(tuple(result["sequences"]), SEQUENCE_IDS)
        incomplete = dict(complete)
        incomplete.pop(SEQUENCE_IDS[-1])
        with self.assertRaisesRegex(ValueError, "sequences"):
            evaluate_study(incomplete, SEQUENCE_IDS, self.statistics)


if __name__ == "__main__":
    unittest.main()
