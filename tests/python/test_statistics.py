from __future__ import annotations

import pytest

from semantic_py.openvocab_slam.experiments import SEEDS, SEQUENCE_IDS, paired_statistics
from tools.analyze_study import _classification


def _rows(delta: float = -0.02) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence_index, sequence in enumerate(SEQUENCE_IDS):
        for seed_index, seed in enumerate(SEEDS):
            baseline = 0.1 + 0.01 * sequence_index + 0.001 * seed_index
            for mode, ate in (("baseline", baseline), ("semantic-feedback", baseline + delta)):
                rows.append({
                    "sequence_id": sequence,
                    "seed": seed,
                    "mode": mode,
                    "ate_translation_rmse_m": ate,
                })
    return rows


def test_paired_difference_is_semantic_minus_baseline_for_every_sequence() -> None:
    result = paired_statistics(_rows(-0.02))
    assert tuple(result["sequences"]) == SEQUENCE_IDS
    assert len(result["pairs"]) == 30
    assert all(pair["ate_delta_m"] == pytest.approx(-0.02) for pair in result["pairs"])
    assert result["overall"]["median_ate_delta_m"] == pytest.approx(-0.02)
    assert result["overall"]["bootstrap_ci95_m"] == pytest.approx([-0.02, -0.02])


def test_bootstrap_is_reproducible() -> None:
    first = paired_statistics(_rows(-0.01))
    second = paired_statistics(_rows(-0.01))
    assert first["overall"] == second["overall"]
    assert first["bootstrap"] == second["bootstrap"]


def test_pairing_rejects_duplicate_or_missing_seed() -> None:
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        paired_statistics(rows)
    with pytest.raises(ValueError, match="exactly 60"):
        paired_statistics(_rows()[:-1])


def test_pairing_requires_all_frozen_sequences() -> None:
    rows = [row for row in _rows() if row["sequence_id"] != SEQUENCE_IDS[-1]]
    with pytest.raises(ValueError, match="exactly 60|sequences"):
        paired_statistics(rows)


def _classification_fixture(overall_ci: list[float], sequence_cis: list[list[float]]) -> dict[str, object]:
    return {
        "overall": {"bootstrap_ci95_m": overall_ci},
        "sequences": {
            sequence: {
                "median_ate_delta_m": 1.0,
                "bootstrap_ci95_m": interval,
            }
            for sequence, interval in zip(SEQUENCE_IDS, sequence_cis, strict=True)
        },
    }


@pytest.mark.parametrize(
    ("overall", "sequences", "expected"),
    [
        ([-0.2, -0.1], [[0.1, 0.2]] * 6, "improvement"),
        ([0.1, 0.2], [[-0.2, -0.1]] * 6, "negative"),
        ([-0.1, 0.1], [[-0.2, 0.2]] * 6, "neutral"),
        ([-0.1, 0.1], [[-0.2, 0.2]] * 5 + [[0.1, 0.2]], "mixed"),
    ],
)
def test_classification_follows_frozen_documented_ci_rule(
    overall: list[float], sequences: list[list[float]], expected: str
) -> None:
    assert _classification(_classification_fixture(overall, sequences)) == expected
