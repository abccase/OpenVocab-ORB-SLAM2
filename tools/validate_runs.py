#!/usr/bin/env python3
"""Strictly validate the P08 60-run formal matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from semantic_py.openvocab_slam.experiments import STUDY_ID, read_run_matrix  # noqa: E402
from tools.run_study import (  # noqa: E402
    _read_json,
    collect_valid_attempts,
    validate_registration_current,
)


def validate_study(run_root: Path, expected: int = 60) -> dict[str, object]:
    registration = _read_json(Path(run_root) / "study_registration.json", "study registration")
    validate_registration_current(registration)
    rows = read_run_matrix(Path(run_root) / "run_matrix.csv")
    completed, invalid_attempts = collect_valid_attempts(run_root, registration, rows)
    missing = [
        {
            "order_index": row["order_index"],
            "sequence_id": row["sequence_id"],
            "mode": row["mode"],
            "seed": row["seed"],
        }
        for row in rows
        if int(row["order_index"]) not in completed
    ]
    valid = len(completed) == expected and not missing
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "expected": expected,
        "valid_count": len(completed),
        "invalid_attempt_count": len(invalid_attempts),
        "missing": missing,
        "valid": valid,
        "run_order_sha256": registration["run_order_sha256"],
        "producer_commit": registration["producer_commit"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=REPOSITORY_ROOT / "runs" / STUDY_ID)
    parser.add_argument("--expect", type=int, default=60)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_study(args.runs, args.expect)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.strict and not result["valid"]:
            return 1
    except (KeyError, OSError, ValueError) as exc:
        print(f"P08_VALIDATE INVALID: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
