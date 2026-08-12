#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.preflight import (
    FROZEN_UPSTREAM_COMMIT,
    REQUIRED_FREE_BYTES,
    collect_preflight,
    validate_preflight,
    write_json_atomic,
)


PRODUCT_ROOT = Path(__file__).resolve().parents[2]


def valid_facts() -> dict[str, object]:
    return {
        "os": {"version_id": "22.04", "pretty_name": "Ubuntu 22.04.5 LTS"},
        "disk_free_bytes": REQUIRED_FREE_BYTES,
        "python": "Python 3.10.12",
        "cmake": "cmake version 3.22.1",
        "compiler": "g++ 11.4.0",
        "cpu": "x86_64",
        "ram_bytes": 64 * 1024**3,
        "nvidia_driver": None,
        "gpu": None,
        "gpu_status": "NVIDIA_SMI_UNAVAILABLE",
        "git": {
            "inside_worktree": True,
            "head": FROZEN_UPSTREAM_COMMIT,
            "frozen_commit": FROZEN_UPSTREAM_COMMIT,
            "frozen_commit_present": True,
            "frozen_commit_is_ancestor": True,
        },
    }


class PreflightTests(unittest.TestCase):
    def test_collect_preflight_reports_required_facts(self) -> None:
        facts = collect_preflight(PRODUCT_ROOT)
        required = {
            "os",
            "disk_free_bytes",
            "python",
            "cmake",
            "compiler",
            "cpu",
            "ram_bytes",
            "nvidia_driver",
            "gpu",
            "git",
        }
        self.assertTrue(required.issubset(facts), required - set(facts))
        self.assertEqual(facts["git"]["head"], FROZEN_UPSTREAM_COMMIT)

    def test_validate_preflight_rejects_non_ubuntu_2204(self) -> None:
        facts = valid_facts()
        facts["os"] = {"version_id": "24.04", "pretty_name": "Ubuntu 24.04 LTS"}
        result = validate_preflight(facts)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "UNSUPPORTED_OS")

    def test_validate_preflight_rejects_less_than_50_gib_free(self) -> None:
        facts = valid_facts()
        facts["disk_free_bytes"] = REQUIRED_FREE_BYTES - 1
        result = validate_preflight(facts)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "DISK_BELOW_50_GIB")

    def test_validate_preflight_rejects_unpinned_git_history(self) -> None:
        facts = copy.deepcopy(valid_facts())
        facts["git"]["frozen_commit_is_ancestor"] = False
        result = validate_preflight(facts)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "UPSTREAM_PIN_NOT_ANCESTOR")

    def test_validate_preflight_rejects_missing_upstream_commit(self) -> None:
        facts = copy.deepcopy(valid_facts())
        facts["git"]["frozen_commit_present"] = False
        result = validate_preflight(facts)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "UPSTREAM_PIN_MISSING")

    def test_validate_preflight_accepts_missing_optional_gpu(self) -> None:
        result = validate_preflight(valid_facts())
        self.assertTrue(result.ok)
        self.assertEqual(result.code, "OK")

    def test_write_json_atomic_leaves_complete_json_without_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "preflight.json"
            write_json_atomic(output, {"validation": {"ok": True, "code": "OK"}})
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"validation": {"ok": True, "code": "OK"}},
            )
            self.assertEqual(list(output.parent.glob("*.partial")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
