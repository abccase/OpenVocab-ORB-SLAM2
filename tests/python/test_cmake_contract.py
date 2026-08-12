#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


PRODUCT_ROOT = Path(__file__).resolve().parents[2]


class CMakeContractTests(unittest.TestCase):
    def test_headless_configuration_registers_rgbd_api_test(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ovorb2-cmake-") as temporary:
            build_dir = Path(temporary) / "build"
            configured = subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(PRODUCT_ROOT),
                    "-B",
                    str(build_dir),
                    "-DORB_SLAM2_BUILD_VIEWER=OFF",
                    "-DBUILD_TESTING=ON",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                configured.returncode,
                0,
                configured.stdout + configured.stderr,
            )

            listed = subprocess.run(
                ["ctest", "--test-dir", str(build_dir), "-N"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(listed.returncode, 0, listed.stdout + listed.stderr)
            self.assertIn("rgbd_api", listed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
