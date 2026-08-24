from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.audit_p05_baseline_access import audit_trace


class P05AccessAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cwd = self.root / "run"
        self.cwd.mkdir()
        self.semantic_root = self.root / "repo/cache/semantic"
        self.dynamic_root = self.root / "repo/cache/dynamic"
        self.prompt = self.root / "repo/config/PROMPTS.yaml"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _trace(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_forbidden_cache_open_is_rejected_even_when_syscall_fails(self) -> None:
        trace = self._trace(
            "trace.123",
            f'openat(AT_FDCWD, "{self.semantic_root}/v1/index", O_RDONLY) = -1 ENOENT\n',
        )
        with self.assertRaisesRegex(ValueError, "forbidden file access"):
            audit_trace(
                [trace], [self.semantic_root, self.dynamic_root], [], cwd=self.cwd
            )

    def test_dataset_library_and_output_accesses_are_allowed(self) -> None:
        trace = self._trace(
            "trace.124",
            "openat(AT_FDCWD, \"/usr/lib/libopencv_core.so\", O_RDONLY) = 3\n"
            "openat(AT_FDCWD, \"/repo/data/tum/rgb.png\", O_RDONLY) = 4\n"
            "openat(AT_FDCWD, \"CameraTrajectory.txt\", O_WRONLY) = 5\n",
        )
        result = audit_trace(
            [trace], [self.semantic_root], [self.prompt], cwd=self.cwd
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["forbidden_accesses"], [])
        self.assertEqual(result["parsed_file_events"], 3)

    def test_forbidden_exact_files_relative_paths_and_escaped_paths_are_rejected(self) -> None:
        relative_forbidden = self.cwd / "config/PROMPTS.yaml"
        escaped_forbidden = self.root / "repo/config/semantic models.json"
        traces = (
            self._trace(
                "trace.relative",
                'newfstatat(AT_FDCWD, "config/PROMPTS.yaml", {st_mode=S_IFREG}, 0) = 0\n',
            ),
            self._trace(
                "trace.escaped",
                f'access("{str(escaped_forbidden).replace(" ", chr(92) + "040")}", R_OK) = 0\n',
            ),
        )
        with self.assertRaisesRegex(ValueError, "forbidden file access"):
            audit_trace(
                traces,
                [],
                [relative_forbidden, escaped_forbidden],
                cwd=self.cwd,
            )

    def test_pid_prefix_unfinished_and_multiple_ff_traces_are_parsed(self) -> None:
        first = self._trace(
            "trace.200",
            f'[pid 200] openat(AT_FDCWD, "{self.dynamic_root}/v1" <unfinished ...>\n',
        )
        second = self._trace(
            "trace.201",
            '<... openat resumed>, O_RDONLY) = 3\n'
            'readlink("/proc/self/exe", "/repo/bin", 4096) = 9\n',
        )
        with self.assertRaisesRegex(ValueError, "forbidden file access"):
            audit_trace(
                [first, second], [self.dynamic_root], [], cwd=self.cwd
            )

    def test_missing_trace_and_unparseable_nul_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "trace"):
            audit_trace([self.root / "missing"], [], [], cwd=self.cwd)
        trace = self.root / "trace.bad"
        trace.write_bytes(b'open("/tmp/\x00bad", O_RDONLY) = 3\n')
        with self.assertRaises((UnicodeError, ValueError)):
            audit_trace([trace], [], [], cwd=self.cwd)


if __name__ == "__main__":
    unittest.main()
