from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.build_p05_oracle import ORACLE_COMMIT, build_oracle, oracle_commands


class FakeRunner:
    def __init__(
        self,
        *,
        resolved_commit: str = ORACLE_COMMIT,
        status: str = "",
        create_executable: bool = True,
        fail_build: bool = False,
    ) -> None:
        self.resolved_commit = resolved_commit
        self.status = status
        self.create_executable = create_executable
        self.fail_build = fail_build
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, command, **kwargs):
        command = [str(value) for value in command]
        self.calls.append((command, kwargs))
        if len(command) >= 4 and command[0] == "git" and command[3] == "worktree":
            source = Path(command[-2])
            source.mkdir(parents=True)
            (source / "build.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[-3:] == ["status", "--porcelain", "--untracked-files=no"]:
            return subprocess.CompletedProcess(command, 0, stdout=self.status, stderr="")
        if command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=self.resolved_commit + "\n", stderr=""
            )
        if command[0] == "bash":
            if self.fail_build:
                raise subprocess.CalledProcessError(7, command, stderr="build failed")
            if self.create_executable:
                executable = Path(command[1]).parent / "Examples/RGB-D/rgbd_tum"
                executable.parent.mkdir(parents=True)
                executable.write_bytes(b"legacy-oracle-executable\n")
                executable.chmod(0o755)
            return subprocess.CompletedProcess(command, 0, stdout="built\n", stderr="")
        outputs = {
            ("cmake", "--version"): "cmake version 3.22.1\n",
            ("c++", "--version"): "g++ 11.4.0\n",
            ("pkg-config", "--modversion", "opencv4"): "4.5.4\n",
            ("pkg-config", "--modversion", "eigen3"): "3.4.0\n",
            ("openssl", "version"): "OpenSSL 3.0.2\n",
        }
        output = outputs.get(tuple(command))
        if output is None:
            raise AssertionError(f"unexpected command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")


class P05OracleBuilderTests(unittest.TestCase):
    def test_commands_create_detached_worktree_and_build_tracked_recipe(self) -> None:
        commands = oracle_commands(
            Path("/repo"), Path("/ignored/oracle"), ORACLE_COMMIT, 2
        )
        self.assertEqual(commands, [
            ["git", "-C", "/repo", "worktree", "add", "--detach",
             "/ignored/oracle/source", ORACLE_COMMIT],
            ["git", "-C", "/ignored/oracle/source", "status", "--porcelain",
             "--untracked-files=no"],
            ["git", "-C", "/ignored/oracle/source", "rev-parse", "HEAD"],
            ["bash", "/ignored/oracle/source/build.sh"],
            ["git", "-C", "/ignored/oracle/source", "status", "--porcelain",
             "--untracked-files=no"],
        ])

    def test_build_writes_complete_atomic_identity_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            build_root = repository / "artifacts/p05-v2/oracle"
            runner = FakeRunner()

            manifest = build_oracle(
                repository, build_root, ORACLE_COMMIT, 2, runner=runner,
                require_ignored=False,
            )

            manifest_path = build_root / "oracle_build_manifest.json"
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")), manifest
            )
            self.assertFalse((build_root / ".oracle_build_manifest.json.partial").exists())
            self.assertEqual(manifest["source_commit"], ORACLE_COMMIT)
            self.assertTrue(manifest["worktree_clean"])
            self.assertEqual(manifest["build"]["type"], "Release")
            self.assertFalse(manifest["build"]["viewer"])
            self.assertTrue(manifest["build"]["testing"])
            self.assertEqual(manifest["build"]["jobs"], 2)
            self.assertEqual(manifest["versions"]["opencv"], "4.5.4")
            self.assertEqual(
                manifest["executable"]["sha256"],
                hashlib.sha256(b"legacy-oracle-executable\n").hexdigest(),
            )
            build_calls = [call for call in runner.calls if call[0][0] == "bash"]
            self.assertEqual(build_calls[0][1]["env"]["ORB_SLAM2_BUILD_JOBS"], "2")

    def test_dirty_or_wrong_commit_worktree_is_rejected(self) -> None:
        cases = (
            (FakeRunner(status=" M Tracking.cc\n"), "dirty"),
            (FakeRunner(resolved_commit="0" * 40), "commit"),
        )
        for runner, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repository = root / "repository"
                repository.mkdir()
                build_root = repository / "artifacts/oracle"
                with self.assertRaisesRegex(ValueError, message):
                    build_oracle(
                        repository, build_root, ORACLE_COMMIT, 2,
                        runner=runner, require_ignored=False,
                    )
                self.assertFalse((build_root / "oracle_build_manifest.json").exists())

    def test_missing_executable_and_failed_build_publish_no_manifest(self) -> None:
        cases = (
            (FakeRunner(create_executable=False), "executable"),
            (FakeRunner(fail_build=True), "build failed"),
        )
        for runner, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repository = root / "repository"
                repository.mkdir()
                build_root = repository / "artifacts/oracle"
                with self.assertRaisesRegex((ValueError, subprocess.CalledProcessError), message):
                    build_oracle(
                        repository, build_root, ORACLE_COMMIT, 2,
                        runner=runner, require_ignored=False,
                    )
                self.assertFalse((build_root / "oracle_build_manifest.json").exists())
                self.assertFalse(
                    (build_root / ".oracle_build_manifest.json.partial").exists()
                )

    def test_nonempty_source_and_unsafe_build_root_are_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            build_root = repository / "artifacts/oracle"
            source = build_root / "source"
            source.mkdir(parents=True)
            (source / "unexpected").write_text("data", encoding="utf-8")
            runner = FakeRunner()
            with self.assertRaisesRegex(ValueError, "source directory"):
                build_oracle(
                    repository, build_root, ORACLE_COMMIT, 2,
                    runner=runner, require_ignored=False,
                )
            self.assertEqual(runner.calls, [])
            with self.assertRaisesRegex(ValueError, "build root"):
                build_oracle(
                    repository, repository, ORACLE_COMMIT, 2,
                    runner=runner, require_ignored=False,
                )

    def test_only_frozen_commit_and_positive_job_count_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            for commit, jobs, message in (
                ("0" * 40, 2, "oracle commit"),
                (ORACLE_COMMIT, 0, "job count"),
            ):
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        build_oracle(
                            repository, repository / "artifacts/oracle",
                            commit, jobs, runner=FakeRunner(), require_ignored=False,
                        )


if __name__ == "__main__":
    unittest.main()
