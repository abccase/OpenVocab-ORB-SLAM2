#!/usr/bin/env python3
"""Build the frozen P05 legacy oracle in an isolated detached worktree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from semantic_py.openvocab_slam.p05_protocol import (  # noqa: E402
    ORACLE_COMMIT,
    sha256_file,
)


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def oracle_commands(
    repository: Path,
    build_root: Path,
    commit: str,
    jobs: int,
) -> list[list[str]]:
    del jobs
    source = build_root / "source"
    return [
        ["git", "-C", str(repository), "worktree", "add", "--detach",
         str(source), commit],
        ["git", "-C", str(source), "status", "--porcelain",
         "--untracked-files=no"],
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        ["bash", str(source / "build.sh")],
        ["git", "-C", str(source), "status", "--porcelain",
         "--untracked-files=no"],
    ]


def _invoke(
    runner: Runner,
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return runner(
        command,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        capture_output=True,
    )


def _version(
    runner: Runner,
    command: list[str],
    *,
    cwd: Path,
) -> str:
    completed = _invoke(runner, command, cwd=cwd)
    value = completed.stdout.strip()
    if not value:
        raise ValueError(f"version probe returned no output: {command}")
    return value.splitlines()[0]


def build_oracle(
    repository: Path,
    build_root: Path,
    commit: str,
    jobs: int,
    *,
    runner: Runner = subprocess.run,
    require_ignored: bool = True,
) -> dict[str, object]:
    repository = Path(repository).resolve()
    build_root = Path(build_root).resolve()
    if commit != ORACLE_COMMIT:
        raise ValueError("oracle commit differs from the frozen P05 identity")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs <= 0:
        raise ValueError("build job count must be a positive integer")
    if not repository.is_dir():
        raise ValueError(f"repository does not exist: {repository}")
    if build_root == repository or build_root in repository.parents:
        raise ValueError("build root cannot contain or equal the repository")
    if require_ignored:
        try:
            relative_build_root = build_root.relative_to(repository)
        except ValueError as exc:
            raise ValueError("build root must be inside the repository to verify ignore rules") from exc
        ignored = _invoke(
            runner,
            ["git", "-C", str(repository), "check-ignore", "-q",
             str(relative_build_root)],
            cwd=repository,
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError("build root is not ignored by Git")

    source = build_root / "source"
    manifest_path = build_root / "oracle_build_manifest.json"
    if manifest_path.exists():
        raise ValueError("oracle build manifest already exists; use a new build root")
    if source.exists() and any(source.iterdir()):
        raise ValueError("oracle source directory already exists and is nonempty")
    build_root.mkdir(parents=True, exist_ok=True)

    commands = oracle_commands(repository, build_root, commit, jobs)
    _invoke(runner, commands[0], cwd=repository)
    initial_status = _invoke(runner, commands[1], cwd=source).stdout.strip()
    if initial_status:
        raise ValueError(f"oracle worktree is dirty before build: {initial_status}")
    resolved_commit = _invoke(runner, commands[2], cwd=source).stdout.strip()
    if resolved_commit != commit:
        raise ValueError(
            f"oracle worktree commit mismatch: {resolved_commit} != {commit}"
        )

    environment = os.environ.copy()
    environment["ORB_SLAM2_BUILD_JOBS"] = str(jobs)
    try:
        _invoke(runner, commands[3], cwd=source, env=environment)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if isinstance(exc.stderr, str) else str(exc)
        raise ValueError(f"oracle build failed: {detail}") from exc

    final_status = _invoke(runner, commands[4], cwd=source).stdout.strip()
    if final_status:
        raise ValueError(f"oracle worktree is dirty after build: {final_status}")
    executable = source / "Examples/RGB-D/rgbd_tum"
    if not executable.is_file():
        raise ValueError(f"oracle executable is missing: {executable}")
    build_script = source / "build.sh"
    if not build_script.is_file():
        raise ValueError(f"oracle build recipe is missing: {build_script}")

    compiler = os.environ.get("CXX", "c++")
    versions = {
        "cmake": _version(runner, ["cmake", "--version"], cwd=source),
        "compiler": _version(runner, [compiler, "--version"], cwd=source),
        "opencv": _version(
            runner, ["pkg-config", "--modversion", "opencv4"], cwd=source
        ),
        "eigen": _version(
            runner, ["pkg-config", "--modversion", "eigen3"], cwd=source
        ),
        "openssl": _version(runner, ["openssl", "version"], cwd=source),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "state": "COMPLETED",
        "created_utc": _utc_now(),
        "repository": str(repository),
        "build_root": str(build_root),
        "source_worktree": str(source),
        "source_commit": resolved_commit,
        "worktree_clean": True,
        "build_script": {
            "path": str(build_script),
            "sha256": sha256_file(build_script),
        },
        "build": {
            "type": "Release",
            "viewer": False,
            "testing": True,
            "jobs": jobs,
            "configure_arguments": [
                "-DCMAKE_BUILD_TYPE=Release",
                "-DORB_SLAM2_BUILD_VIEWER=OFF",
                "-DBUILD_TESTING=ON",
            ],
        },
        "versions": versions,
        "executable": {
            "path": str(executable.resolve()),
            "sha256": sha256_file(executable),
            "size_bytes": executable.stat().st_size,
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--commit", default=ORACLE_COMMIT)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    try:
        manifest = build_oracle(
            args.repository,
            args.build_root,
            args.commit,
            args.jobs,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"P05_ORACLE_BUILD_ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
