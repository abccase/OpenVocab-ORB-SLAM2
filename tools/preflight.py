#!/usr/bin/env python3
"""Collect and validate the reproducible machine and repository preflight."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


FROZEN_UPSTREAM_COMMIT = "f2e6f51cdc8d067655d90a78c06261378e07e8f3"
REQUIRED_FREE_BYTES = 50 * 1024**3


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    code: str


def _run(command: Sequence[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, ""
    output = completed.stdout.strip() or completed.stderr.strip()
    return completed.returncode, output


def _os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.lower()] = value.strip().strip('"')
    return result


def _ram_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return 0


def _cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _git_facts(root: Path) -> dict[str, object]:
    inside_code, inside = _run(["git", "rev-parse", "--is-inside-work-tree"], root)
    head_code, head = _run(["git", "rev-parse", "HEAD"], root)
    branch_code, branch = _run(["git", "branch", "--show-current"], root)
    pin_code, _ = _run(
        ["git", "cat-file", "-e", f"{FROZEN_UPSTREAM_COMMIT}^{{commit}}"], root
    )
    ancestor_code, _ = _run(
        ["git", "merge-base", "--is-ancestor", FROZEN_UPSTREAM_COMMIT, "HEAD"], root
    )
    version_code, version = _run(["git", "--version"], root)
    return {
        "inside_worktree": inside_code == 0 and inside == "true",
        "head": head if head_code == 0 else None,
        "branch": branch if branch_code == 0 else None,
        "version": version if version_code == 0 else None,
        "frozen_commit": FROZEN_UPSTREAM_COMMIT,
        "frozen_commit_present": pin_code == 0,
        "frozen_commit_is_ancestor": ancestor_code == 0,
    }


def collect_preflight(root: Path) -> dict[str, object]:
    root = root.resolve()
    cmake_code, cmake = _run(["cmake", "--version"])
    compiler_code, compiler = _run(["c++", "--version"])
    nvidia_code, nvidia = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    gpu: str | None = None
    driver: str | None = None
    gpu_status = "NVIDIA_SMI_UNAVAILABLE"
    if nvidia_code == 0 and nvidia:
        first_gpu = nvidia.splitlines()[0]
        parts = [part.strip() for part in first_gpu.split(",", 1)]
        gpu = parts[0]
        driver = parts[1] if len(parts) == 2 else None
        gpu_status = "OK"

    return {
        "schema_version": 1,
        "root": str(root),
        "os": _os_release(),
        "disk_free_bytes": shutil.disk_usage(root).free,
        "python": platform.python_version(),
        "cmake": cmake.splitlines()[0] if cmake_code == 0 and cmake else None,
        "compiler": compiler.splitlines()[0] if compiler_code == 0 and compiler else None,
        "cpu": _cpu_name(),
        "ram_bytes": _ram_bytes(),
        "nvidia_driver": driver,
        "gpu": gpu,
        "gpu_status": gpu_status,
        "git": _git_facts(root),
    }


def validate_preflight(facts: dict[str, object]) -> CheckResult:
    os_facts = facts.get("os", {})
    if not isinstance(os_facts, dict) or os_facts.get("version_id") != "22.04":
        return CheckResult(False, "UNSUPPORTED_OS")
    if int(facts.get("disk_free_bytes", 0)) < REQUIRED_FREE_BYTES:
        return CheckResult(False, "DISK_BELOW_50_GIB")
    git_facts = facts.get("git", {})
    if not isinstance(git_facts, dict) or not git_facts.get("inside_worktree"):
        return CheckResult(False, "NOT_A_GIT_WORKTREE")
    if not git_facts.get("frozen_commit_present"):
        return CheckResult(False, "UPSTREAM_PIN_MISSING")
    if not git_facts.get("frozen_commit_is_ancestor"):
        return CheckResult(False, "UPSTREAM_PIN_NOT_ANCESTOR")
    return CheckResult(True, "OK")


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    facts = collect_preflight(args.root)
    result = validate_preflight(facts)
    payload = {"facts": facts, "validation": asdict(result)}
    write_json_atomic(args.output, payload)
    print(f"PREFLIGHT: {result.code}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
