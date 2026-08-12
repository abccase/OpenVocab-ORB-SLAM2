#!/usr/bin/env python3
"""Reject P01 changes that cross the compatibility-only source boundary."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


FROZEN_UPSTREAM_COMMIT = "f2e6f51cdc8d067655d90a78c06261378e07e8f3"
PROTECTED_PATHS = (
    "src/Tracking.cc",
    "src/Optimizer.cc",
    "src/ORBmatcher.cc",
    "src/LocalMapping.cc",
    "src/LoopClosing.cc",
    "include/LoopClosing.h",
)


def _is_allowed_change(path: str, line: str) -> bool:
    content = line[1:].strip()
    if path == "src/Tracking.cc":
        if line.startswith("+") and content in {"#include<unistd.h>", "#include <unistd.h>"}:
            return True
        return "cvtColor(" in content and (
            "CV_RGB2GRAY" in content
            or "CV_BGR2GRAY" in content
            or "CV_RGBA2GRAY" in content
            or "CV_BGRA2GRAY" in content
            or "cv::COLOR_RGB2GRAY" in content
            or "cv::COLOR_BGR2GRAY" in content
            or "cv::COLOR_RGBA2GRAY" in content
            or "cv::COLOR_BGRA2GRAY" in content
        )
    if path in {"src/LocalMapping.cc", "src/LoopClosing.cc"}:
        return line.startswith("+") and content == "#include <unistd.h>"
    if path == "include/LoopClosing.h":
        return "Eigen::aligned_allocator<std::pair<" in content and (
            "const KeyFrame*" in content or "KeyFrame* const" in content
        )
    return False


def audit_patch(patch: str) -> list[str]:
    violations: list[str] = []
    current_path: str | None = None
    for line in patch.splitlines():
        if line.startswith("diff --git a/"):
            fields = line.split()
            current_path = fields[2][2:] if len(fields) >= 3 else None
            continue
        if current_path not in PROTECTED_PATHS:
            continue
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        if not _is_allowed_change(current_path, line):
            violations.append(f"{current_path}: {line}")
    return violations


def collect_patch(root: Path, base: str) -> str:
    completed = subprocess.run(
        ["git", "diff", "--unified=0", base, "--", *PROTECTED_PATHS],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git diff failed")
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base", default=FROZEN_UPSTREAM_COMMIT)
    args = parser.parse_args()

    violations = audit_patch(collect_patch(args.root.resolve(), args.base))
    if violations:
        for violation in violations:
            print(f"VIOLATION: {violation}")
        print(f"COMPATIBILITY_AUDIT: FAIL ({len(violations)} violations)")
        return 1
    print("COMPATIBILITY_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
