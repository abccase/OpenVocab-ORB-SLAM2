#!/usr/bin/env python3
"""Fail closed when a P05 candidate-baseline trace touches semantic inputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from semantic_py.openvocab_slam.p05_protocol import sha256_file  # noqa: E402


FILE_CALL = re.compile(
    r"^(?:\[pid\s+\d+\]\s+|\d+\s+)?"
    r"(?P<syscall>execve|openat2|openat|open|newfstatat|stat|lstat|"
    r"faccessat2|faccessat|access|readlinkat|readlink)"
    r"\([^\"\n]*\"(?P<path>(?:\\.|[^\"\\])*)\""
)


def _decoded_strace_path(raw: str) -> str:
    try:
        decoded = bytes(raw, "utf-8").decode("unicode_escape")
    except UnicodeError as exc:
        raise ValueError("trace contains an invalid escaped path") from exc
    if "\x00" in decoded:
        raise ValueError("trace contains a NUL path")
    return decoded


def _beneath(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def audit_trace(
    trace_paths: Sequence[Path],
    forbidden_roots: Sequence[Path],
    forbidden_files: Sequence[Path],
    cwd: Path = Path("/"),
) -> dict[str, object]:
    if not trace_paths:
        raise ValueError("at least one strace trace file is required")
    cwd = Path(cwd).resolve()
    roots = tuple(Path(path).resolve() for path in forbidden_roots)
    files = {Path(path).resolve() for path in forbidden_files}
    accesses: list[dict[str, object]] = []
    parsed_events = 0
    trace_identities: list[dict[str, object]] = []

    for supplied_path in trace_paths:
        trace_path = Path(supplied_path)
        if not trace_path.is_file():
            raise ValueError(f"strace trace file is missing: {trace_path}")
        try:
            lines = trace_path.read_text(encoding="utf-8", errors="strict").splitlines()
        except UnicodeError as exc:
            raise ValueError(f"strace trace is not UTF-8: {trace_path}") from exc
        trace_identities.append(
            {
                "path": str(trace_path.resolve()),
                "sha256": sha256_file(trace_path),
                "size_bytes": trace_path.stat().st_size,
            }
        )
        for line_number, line in enumerate(lines, 1):
            match = FILE_CALL.search(line)
            if match is None:
                continue
            parsed_events += 1
            decoded = _decoded_strace_path(match.group("path"))
            path = Path(decoded)
            resolved = (path if path.is_absolute() else cwd / path).resolve(strict=False)
            if resolved in files or any(_beneath(resolved, root) for root in roots):
                accesses.append(
                    {
                        "trace": str(trace_path.resolve()),
                        "line_number": line_number,
                        "syscall": match.group("syscall"),
                        "path": str(resolved),
                        "raw_line": line,
                    }
                )
    if accesses:
        raise ValueError(f"forbidden file access: {accesses}")
    return {
        "schema_version": 1,
        "valid": True,
        "cwd": str(cwd),
        "forbidden_roots": [str(path) for path in roots],
        "forbidden_files": sorted(str(path) for path in files),
        "forbidden_accesses": [],
        "parsed_file_events": parsed_events,
        "trace_files": trace_identities,
    }


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
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--forbidden-root", type=Path, action="append", default=[])
    parser.add_argument("--forbidden-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit_trace(
            args.trace,
            args.forbidden_root,
            args.forbidden_file,
            cwd=args.cwd,
        )
        if args.output is not None:
            _write_json_atomic(args.output, result)
    except (OSError, ValueError) as exc:
        print(f"P05_ACCESS_AUDIT_ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
