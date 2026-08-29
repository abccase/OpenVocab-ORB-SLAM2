#!/usr/bin/env python3
"""Fail-closed clean-checkout reproduction of approved OpenVocab-ORB-SLAM2 evidence.

This command intentionally has no download, bootstrap, or network mode.  A
checkout contains source only; H01 data, frozen caches, runs, maps, and reports
must be supplied through ``--asset-root`` (normally the primary workspace).
All generated logs live under ``--output-root`` and are atomically written.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SEQUENCES = (
    "fr1_desk", "fr1_room", "fr3_sitting_halfsphere", "fr3_sitting_xyz",
    "fr3_walking_halfsphere", "fr3_walking_xyz",
)


@dataclass(frozen=True)
class ReproductionPlan:
    stages: list[str]


def load_reproduction_plan() -> ReproductionPlan:
    return ReproductionPlan([
        "preflight", "build", "unit", "data-validate", "cache-validate",
        "smoke", "metrics", "map-validate",
    ])


def resolve_asset_root(value: Path | None, repository_root: Path) -> Path:
    if value is None:
        raise ValueError("an explicit asset root (--asset-root) is required; this command never downloads H01 data")
    root = value.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"asset root does not exist: {root}")
    required = ("data/tum/manifests", "cache/semantic/v1", "cache/dynamic/v1", "runs", "reports/final", "artifacts/maps")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise ValueError("asset root is incomplete: " + ", ".join(missing))
    # It is permitted for validation in the primary checkout, but a clean
    # checkout must still name the external root explicitly.
    return root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected object")
    return value


def _command_stage(name: str, command: Sequence[str], cwd: Path, output_root: Path) -> dict[str, Any]:
    started = _utc_now()
    completed = subprocess.run(list(command), cwd=cwd, text=True, capture_output=True, check=False)
    log = output_root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "$ " + " ".join(command) + "\n\n[stdout]\n" + completed.stdout +
        "\n[stderr]\n" + completed.stderr, encoding="utf-8"
    )
    return {
        "name": name, "command": list(command), "started_utc": started,
        "ended_utc": _utc_now(), "exit_code": completed.returncode,
        "log": str(log.resolve()), "log_sha256": sha256_file(log),
        "ok": completed.returncode == 0,
    }


def _validate_data(asset_root: Path) -> dict[str, Any]:
    registrations = _read_json(asset_root / "runs/ovorb2_tum_v1/study_registration.json", "study registration")
    datasets = registrations.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(SEQUENCES):
        raise ValueError("formal registration does not bind all six datasets")
    facts: dict[str, Any] = {}
    for sequence in SEQUENCES:
        manifest = asset_root / "data/tum/manifests" / f"{sequence}.json"
        dataset = _read_json(manifest, f"dataset manifest {sequence}")
        sequence_root = asset_root / "data/tum/raw" / f"rgbd_dataset_freiburg{sequence[2:]}"
        association = sequence_root / "associate.txt"
        groundtruth = sequence_root / "groundtruth.txt"
        if not association.is_file() or not groundtruth.is_file():
            raise ValueError(f"missing H01 extracted input: {sequence}")
        expected = datasets[sequence]
        if sha256_file(manifest) != expected.get("dataset_manifest_sha256"):
            raise ValueError(f"dataset manifest hash mismatch: {sequence}")
        if dataset.get("sequence_id") != sequence:
            raise ValueError(f"dataset manifest sequence mismatch: {sequence}")
        facts[sequence] = {"manifest_sha256": sha256_file(manifest), "association_sha256": sha256_file(association), "groundtruth_sha256": sha256_file(groundtruth)}
    return facts


def _validate_caches(asset_root: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for sequence in SEQUENCES:
        semantic = asset_root / "cache/semantic/v1" / sequence
        dynamic = asset_root / "cache/dynamic/v1" / sequence
        for root, kind in ((semantic, "semantic"), (dynamic, "dynamic")):
            manifest = _read_json(root / "cache_manifest.json", f"{kind} cache manifest {sequence}")
            completion = _read_json(root / "cache_complete.json", f"{kind} cache completion {sequence}")
            index = root / "cache_index.jsonl"
            if not index.is_file() or not manifest.get("sequence_id") == sequence:
                raise ValueError(f"incomplete {kind} cache: {sequence}")
            if completion.get("manifest_sha256") != sha256_file(root / "cache_manifest.json"):
                raise ValueError(f"unbound {kind} cache completion: {sequence}")
            if completion.get("index_sha256") != sha256_file(index):
                raise ValueError(f"unbound {kind} cache index: {sequence}")
        facts[sequence] = {"semantic_manifest_sha256": sha256_file(semantic / "cache_manifest.json"), "dynamic_manifest_sha256": sha256_file(dynamic / "cache_manifest.json")}
    return facts


def _validate_metrics(asset_root: Path) -> dict[str, Any]:
    summary_path = asset_root / "reports/final/summary.json"
    summary = _read_json(summary_path, "P08 summary")
    if summary.get("outcome_classification") != "neutral":
        raise ValueError("P08 result classification is not the approved neutral result")
    if summary.get("invalid_attempt_count") != 0:
        raise ValueError("P08 summary contains invalid formal attempts")
    overall = summary.get("paired_statistics", {}).get("overall", {}) if isinstance(summary.get("paired_statistics"), dict) else {}
    if overall.get("pair_count") != 30:
        raise ValueError("P08 summary does not contain 30 paired results")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("P08 summary lacks artifact identities")
    identities: dict[str, str] = {}
    for name, item in artifacts.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"malformed P08 artifact identity: {name}")
        path = asset_root / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"P08 artifact hash mismatch: {name}")
        identities[name] = str(item["sha256"])
    return {"summary_sha256": sha256_file(summary_path), "artifacts": identities}


def _validate_maps(asset_root: Path, repository_root: Path, output_root: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for sequence in ("fr1_desk", "fr3_walking_xyz"):
        integrity = asset_root / "artifacts/maps" / f"{sequence}-integrity.json"
        value = _read_json(integrity, f"map integrity {sequence}")
        if value.get("valid") is not True or not isinstance(value.get("map_root"), str):
            raise ValueError(f"invalid map integrity: {sequence}")
        map_root = Path(str(value["map_root"]))
        if not map_root.is_dir():
            raise ValueError(f"map root absent: {map_root}")
        report = output_root / "map-validation" / f"{sequence}.json"
        stage = _command_stage("map-" + sequence, [sys.executable, str(repository_root / "tools/validate_map.py"), str(map_root), "--report", str(report)], repository_root, output_root)
        if not stage["ok"]:
            raise ValueError(f"map validator failed: {sequence}; see {stage['log']}")
        facts[sequence] = {"integrity_sha256": sha256_file(integrity), "validator": stage}
    return facts


def _smoke(repository_root: Path, asset_root: Path, output_root: Path) -> dict[str, Any]:
    executable_baseline = repository_root / "Examples/RGB-D/rgbd_tum"
    executable_semantic = repository_root / "Examples/RGB-D/rgbd_tum_ov"
    if not executable_baseline.is_file() or not executable_semantic.is_file():
        raise ValueError("smoke executables are absent; build stage did not produce RGB-D binaries")
    common = ["--sequence", "fr3_walking_xyz", "--seed", "23011", "--manifest", str(repository_root / "config/EXPERIMENT_MANIFEST.yaml"), "--data-root", str(asset_root / "data/tum/raw"), "--data-manifests", str(asset_root / "data/tum/manifests"), "--vocabulary", str(repository_root / "Vocabulary/ORBvoc.txt"), "--semantic-cache-root", str(asset_root / "cache/semantic/v1"), "--dynamic-cache-root", str(asset_root / "cache/dynamic/v1"), "--dynamic-cache-identities", str(repository_root / "config/DYNAMIC_CACHE_IDENTITY.json")]
    registry = output_root / "smoke/runs/registry.jsonl"
    results: dict[str, Any] = {}
    for mode, executable in (("baseline", executable_baseline), ("semantic-feedback", executable_semantic)):
        output = output_root / "smoke/runs" / mode
        command = [sys.executable, str(repository_root / "tools/run_orb_tum.py"), "--mode", mode, "--study", "smoke", "--executable", str(executable), "--output-root", str(output), "--registry", str(registry), *common]
        stage = _command_stage("smoke-" + mode, command, repository_root, output_root)
        if not stage["ok"]:
            raise ValueError(f"{mode} smoke failed; see {stage['log']}")
        manifests = sorted(output.glob("**/run_manifest.json"))
        if len(manifests) != 1:
            raise ValueError(f"{mode} smoke did not create exactly one manifest")
        manifest = _read_json(manifests[0], f"{mode} smoke manifest")
        if manifest.get("valid") is not True or manifest.get("mode") != mode or manifest.get("sequence_id") != "fr3_walking_xyz":
            raise ValueError(f"{mode} smoke manifest is invalid")
        results[mode] = {"manifest": str(manifests[0]), "manifest_sha256": sha256_file(manifests[0]), "stage": stage}
    return results


def run_reproduction(repository_root: Path, asset_root: Path, output_root: Path, *, validate_existing: bool, smoke: bool, build_dir: Path) -> dict[str, Any]:
    if not validate_existing and not smoke:
        raise ValueError("select --validate-existing and/or --smoke")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, Any]] = []
    stages.append({"name": "preflight", "ok": True, "asset_root": str(asset_root), "repository_root": str(repository_root), "no_download_path": True})
    if smoke:
        build_dir = build_dir.resolve()
        stages.append(_command_stage("build-configure", ["cmake", "-S", str(repository_root), "-B", str(build_dir)], repository_root, output_root))
        if stages[-1]["ok"]:
            stages.append(_command_stage("build", ["cmake", "--build", str(build_dir), "-j2"], repository_root, output_root))
        if not stages[-1]["ok"]:
            raise ValueError(f"build failed; see {stages[-1]['log']}")
        stages.append(_command_stage("unit", [sys.executable, "-m", "pytest", "tests/python", "-q"], repository_root, output_root))
        if not stages[-1]["ok"]:
            raise ValueError(f"unit tests failed; see {stages[-1]['log']}")
    else:
        stages.extend([{ "name": "build", "ok": True, "status": "not rerun by --validate-existing" }, { "name": "unit", "ok": True, "status": "not rerun by --validate-existing" }])
    data = _validate_data(asset_root)
    stages.append({"name": "data-validate", "ok": True, "facts": data})
    caches = _validate_caches(asset_root)
    stages.append({"name": "cache-validate", "ok": True, "facts": caches})
    smoke_facts: dict[str, Any] = {"status": "not requested"}
    if smoke:
        smoke_facts = _smoke(repository_root, asset_root, output_root)
    stages.append({"name": "smoke", "ok": True, "facts": smoke_facts})
    metrics = _validate_metrics(asset_root)
    stages.append({"name": "metrics", "ok": True, "facts": metrics})
    maps = _validate_maps(asset_root, repository_root, output_root)
    stages.append({"name": "map-validate", "ok": True, "facts": maps})
    expected = load_reproduction_plan().stages
    actual = ["build" if stage["name"] == "build-configure" else stage["name"] for stage in stages if stage["name"] in expected or stage["name"] == "build-configure"]
    # The emitted stage names preserve the contract while build-configure is a
    # subcommand recorded in the build stage's log.
    return {"schema_version": 1, "created_utc": _utc_now(), "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository_root, text=True).strip(), "asset_root": str(asset_root), "stages": stages, "contract_stages": expected, "contract_observed": actual, "valid": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True, help="existing ignored H01/cache/run/map/report root")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/openvocab-reproduction"))
    parser.add_argument("--build-dir", type=Path, default=REPOSITORY_ROOT / "build-reproduction")
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        assets = resolve_asset_root(args.asset_root, REPOSITORY_ROOT)
        report = run_reproduction(REPOSITORY_ROOT.resolve(), assets, args.output_root, validate_existing=args.validate_existing, smoke=args.smoke, build_dir=args.build_dir)
        destination = args.output_root.resolve() / "reproduction_manifest.json"
        write_json_atomic(destination, report)
        print(f"REPRODUCTION_VALID: {destination}")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"REPRODUCTION_INVALID: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
