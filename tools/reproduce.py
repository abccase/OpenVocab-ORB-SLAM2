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
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from semantic_py.openvocab_slam.cache import validate_cache
from semantic_py.openvocab_slam.dynamic_cache import hash_dataset_tree
from semantic_py.openvocab_slam.schemas import CacheManifest
from semantic_py.openvocab_slam.experiments import read_run_matrix
from tools.run_study import validate_attempt


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


def resolve_python(asset_root: Path) -> Path:
    """Use the explicitly supplied semantic environment when it exists."""
    declared = asset_root / "venv/semantic-gpu/bin/python"
    return declared if declared.is_file() else Path(sys.executable)


def attach_asset_links(repository_root: Path, asset_root: Path) -> None:
    """Expose ignored external sources required by the full unit contract."""
    source = asset_root / "external"
    destination = repository_root / "external"
    if not source.is_dir():
        raise ValueError(f"asset root lacks external dependencies: {source}")
    if repository_root.resolve() == asset_root.resolve():
        return
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise ValueError(f"refusing to replace existing external path: {destination}")
    destination.symlink_to(source, target_is_directory=True)


def fresh_build_command(repository_root: Path, build_dir: Path) -> list[str]:
    """Use the project entrypoint because it builds generated g2o headers too."""
    del build_dir  # build.sh owns a complete isolated in-tree build graph.
    return ["bash", str(repository_root / "build.sh")]


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


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid {label}: expected JSON objects")
    return rows


def validate_index_payloads(root: Path, index_path: Path, label: str) -> list[dict[str, Any]]:
    """Validate every indexed payload, including its safe relative path and hash."""
    rows = _read_jsonl(index_path, f"{label} index")
    seen: set[Path] = set()
    for row in rows:
        relative = Path(str(row.get("path", "")))
        expected = row.get("sha256")
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe {label} payload path: {relative}")
        payload = root / relative
        if payload in seen:
            raise ValueError(f"duplicate {label} payload: {relative}")
        seen.add(payload)
        if not payload.is_file():
            raise ValueError(f"missing {label} payload: {relative}")
        if not isinstance(expected, str) or sha256_file(payload) != expected:
            raise ValueError(f"{label} payload hash mismatch: {relative}")
    return rows


def _command_stage(name: str, command: Sequence[str], cwd: Path, output_root: Path) -> dict[str, Any]:
    started = _utc_now()
    completed = subprocess.run(list(command), cwd=cwd, text=True, capture_output=True, check=False)
    log = output_root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".partial", dir=log.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n[stdout]\n" + completed.stdout + "\n[stderr]\n" + completed.stderr)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary_name, log)
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
        if hash_dataset_tree(sequence_root) != expected.get("source_tree_sha256"):
            raise ValueError(f"dataset source-tree hash mismatch: {sequence}")
        if sha256_file(association) != expected.get("association_sha256"):
            raise ValueError(f"dataset association hash mismatch: {sequence}")
        if sha256_file(groundtruth) != expected.get("groundtruth_sha256"):
            raise ValueError(f"dataset groundtruth hash mismatch: {sequence}")
        facts[sequence] = {"manifest_sha256": sha256_file(manifest), "association_sha256": sha256_file(association), "groundtruth_sha256": sha256_file(groundtruth)}
    return facts


def _validate_caches(asset_root: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for sequence in SEQUENCES:
        semantic = asset_root / "cache/semantic/v1" / sequence
        dynamic = asset_root / "cache/dynamic/v1" / sequence
        registration_dataset = _read_json(asset_root / "runs/ovorb2_tum_v1/study_registration.json", "study registration")["datasets"][sequence]
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
        semantic_manifest = CacheManifest.from_primitive(_read_json(semantic / "cache_manifest.json", "semantic cache manifest"))
        if (semantic_manifest.source_tree_sha256 != registration_dataset["source_tree_sha256"] or
                semantic_manifest.association_sha256 != registration_dataset["association_sha256"] or
                semantic_manifest.prompt_sha256 != registration_dataset["prompt_sha256"]):
            raise ValueError(f"semantic cache does not bind registration: {sequence}")
        semantic_result = validate_cache(semantic, semantic_manifest)
        if not semantic_result.valid:
            raise ValueError(f"semantic packet validation failed: {sequence}: {semantic_result.errors[0]}")
        dynamic_manifest = _read_json(dynamic / "cache_manifest.json", "dynamic cache manifest")
        if (dynamic_manifest.get("source_tree_sha256") != registration_dataset["source_tree_sha256"] or
                dynamic_manifest.get("association_sha256") != registration_dataset["association_sha256"] or
                dynamic_manifest.get("semantic_manifest_sha256") != sha256_file(semantic / "cache_manifest.json")):
            raise ValueError(f"dynamic cache does not bind registration: {sequence}")
        score_rows = validate_index_payloads(dynamic, dynamic / "cache_index.jsonl", "score map")
        track_rows = _read_jsonl(dynamic / "dynamic_tracks.jsonl", "dynamic tracks")
        diagnostic_rows = validate_index_payloads(dynamic, dynamic / "diagnostics_index.jsonl", "diagnostic")
        if len(score_rows) != semantic_manifest.expected_frame_count or len(track_rows) < 0:
            raise ValueError(f"dynamic cache frame/track count mismatch: {sequence}")
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


def _validate_formal_runs(asset_root: Path) -> dict[str, Any]:
    """Replay every registered P08 condition without rewriting the artifact root."""
    root = asset_root / "runs/ovorb2_tum_v1"
    registration = _read_json(root / "study_registration.json", "study registration")
    matrix = read_run_matrix(root / "run_matrix.csv")
    if len(matrix) != 60:
        raise ValueError("formal matrix does not contain exactly 60 conditions")
    valid = 0
    manifest_hashes: dict[str, str] = {}
    for condition in matrix:
        mode, sequence, seed = str(condition["mode"]), str(condition["sequence_id"]), int(condition["seed"])
        attempts = sorted((root / mode / sequence / f"seed-{seed}").glob("attempt-*"))
        if len(attempts) != 1:
            raise ValueError(f"formal run is missing or duplicated: {mode}/{sequence}/{seed}")
        ok, reason, _ = validate_attempt(attempts[0], condition, registration)
        if not ok:
            raise ValueError(f"formal run validation failed {mode}/{sequence}/{seed}: {reason}")
        manifest = attempts[0] / "run_manifest.json"
        manifest_hashes[f"{mode}/{sequence}/{seed}"] = sha256_file(manifest)
        valid += 1
    return {"registration_sha256": sha256_file(root / "study_registration.json"), "matrix_sha256": sha256_file(root / "run_matrix.csv"), "valid_runs": valid, "run_manifest_hashes": manifest_hashes}


def _validate_maps(asset_root: Path, repository_root: Path, output_root: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    expected_roots = {
        "fr1_desk": asset_root / "artifacts/maps/smoke-semantic-feedback-fr1_desk-seed-23011-attempt-002",
        "fr3_walking_xyz": asset_root / "artifacts/maps/smoke-semantic-feedback-fr3_walking_xyz-seed-23011-attempt-001",
    }
    for sequence in ("fr1_desk", "fr3_walking_xyz"):
        integrity = asset_root / "artifacts/maps" / f"{sequence}-integrity.json"
        value = _read_json(integrity, f"map integrity {sequence}")
        if value.get("valid") is not True or not isinstance(value.get("map_root"), str):
            raise ValueError(f"invalid map integrity: {sequence}")
        map_root = Path(str(value["map_root"]))
        if map_root.resolve() != expected_roots[sequence].resolve():
            raise ValueError(f"map integrity redirects expected root: {sequence}")
        if not map_root.is_dir():
            raise ValueError(f"map root absent: {map_root}")
        report = output_root / "map-validation" / f"{sequence}.json"
        stage = _command_stage("map-" + sequence, [sys.executable, str(repository_root / "tools/validate_map.py"), str(map_root), "--report", str(report)], repository_root, output_root)
        if not stage["ok"]:
            raise ValueError(f"map validator failed: {sequence}; see {stage['log']}")
        facts[sequence] = {"integrity_sha256": sha256_file(integrity), "validator": stage}
    return facts


def _smoke(repository_root: Path, asset_root: Path, output_root: Path, python: Path) -> dict[str, Any]:
    executable_baseline = repository_root / "Examples/RGB-D/rgbd_tum"
    executable_semantic = repository_root / "Examples/RGB-D/rgbd_tum_ov"
    if not executable_baseline.is_file() or not executable_semantic.is_file():
        raise ValueError("smoke executables are absent; build stage did not produce RGB-D binaries")
    common = ["--sequence", "fr3_walking_xyz", "--seed", "23011", "--manifest", str(repository_root / "config/EXPERIMENT_MANIFEST.yaml"), "--data-root", str(asset_root / "data/tum/raw"), "--data-manifests", str(asset_root / "data/tum/manifests"), "--vocabulary", str(repository_root / "Vocabulary/ORBvoc.txt"), "--semantic-cache-root", str(asset_root / "cache/semantic/v1"), "--dynamic-cache-root", str(asset_root / "cache/dynamic/v1"), "--dynamic-cache-identities", str(repository_root / "config/DYNAMIC_CACHE_IDENTITY.json")]
    registry = output_root / "smoke/runs/registry.jsonl"
    results: dict[str, Any] = {}
    for mode, executable in (("baseline", executable_baseline), ("semantic-feedback", executable_semantic)):
        output = output_root / "smoke/runs" / mode
        command = [str(python), str(repository_root / "tools/run_orb_tum.py"), "--mode", mode, "--study", "smoke", "--executable", str(executable), "--output-root", str(output), "--registry", str(registry), *common]
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
    python = resolve_python(asset_root)
    attach_asset_links(repository_root, asset_root)
    stages: list[dict[str, Any]] = []
    stages.append({"name": "preflight", "ok": True, "asset_root": str(asset_root), "repository_root": str(repository_root), "no_download_path": True})
    if smoke:
        build_dir = build_dir.resolve()
        stages.append(_command_stage("build", fresh_build_command(repository_root, build_dir), repository_root, output_root))
        if not stages[-1]["ok"]:
            raise ValueError(f"build failed; see {stages[-1]['log']}")
        stages.append(_command_stage("unit", [str(python), "-m", "pytest", "tests/python", "-q"], repository_root, output_root))
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
        smoke_facts = _smoke(repository_root, asset_root, output_root, python)
    stages.append({"name": "smoke", "ok": True, "facts": smoke_facts})
    metrics = _validate_metrics(asset_root)
    metrics["formal_runs"] = _validate_formal_runs(asset_root)
    stages.append({"name": "metrics", "ok": True, "facts": metrics})
    maps = _validate_maps(asset_root, repository_root, output_root)
    stages.append({"name": "map-validate", "ok": True, "facts": maps})
    expected = load_reproduction_plan().stages
    actual = [stage["name"] for stage in stages if stage["name"] in expected]
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
