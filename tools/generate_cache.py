#!/usr/bin/env python3
"""Generate and validate immutable semantic frame caches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_py.openvocab_slam.cache import CacheWriter, validate_cache
from semantic_py.openvocab_slam.config import InferenceConfig, normalize_formal_prompt
from semantic_py.openvocab_slam.inference import infer_instances, load_model_bundle
from semantic_py.openvocab_slam.schemas import CacheManifest, SemanticFramePacket


@dataclass(frozen=True)
class SequenceCacheJob:
    dataset_root: Path
    association_path: Path
    cache_root: Path
    manifest: CacheManifest


def build_sequence_jobs(
    project_root,
    experiment,
    cfg,
    *,
    prompt_sha256,
    model_manifest_sha256,
    producer_commit,
    enforce_producer_commit=True,
):
    root = Path(project_root)
    jobs: list[SequenceCacheJob] = []
    for dataset in experiment["datasets"]:
        sequence_id = str(dataset["id"])
        sequence_manifest_path = root / "data/tum/manifests" / f"{sequence_id}.json"
        sequence_manifest = json.loads(sequence_manifest_path.read_text(encoding="utf-8"))
        if sequence_manifest["archive"] != dataset["archive"]:
            raise ValueError(f"archive identity mismatch for {sequence_id}")
        dataset_root = root / "data/tum/raw" / Path(str(dataset["archive"])).stem
        association_path = dataset_root / "associate.txt"
        if _sha256_file(association_path) != sequence_manifest["association_sha256"]:
            raise ValueError(f"association hash mismatch for {sequence_id}")
        if _hash_extracted_tree(dataset_root) != sequence_manifest["extracted_tree_sha256"]:
            raise ValueError(f"source tree hash mismatch for {sequence_id}")
        cache_root = root / "cache/semantic/v1" / sequence_id
        existing_manifest = cache_root / "cache_manifest.json"
        selected_cfg = cfg
        selected_producer = producer_commit
        resolution_fallback = None
        if existing_manifest.is_file():
            observed = CacheManifest.from_primitive(json.loads(existing_manifest.read_text()))
            if enforce_producer_commit and observed.producer_commit != producer_commit:
                raise ValueError(f"producer commit mismatch for {sequence_id}")
            selected_producer = observed.producer_commit
            fallback_long_side = int(experiment["cache"]["oom_fallback_long_side"])
            expected_fallback = f"cuda_oom_{cfg.image_long_side}_to_{fallback_long_side}"
            if observed.resolution_fallback is None and observed.image_long_side == cfg.image_long_side:
                pass
            elif (
                observed.resolution_fallback == expected_fallback
                and observed.image_long_side == fallback_long_side
            ):
                selected_cfg = replace(cfg, image_long_side=fallback_long_side)
                resolution_fallback = expected_fallback
            else:
                raise ValueError(f"invalid resolution fallback identity for {sequence_id}")
        jobs.append(
            SequenceCacheJob(
                dataset_root=dataset_root,
                association_path=association_path,
                cache_root=cache_root,
                manifest=CacheManifest(
                    schema=selected_cfg.schema,
                    study_id=str(experiment["study_id"]),
                    sequence_id=sequence_id,
                    source_tree_sha256=str(sequence_manifest["extracted_tree_sha256"]),
                    association_sha256=str(sequence_manifest["association_sha256"]),
                    prompt_sha256=prompt_sha256,
                    model_manifest_sha256=model_manifest_sha256,
                    inference_config_sha256=selected_cfg.sha256(),
                    producer_commit=selected_producer,
                    image_long_side=selected_cfg.image_long_side,
                    expected_frame_count=int(sequence_manifest["counts"]["associations"]),
                    resolution_fallback=resolution_fallback,
                ),
            )
        )
    return jobs


def generate_sequence_cache(job, prompt, models, cfg, *, infer):
    rows = _read_association(job.association_path)
    if len(rows) != job.manifest.expected_frame_count:
        raise ValueError("association count does not match cache manifest")
    writer = CacheWriter(job.cache_root, job.manifest)
    for frame_id, (timestamp, relative_image) in enumerate(rows):
        image_path = job.dataset_root / relative_image
        source_hash = _sha256_file(image_path)
        if writer.has_valid_frame(frame_id, timestamp, source_hash):
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to read RGB image: {image_path}")
        started = time.perf_counter()
        instances = tuple(infer(image, prompt, models, cfg))
        elapsed = time.perf_counter() - started
        writer.add(
            SemanticFramePacket(
                schema=job.manifest.schema,
                study_id=job.manifest.study_id,
                sequence_id=job.manifest.sequence_id,
                frame_id=frame_id,
                timestamp=timestamp,
                source_image_sha256=source_hash,
                image_width=int(image.shape[1]),
                image_height=int(image.shape[0]),
                prompt_sha256=job.manifest.prompt_sha256,
                model_manifest_sha256=job.manifest.model_manifest_sha256,
                inference_config_sha256=job.manifest.inference_config_sha256,
                inference_time_seconds=elapsed,
                instances=instances,
            )
        )
        if (frame_id + 1) % 10 == 0 or frame_id + 1 == len(rows):
            print(f"CACHE_PROGRESS {job.manifest.sequence_id} {frame_id + 1}/{len(rows)}", flush=True)
    return writer.finalize()


def run_job_with_oom_fallback(
    job,
    prompt,
    models,
    cfg,
    *,
    infer,
    fallback_long_side,
    generate=generate_sequence_cache,
    oom_type=None,
):
    if oom_type is None:
        import torch

        oom_type = torch.cuda.OutOfMemoryError
    models.detector.image_long_side = cfg.image_long_side
    if job.manifest.resolution_fallback is not None or cfg.image_long_side == fallback_long_side:
        return generate(job, prompt, models, cfg, infer=infer)
    try:
        return generate(job, prompt, models, cfg, infer=infer)
    except oom_type:
        if job.cache_root.exists():
            failed_root = job.cache_root.with_name(
                f"{job.cache_root.name}.failed-long-side-{cfg.image_long_side}"
            )
            if failed_root.exists():
                failed_root = failed_root.with_name(f"{failed_root.name}-{time.time_ns()}")
            job.cache_root.rename(failed_root)
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
        fallback_cfg = replace(cfg, image_long_side=int(fallback_long_side))
        fallback_job = replace(
            job,
            manifest=replace(
                job.manifest,
                inference_config_sha256=fallback_cfg.sha256(),
                image_long_side=fallback_cfg.image_long_side,
                resolution_fallback=(
                    f"cuda_oom_{cfg.image_long_side}_to_{fallback_cfg.image_long_side}"
                ),
            ),
        )
        models.detector.image_long_side = fallback_cfg.image_long_side
        return generate(fallback_job, prompt, models, fallback_cfg, infer=infer)


def _read_association(path: Path) -> list[tuple[float, Path]]:
    rows: list[tuple[float, Path]] = []
    seen: set[float] = set()
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: association row must have four fields")
        timestamp = float(fields[0])
        if timestamp in seen:
            raise ValueError(f"{path}:{line_number}: duplicate RGB timestamp")
        seen.add(timestamp)
        relative = Path(fields[1])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("rgb",):
            raise ValueError(f"{path}:{line_number}: unsafe RGB path")
        rows.append((timestamp, relative))
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_extracted_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.is_symlink() or path.name == "associate.txt":
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_output(*args, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _require_clean_product_tree(root: Path) -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
    )
    if status:
        raise ValueError("product tree must be clean before cache generation")


def _validate_source_repository(
    source: Path,
    expected_commit: str,
    *,
    expected_diff: str,
    allowed_untracked: dict[str, str] | None = None,
    allowed_ignored_sha256: dict[str, str] | None = None,
) -> None:
    source = Path(source)
    if _git_output("rev-parse", "HEAD", cwd=source) != expected_commit:
        raise ValueError(f"source commit mismatch: {source.name}")
    observed_diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--binary", "--", "."], cwd=source, text=True
    )
    if observed_diff != expected_diff:
        raise ValueError(f"tracked source diff mismatch: {source.name}")
    allowed = allowed_untracked or {}
    allowed_ignored = allowed_ignored_sha256 or {}
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=source, text=True
    ).splitlines()
    for relative in untracked:
        if relative not in allowed:
            raise ValueError(f"untracked source file: {source.name}/{relative}")
        if (source / relative).read_text(encoding="utf-8") != allowed[relative]:
            raise ValueError(f"untracked source content mismatch: {source.name}/{relative}")
    ignored = subprocess.check_output(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
        cwd=source,
        text=True,
    ).splitlines()
    for relative in ignored:
        if relative in allowed_ignored:
            if _sha256_file(source / relative) != allowed_ignored[relative]:
                raise ValueError(f"ignored source hash mismatch: {source.name}/{relative}")
            continue
        path = Path(relative)
        is_bytecode = "__pycache__" in path.parts and path.suffix == ".pyc"
        is_build_output = path.parts[:1] == ("build",)
        if path.suffix in {".py", ".so"} and not (is_bytecode or is_build_output):
            raise ValueError(f"ignored importable source file: {source.name}/{relative}")


def _validate_cuda_extension(source: Path, identity: dict) -> None:
    extension = source / identity["path"]
    if extension.stat().st_size != int(identity["size"]):
        raise ValueError("GroundingDINO CUDA extension size mismatch")
    if _sha256_file(extension) != identity["sha256"]:
        raise ValueError("GroundingDINO CUDA extension hash mismatch")
    spec = importlib.util.find_spec("groundingdino._C")
    if spec is None or spec.origin is None:
        raise ValueError("GroundingDINO CUDA extension is not importable")
    if Path(spec.origin).resolve() != extension.resolve():
        raise ValueError("GroundingDINO CUDA extension import path mismatch")


def _validate_model_assets(root: Path, assets: dict) -> None:
    paths = {
        "grounding_dino": root / "external/GroundingDINO",
        "sam": root / "external/segment-anything",
    }
    patch_path = root / assets["grounding_dino"]["compatibility_patch"]
    extension_identity = assets["grounding_dino"]["cuda_extension"]
    if _sha256_file(patch_path) != assets["grounding_dino"]["compatibility_patch_sha256"]:
        raise ValueError("GroundingDINO compatibility patch hash mismatch")
    _validate_source_repository(
        paths["grounding_dino"],
        assets["grounding_dino"]["commit"],
        expected_diff=patch_path.read_text(encoding="utf-8"),
        allowed_untracked={"groundingdino/version.py": "__version__ = '0.1.0'\n"},
        allowed_ignored_sha256={
            extension_identity["path"]: extension_identity["sha256"]
        },
    )
    _validate_cuda_extension(paths["grounding_dino"], extension_identity)
    _validate_source_repository(
        paths["sam"], assets["sam"]["commit"], expected_diff=""
    )
    dino_config = paths["grounding_dino"] / assets["grounding_dino"]["config"]
    if _sha256_file(dino_config) != assets["grounding_dino"]["config_sha256"]:
        raise ValueError("GroundingDINO config hash mismatch")
    for name in ("grounding_dino", "sam"):
        entry = assets[name]
        path = root / "weights" / entry["weights"]
        if path.stat().st_size != int(entry["weights_size"]):
            raise ValueError(f"{name} weight size mismatch")
        if _sha256_file(path) != entry["weights_sha256"]:
            raise ValueError(f"{name} weight hash mismatch")
    bert_root = root / "weights" / assets["bert"]["local_directory"]
    for relative, identity in assets["bert"]["files"].items():
        path = bert_root / relative
        if path.stat().st_size != int(identity["size"]):
            raise ValueError(f"BERT asset size mismatch: {relative}")
        if _sha256_file(path) != identity["sha256"]:
            raise ValueError(f"BERT asset hash mismatch: {relative}")


def _set_deterministic_runtime() -> None:
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_frozen_inputs(root: Path, manifest_path: Path):
    experiment = _load_json(manifest_path)
    prompts = _load_json(root / experiment["cache"]["prompts_file"])
    prompt = normalize_formal_prompt(prompts["frozen_formal_prompt"])
    assets_path = root / "config/SEMANTIC_MODELS.json"
    assets = _load_json(assets_path)
    _validate_model_assets(root, assets)
    cfg = InferenceConfig(
        schema=str(experiment["cache"]["schema"]),
        image_long_side=int(experiment["cache"]["image_long_side"]),
        box_threshold=float(experiment["cache"]["box_threshold"]),
        text_threshold=float(experiment["cache"]["text_threshold"]),
        mask_threshold=float(experiment["cache"]["mask_threshold"]),
    )
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return experiment, assets, prompt, prompt_sha, _sha256_file(assets_path), cfg


def _load_models(root: Path, assets: dict, cfg: InferenceConfig, device: str):
    return load_model_bundle(
        root / "external/GroundingDINO" / assets["grounding_dino"]["config"],
        root / "weights" / assets["grounding_dino"]["weights"],
        root / "weights" / assets["sam"]["weights"],
        bert_directory=root / "weights" / assets["bert"]["local_directory"],
        device=device,
        image_long_side=cfg.image_long_side,
    )


def _smoke(job, prompt, models, cfg, output_path=None) -> None:
    import torch

    timestamp, relative_image = _read_association(job.association_path)[0]
    image_path = job.dataset_root / relative_image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read smoke image: {image_path}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    first = infer_instances(image, prompt, models, cfg)
    second = infer_instances(image, prompt, models, cfg)
    if len(first) != len(second):
        raise ValueError("real-model smoke is nondeterministic: instance count")
    for left, right in zip(first, second):
        if left.label != right.label or left.mask_rle != right.mask_rle:
            raise ValueError("real-model smoke is nondeterministic: label or mask")
        if max(abs(a - b) for a, b in zip(left.box_xyxy, right.box_xyxy)) > 1e-6:
            raise ValueError("real-model smoke is nondeterministic: boxes")
        if abs(left.score - right.score) > 1e-6:
            raise ValueError("real-model smoke is nondeterministic: scores")
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema": cfg.schema,
                    "sequence_id": job.manifest.sequence_id,
                    "timestamp": timestamp,
                    "source_image_sha256": _sha256_file(image_path),
                    "prompt_sha256": job.manifest.prompt_sha256,
                    "model_manifest_sha256": job.manifest.model_manifest_sha256,
                    "inference_config_sha256": cfg.sha256(),
                    "instances": [item.to_primitive() for item in first],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    print(
        f"REAL_SMOKE PASS sequence={job.manifest.sequence_id} timestamp={timestamp} "
        f"instances={len(first)} image_sha256={_sha256_file(image_path)} "
        f"peak_allocated_bytes={peak_allocated} peak_reserved_bytes={peak_reserved}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("config/EXPERIMENT_MANIFEST.yaml"))
    parser.add_argument("--resume", action="store_true", help="validate and skip exact existing packets")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if not args.validate_only:
        _require_clean_product_tree(root)
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    experiment, assets, prompt, prompt_sha, model_sha, cfg = _load_frozen_inputs(root, manifest_path)
    producer = _git_output("rev-parse", "HEAD", cwd=root)
    jobs = build_sequence_jobs(
        root,
        experiment,
        cfg,
        prompt_sha256=prompt_sha,
        model_manifest_sha256=model_sha,
        producer_commit=producer,
        enforce_producer_commit=not args.validate_only,
    )
    if args.validate_only:
        failed = False
        for job in jobs:
            result = validate_cache(job.cache_root, job.manifest)
            print(
                f"CACHE_VALIDATE {job.manifest.sequence_id} "
                f"{'PASS' if result.valid else 'FAIL'} frames={result.frame_count}"
            )
            for error in result.errors:
                print(f"  {error}")
            failed = failed or not result.valid
        return int(failed)

    _set_deterministic_runtime()
    models = _load_models(root, assets, cfg, args.device)
    if args.smoke:
        _smoke(jobs[0], prompt, models, cfg, args.smoke_output)
        return 0
    if not args.resume and any(job.cache_root.exists() for job in jobs):
        raise ValueError("semantic cache already exists; pass --resume to validate and continue")
    for job in jobs:
        job_cfg = replace(cfg, image_long_side=job.manifest.image_long_side)
        run_job_with_oom_fallback(
            job,
            prompt,
            models,
            job_cfg,
            infer=infer_instances,
            fallback_long_side=int(experiment["cache"]["oom_fallback_long_side"]),
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CACHE_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
