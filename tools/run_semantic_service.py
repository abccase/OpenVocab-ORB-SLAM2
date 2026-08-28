#!/usr/bin/env python3
"""Run the latest-frame Grounding DINO + SAM ZeroMQ service for P06."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import zmq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_py.openvocab_slam.inference import infer_instances
from semantic_py.openvocab_slam.ipc import LatestFrameService, create_service_sockets
from tools.generate_cache import _load_frozen_inputs, _load_models, _set_deterministic_runtime


def _record_event(path: Path, state: str, **details: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "state": state,
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **details,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def make_online_infer(
    prompt,
    models,
    cfg,
    event_log: Path,
    *,
    infer_fn=infer_instances,
    oom_type=None,
    empty_cache=None,
    record=_record_event,
):
    """Bound CUDA fragmentation and retry one frame at the frozen 640 fallback."""
    if oom_type is None or empty_cache is None:
        import torch

        if oom_type is None:
            oom_type = torch.cuda.OutOfMemoryError
        if empty_cache is None:
            empty_cache = torch.cuda.empty_cache

    def infer(image):
        try:
            try:
                return tuple(infer_fn(image, prompt, models, cfg))
            except oom_type:
                from_long_side = int(models.detector.image_long_side)
                to_long_side = min(from_long_side, 640)
                empty_cache()
                models.detector.image_long_side = to_long_side
                record(
                    event_log, "RESOLUTION_FALLBACK",
                    from_long_side=from_long_side, to_long_side=to_long_side,
                    reason="CUDA_OOM",
                )
                return tuple(infer_fn(image, prompt, models, cfg))
        finally:
            empty_cache()

    return infer


def _release_detector_before_segmentation(models) -> None:
    import torch

    models.detector.model.to("cpu")
    torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-set", default="tum_office_v1")
    parser.add_argument("--request-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--result-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--backend", choices=("real", "protocol-test"), default="real",
        help="protocol-test exercises the production wire service with empty fast results",
    )
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "config/EXPERIMENT_MANIFEST.yaml",
    )
    return parser.parse_args(argv)


def run_service(args: argparse.Namespace) -> int:
    prompts = json.loads((PROJECT_ROOT / "config/PROMPTS.yaml").read_text(encoding="utf-8"))
    if prompts["prompt_set_id"] != args.prompt_set:
        raise ValueError("prompt-set identity mismatch")
    _, assets, prompt, prompt_sha256, model_sha256, cfg = _load_frozen_inputs(
        PROJECT_ROOT, args.manifest
    )
    if args.backend == "real":
        import torch

        _set_deterministic_runtime()
        models = _load_models(PROJECT_ROOT, assets, cfg, args.device)
        models.segmenter.use_autocast = args.device == "cuda"
        infer = make_online_infer(
            prompt, models, cfg, args.event_log,
            infer_fn=lambda image, active_prompt, active_models, active_cfg: infer_instances(
                image, active_prompt, active_models, active_cfg,
                before_segmentation=_release_detector_before_segmentation,
            ),
        )
        recoverable_oom = torch.cuda.OutOfMemoryError
    else:
        infer = lambda _image: ()
        recoverable_oom = ()
    service = LatestFrameService(
        run_id=args.run_id,
        prompt_sha256=prompt_sha256,
        model_manifest_sha256=model_sha256,
        infer=infer,
        event_log=args.event_log,
    )
    context = zmq.Context()
    subscriber, publisher = create_service_sockets(
        context,
        request_endpoint=args.request_endpoint,
        result_endpoint=args.result_endpoint,
    )
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    _record_event(args.event_log, "SERVICE_READY")
    try:
        while True:
            if subscriber in dict(poller.poll(100)):
                try:
                    service.serve_once(subscriber, publisher)
                except recoverable_oom as error:
                    _record_event(
                        args.event_log, "SERVICE_CONTINUING_AFTER_OOM",
                        error=str(error),
                    )
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        return 0
    finally:
        subscriber.close(0)
        publisher.close(0)
        context.term()


def run_with_failure_event(args: argparse.Namespace, *, run=run_service) -> int:
    """Persist fatal startup/runtime failures once argparse provides an event path."""
    try:
        return run(args)
    except Exception as error:
        _record_event(args.event_log, "SERVICE_FAILED", error=str(error))
        raise


def main(argv: list[str] | None = None) -> int:
    return run_with_failure_event(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"SEMANTIC_SERVICE_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
