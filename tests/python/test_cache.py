from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from semantic_py.openvocab_slam.cache import CacheWriter, read_cache_frame, validate_cache, write_cache_frame
from semantic_py.openvocab_slam.config import InferenceConfig
from semantic_py.openvocab_slam.inference import ModelBundle
from semantic_py.openvocab_slam.schemas import CacheManifest, SemanticFramePacket
from tools.generate_cache import (
    SequenceCacheJob,
    build_sequence_jobs,
    generate_sequence_cache,
    run_job_with_oom_fallback,
    _set_deterministic_runtime,
)


def packet(frame_id: int, timestamp: float, source_hash: str) -> SemanticFramePacket:
    return SemanticFramePacket(
        schema="ovorb.semantic-cache.v1",
        study_id="ovorb2_tum_v1",
        sequence_id="tiny",
        frame_id=frame_id,
        timestamp=timestamp,
        source_image_sha256=source_hash,
        image_width=4,
        image_height=3,
        prompt_sha256="2" * 64,
        model_manifest_sha256="3" * 64,
        inference_config_sha256="4" * 64,
        inference_time_seconds=0.1,
        instances=(),
    )


def manifest(frame_count: int = 2) -> CacheManifest:
    return CacheManifest(
        schema="ovorb.semantic-cache.v1",
        study_id="ovorb2_tum_v1",
        sequence_id="tiny",
        source_tree_sha256="5" * 64,
        association_sha256="6" * 64,
        prompt_sha256="2" * 64,
        model_manifest_sha256="3" * 64,
        inference_config_sha256="4" * 64,
        producer_commit="7" * 40,
        image_long_side=800,
        expected_frame_count=frame_count,
        resolution_fallback=None,
    )


class CacheTests(unittest.TestCase):
    def test_deterministic_runtime_initialization_is_callable(self) -> None:
        _set_deterministic_runtime()

    def test_oom_fallback_restarts_whole_sequence_once_at_640(self) -> None:
        class SimulatedOom(RuntimeError):
            pass

        class Detector:
            image_long_side = 800

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_root = root / "tiny"
            job = SequenceCacheJob(root, root / "associate.txt", cache_root, manifest())
            cfg = InferenceConfig("ovorb.semantic-cache.v1", 800, 0.35, 0.25, 0.5)
            models = ModelBundle(Detector(), object())
            calls = []

            def generate(observed_job, prompt, observed_models, observed_cfg, *, infer):
                calls.append((observed_job, observed_cfg))
                if len(calls) == 1:
                    cache_root.mkdir()
                    (cache_root / "partial").write_text("partial")
                    raise SimulatedOom("simulated")
                return "complete"

            result = run_job_with_oom_fallback(
                job,
                "person .",
                models,
                cfg,
                infer=lambda *args: [],
                fallback_long_side=640,
                generate=generate,
                oom_type=SimulatedOom,
            )

            self.assertEqual(result, "complete")
            self.assertEqual([call[1].image_long_side for call in calls], [800, 640])
            self.assertEqual(calls[1][0].manifest.image_long_side, 640)
            self.assertEqual(calls[1][0].manifest.resolution_fallback, "cuda_oom_800_to_640")
            self.assertEqual(models.detector.image_long_side, 640)
            failed = list(root.glob("tiny.failed-long-side-800*"))
            self.assertEqual(len(failed), 1)
            self.assertTrue((failed[0] / "partial").is_file())

    def test_generate_cache_cli_starts_outside_repository_cwd(self) -> None:
        script = Path(__file__).parents[2] / "tools/generate_cache.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd="/tmp",
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--validate-only", completed.stdout)

    def test_job_builder_binds_frozen_sequence_and_inference_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_root = root / "data/tum/manifests"
            raw_root = root / "data/tum/raw/rgbd_dataset_freiburg1_desk"
            manifest_root.mkdir(parents=True)
            raw_root.mkdir(parents=True)
            (raw_root / "associate.txt").write_text(
                "1.0 rgb/1.png 1.0 depth/1.png\n", encoding="utf-8"
            )
            association_sha = hashlib.sha256(
                (raw_root / "associate.txt").read_bytes()
            ).hexdigest()
            (manifest_root / "fr1_desk.json").write_text(
                json.dumps(
                    {
                        "archive": "rgbd_dataset_freiburg1_desk.tgz",
                        "association_sha256": association_sha,
                        "extracted_tree_sha256": "5" * 64,
                        "counts": {"associations": 1},
                    }
                ),
                encoding="utf-8",
            )
            experiment = {
                "schema_version": 1,
                "study_id": "ovorb2_tum_v1",
                "datasets": [{"id": "fr1_desk", "archive": "rgbd_dataset_freiburg1_desk.tgz"}],
                "cache": {
                    "schema": "ovorb.semantic-cache.v1",
                    "image_long_side": 800,
                    "box_threshold": 0.35,
                    "text_threshold": 0.25,
                    "mask_threshold": 0.5,
                },
            }
            cfg = InferenceConfig("ovorb.semantic-cache.v1", 800, 0.35, 0.25, 0.5)

            jobs = build_sequence_jobs(
                root,
                experiment,
                cfg,
                prompt_sha256="2" * 64,
                model_manifest_sha256="3" * 64,
                producer_commit="7" * 40,
            )

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].dataset_root, raw_root)
            self.assertEqual(jobs[0].manifest.expected_frame_count, 1)
            self.assertEqual(jobs[0].manifest.inference_config_sha256, cfg.sha256())
            self.assertEqual(jobs[0].cache_root, root / "cache/semantic/v1/fr1_desk")

    def test_packet_round_trip_preserves_packet_and_returns_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "7.msgpack.zst"
            original = packet(7, 7.25, "1" * 64)

            digest = write_cache_frame(path, original)

            self.assertEqual(read_cache_frame(path), original)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertFalse(any(path.parent.glob(".*.partial")))

    def test_atomic_write_keeps_existing_packet_if_replace_is_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "0.msgpack.zst"
            first = packet(0, 1.0, "1" * 64)
            write_cache_frame(path, first)

            with patch("os.replace", side_effect=OSError("simulated interruption")):
                with self.assertRaises(OSError):
                    write_cache_frame(path, packet(0, 1.0, "8" * 64))

            self.assertEqual(read_cache_frame(path), first)
            self.assertFalse(any(path.parent.glob(".*.partial")))

    def test_writer_recovers_orphan_packet_and_resume_skips_only_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = CacheWriter(root, manifest())
            first = packet(0, 1.0, "1" * 64)
            orphan_path = root / "frames/000000.msgpack.zst"
            write_cache_frame(orphan_path, first)

            first_digest = writer.add(first)
            second_digest = writer.add(packet(1, 2.0, "8" * 64))
            resumed = CacheWriter(root, manifest())

            self.assertTrue(resumed.has_valid_frame(0, 1.0, "1" * 64))
            self.assertFalse(resumed.has_valid_frame(99, 99.0, "a" * 64))
            self.assertEqual(resumed.add(first), first_digest)
            self.assertEqual(resumed.add(packet(1, 2.0, "8" * 64)), second_digest)
            self.assertEqual(len((root / "cache_index.jsonl").read_text().splitlines()), 2)
            with self.assertRaises(ValueError):
                resumed.has_valid_frame(0, 1.0, "9" * 64)
            with self.assertRaises(ValueError):
                resumed.add(packet(0, 1.0, "9" * 64))

    def test_validation_fails_closed_for_manifest_index_and_packet_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = manifest()
            writer = CacheWriter(root, expected)
            writer.add(packet(0, 1.0, "1" * 64))
            writer.add(packet(1, 2.0, "8" * 64))
            writer.finalize()
            self.assertTrue(validate_cache(root, expected).valid)

            changed_prompt = CacheManifest.from_primitive({**expected.to_primitive(), "prompt_sha256": "a" * 64})
            self.assertIn("manifest identity mismatch", validate_cache(root, changed_prompt).errors)

            index_path = root / "cache_index.jsonl"
            rows = index_path.read_text().splitlines()
            index_path.write_text("\n".join([rows[0], rows[0]]) + "\n")
            validation = validate_cache(root, expected)
            self.assertFalse(validation.valid)
            self.assertTrue(any("duplicate frame_id" in error for error in validation.errors))

    def test_validation_reports_corrupt_packet_even_if_index_hash_was_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = manifest(frame_count=1)
            writer = CacheWriter(root, expected)
            writer.add(packet(0, 1.0, "1" * 64))
            packet_path = root / "frames/000000.msgpack.zst"
            packet_path.write_bytes(b"not-a-zstd-frame")
            rows = [json.loads(line) for line in (root / "cache_index.jsonl").read_text().splitlines()]
            rows[0]["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
            (root / "cache_index.jsonl").write_text(json.dumps(rows[0]) + "\n")

            validation = validate_cache(root, expected)

            self.assertFalse(validation.valid)
            self.assertTrue(any("invalid index entry" in error for error in validation.errors))

    def test_finalize_rejects_missing_or_extra_packets_and_duplicate_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = CacheWriter(root, manifest())
            writer.add(packet(0, 1.0, "1" * 64))
            with self.assertRaises(ValueError):
                writer.add(packet(1, 1.0, "8" * 64))
            with self.assertRaises(ValueError):
                writer.finalize()

            extra = root / "frames/999999.msgpack.zst"
            write_cache_frame(extra, packet(999999, 9.0, "9" * 64))
            validation = validate_cache(root, manifest())
            self.assertTrue(any("extra packet" in error for error in validation.errors))

    def test_sequence_generation_resumes_without_repeating_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            (dataset / "rgb").mkdir(parents=True)
            image_path = dataset / "rgb/1.png"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((3, 4, 3), np.uint8)))
            association = dataset / "associate.txt"
            association.write_text("1.250 rgb/1.png 1.249 depth/1.png\n", encoding="utf-8")
            cache_manifest = manifest(frame_count=1)
            job = SequenceCacheJob(dataset, association, root / "cache", cache_manifest)
            cfg = InferenceConfig("ovorb.semantic-cache.v1", 800, 0.35, 0.25, 0.5)
            calls = []

            def fake_infer(image, prompt, models, observed_cfg):
                calls.append((image.shape, prompt, observed_cfg))
                return []

            first = generate_sequence_cache(job, "person .", None, cfg, infer=fake_infer)
            second = generate_sequence_cache(job, "person .", None, cfg, infer=fake_infer)

            self.assertTrue(first.valid)
            self.assertTrue(second.valid)
            self.assertEqual(calls, [((3, 4, 3), "person .", cfg)])
            self.assertEqual(len((job.cache_root / "cache_index.jsonl").read_text().splitlines()), 1)
            cached = read_cache_frame(job.cache_root / "frames/000000.msgpack.zst")
            self.assertEqual((cached.frame_id, cached.timestamp, cached.instances), (0, 1.25, ()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
