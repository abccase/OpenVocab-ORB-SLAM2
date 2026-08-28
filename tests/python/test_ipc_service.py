from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import jsonschema
import msgpack
import zmq

from semantic_py.openvocab_slam.ipc import (
    create_service_sockets,
    LatestFrameService,
    PacketExpectations,
    ProtocolError,
    unpack_frame_request,
    unpack_semantic_packet,
)
from semantic_py.openvocab_slam.schemas import InstanceObservation
from tools.run_semantic_service import make_online_infer, run_with_failure_event


FIXTURES = Path(__file__).parents[1] / "fixtures/ipc_packets"
RUN_ID = "p06-fixture-run"
PROMPT_SHA256 = "1" * 64
MODEL_SHA256 = "2" * 64


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def expectations() -> PacketExpectations:
    return PacketExpectations(
        run_id=RUN_ID,
        prompt_sha256=PROMPT_SHA256,
        model_manifest_sha256=MODEL_SHA256,
        image_width=3,
        image_height=2,
        current_timestamp_ns=1_200_000_000,
        max_age_ns=250_000_000,
    )


class IpcWireFixtureTests(unittest.TestCase):
    def test_shared_binary_fixtures_conform_to_frozen_json_schema(self) -> None:
        schema = json.loads(
            (Path(__file__).parents[2] / "config/PROTOCOL_SCHEMA.json").read_text(
                encoding="utf-8"
            )
        )
        def validate_msgpack_type(validator, required, instance, _schema):
            if required == "bin" and not isinstance(instance, bytes):
                yield jsonschema.ValidationError("value is not MessagePack bin")

        validator_type = jsonschema.validators.extend(
            jsonschema.Draft202012Validator,
            {"x-msgpack-type": validate_msgpack_type},
        )
        validator = validator_type(schema)
        validator.check_schema(schema)

        for name in ("valid_frame_request.msgpack", "valid_semantic_packet.msgpack"):
            with self.subTest(name=name):
                validator.validate(msgpack.unpackb(fixture(name), raw=False))

        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(
                msgpack.unpackb(fixture("wrong_field_type.msgpack"), raw=False)
            )
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(
                msgpack.unpackb(fixture("invalid_jpeg_type.msgpack"), raw=False)
            )

    def test_committed_fixtures_are_byte_identical_to_generator_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory)
            subprocess.run(
                [
                    sys.executable,
                    str(FIXTURES / "generate_fixtures.py"),
                    "--output-root",
                    str(generated),
                ],
                check=True,
            )
            committed_names = sorted(path.name for path in FIXTURES.glob("*.msgpack"))
            generated_names = sorted(path.name for path in generated.glob("*.msgpack"))
            self.assertEqual(generated_names, committed_names)
            for name in committed_names:
                with self.subTest(name=name):
                    self.assertEqual((generated / name).read_bytes(), fixture(name))

    def test_valid_frame_request_decodes_binary_jpeg_and_identity(self) -> None:
        request = unpack_frame_request(
            fixture("valid_frame_request.msgpack"),
            expected_run_id=RUN_ID,
            expected_prompt_sha256=PROMPT_SHA256,
        )

        self.assertEqual(request.frame_id, 7)
        self.assertEqual((request.image_width, request.image_height), (3, 2))
        self.assertTrue(request.jpeg_bytes.startswith(b"\xff\xd8"))

    def test_valid_semantic_packet_preserves_instance_and_rle(self) -> None:
        packet = unpack_semantic_packet(fixture("valid_semantic_packet.msgpack"), expectations())

        self.assertEqual(packet.frame_id, 7)
        self.assertEqual(packet.age_ns, 200_000_000)
        self.assertEqual(packet.instances[0].label, "person")
        self.assertEqual(packet.instances[0].mask_rle["counts"], [1, 2, 3])

    def test_invalid_shared_fixtures_fail_with_stable_reason(self) -> None:
        cases = {
            "wrong_version.msgpack": "WRONG_PROTOCOL_VERSION",
            "wrong_run_id.msgpack": "WRONG_RUN_ID",
            "stale_timestamp.msgpack": "STALE_PACKET",
            "malformed_rle.msgpack": "MALFORMED_RLE",
            "wrong_dimensions.msgpack": "WRONG_DIMENSIONS",
            "wrong_field_type.msgpack": "INVALID_PACKET",
            "oversized_dimensions.msgpack": "INVALID_PACKET",
            "oversized_local_id.msgpack": "INVALID_INSTANCE",
            "oversized_rle_size.msgpack": "MALFORMED_RLE",
        }
        for name, reason in cases.items():
            with self.subTest(name=name), self.assertRaises(ProtocolError) as raised:
                unpack_semantic_packet(fixture(name), expectations())
            self.assertEqual(raised.exception.code, reason)

    def test_truncated_messagepack_is_rejected_as_corrupt(self) -> None:
        payload = fixture("valid_semantic_packet.msgpack")[:-3]

        with self.assertRaises(ProtocolError) as raised:
            unpack_semantic_packet(payload, expectations())

        self.assertEqual(raised.exception.code, "CORRUPT_MESSAGEPACK")

    def test_python_rejects_coerced_boolean_fractional_and_string_fields(self) -> None:
        packet = msgpack.unpackb(fixture("valid_semantic_packet.msgpack"), raw=False)
        for field, invalid in (
            ("frame_id", True),
            ("frame_id", 7.5),
            ("source_timestamp_ns", "1000000000"),
            ("image_width", 3.0),
            ("inference_ms", True),
        ):
            with self.subTest(field=field, invalid=invalid):
                mutated = {**packet, field: invalid}
                with self.assertRaises(ProtocolError) as raised:
                    unpack_semantic_packet(
                        msgpack.packb(mutated, use_bin_type=True), expectations()
                    )
                self.assertEqual(raised.exception.code, "INVALID_PACKET")

        request = msgpack.unpackb(fixture("valid_frame_request.msgpack"), raw=False)
        for field, invalid in (("frame_id", False), ("image_height", 2.5)):
            with self.subTest(request_field=field, invalid=invalid):
                mutated = {**request, field: invalid}
                with self.assertRaises(ProtocolError) as raised:
                    unpack_frame_request(
                        msgpack.packb(mutated, use_bin_type=True),
                        expected_run_id=RUN_ID,
                        expected_prompt_sha256=PROMPT_SHA256,
                    )
                self.assertEqual(raised.exception.code, "INVALID_PACKET")


class LatestFrameServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log_path = Path(self._testMethodName + ".jsonl")
        self.addCleanup(self.log_path.unlink, missing_ok=True)

    @staticmethod
    def instance() -> InstanceObservation:
        return InstanceObservation(
            local_id=0,
            label="person",
            score=0.9,
            box_xyxy=(0.0, 0.0, 2.0, 1.0),
            mask_rle={"size": [2, 3], "counts": [1, 2, 3]},
        )

    def service(self, infer):
        return LatestFrameService(
            run_id=RUN_ID,
            prompt_sha256=PROMPT_SHA256,
            model_manifest_sha256=MODEL_SHA256,
            infer=infer,
            event_log=self.log_path,
        )

    def test_valid_request_runs_inference_and_persists_complete_timing(self) -> None:
        seen_shapes: list[tuple[int, ...]] = []

        def infer(image: np.ndarray) -> tuple[InstanceObservation, ...]:
            seen_shapes.append(image.shape)
            return (self.instance(),)

        result = self.service(infer).handle_payload(
            fixture("valid_frame_request.msgpack"),
            produced_timestamp_ns=1_100_000_000,
        )
        packet = unpack_semantic_packet(result, expectations())
        events = [json.loads(line) for line in self.log_path.read_text().splitlines()]

        self.assertEqual(seen_shapes, [(2, 3, 3)])
        self.assertEqual(packet.instances[0].label, "person")
        self.assertEqual([event["state"] for event in events], ["INFERENCE_COMPLETED"])
        self.assertEqual(events[0]["frame_id"], 7)
        self.assertGreaterEqual(events[0]["inference_ms"], 0.0)
        self.assertLessEqual(
            events[0]["receive_monotonic_ns"], events[0]["inference_end_monotonic_ns"]
        )

    def test_invalid_identity_is_rejected_before_inference(self) -> None:
        calls = 0

        def infer(_image: np.ndarray) -> tuple[InstanceObservation, ...]:
            nonlocal calls
            calls += 1
            return ()

        value = msgpack.unpackb(fixture("valid_frame_request.msgpack"), raw=False)
        value["run_id"] = "wrong-run"

        with self.assertRaises(ProtocolError) as raised:
            self.service(infer).handle_payload(msgpack.packb(value, use_bin_type=True))

        self.assertEqual(raised.exception.code, "WRONG_RUN_ID")
        self.assertEqual(calls, 0)
        self.assertEqual(json.loads(self.log_path.read_text())["state"], "REJECTED")

    def test_inference_failure_is_persisted_and_propagated(self) -> None:
        def infer(_image: np.ndarray) -> tuple[InstanceObservation, ...]:
            raise RuntimeError("model crashed")

        with self.assertRaisesRegex(RuntimeError, "model crashed"):
            self.service(infer).handle_payload(fixture("valid_frame_request.msgpack"))

        event = json.loads(self.log_path.read_text())
        self.assertEqual(event["state"], "INFERENCE_FAILED")
        self.assertEqual(event["error"], "model crashed")

    def test_serve_once_uses_real_nonblocking_sockets(self) -> None:
        context = zmq.Context()
        requests_out = context.socket(zmq.PUSH)
        requests_in = context.socket(zmq.PULL)
        results_out = context.socket(zmq.PUSH)
        results_in = context.socket(zmq.PULL)
        endpoint_request = f"inproc://request-{self._testMethodName}"
        endpoint_result = f"inproc://result-{self._testMethodName}"
        requests_in.bind(endpoint_request)
        requests_out.connect(endpoint_request)
        results_out.bind(endpoint_result)
        results_in.connect(endpoint_result)
        self.addCleanup(context.term)
        for socket in (requests_out, requests_in, results_out, results_in):
            self.addCleanup(socket.close, 0)
        service = self.service(lambda _image: (self.instance(),))

        started = time.monotonic()
        self.assertEqual(service.serve_once(requests_in, results_out), "NO_REQUEST")
        self.assertLess(time.monotonic() - started, 0.005)
        requests_out.send(fixture("valid_frame_request.msgpack"))
        self.assertEqual(service.serve_once(requests_in, results_out), "PUBLISHED")
        packet = unpack_semantic_packet(results_in.recv(), expectations())

        self.assertEqual(packet.frame_id, 7)
        events = [json.loads(line) for line in self.log_path.read_text().splitlines()]
        self.assertEqual(
            [event["state"] for event in events],
            ["INFERENCE_COMPLETED", "PUBLISHED"],
        )

    def test_result_eagain_records_drop_identity_without_false_publish(self) -> None:
        class RequestSocket:
            def recv(self, *, flags):
                self.flags = flags
                return fixture("valid_frame_request.msgpack")

        class DroppingSocket:
            def send(self, _payload, *, flags):
                self.flags = flags
                raise zmq.Again()

        service = self.service(lambda _image: (self.instance(),))

        self.assertEqual(
            service.serve_once(RequestSocket(), DroppingSocket()), "RESULT_DROPPED"
        )
        events = [json.loads(line) for line in self.log_path.read_text().splitlines()]

        self.assertEqual(
            [event["state"] for event in events],
            ["INFERENCE_COMPLETED", "RESULT_DROPPED"],
        )
        self.assertEqual(events[-1]["frame_id"], 7)
        self.assertEqual(events[-1]["source_timestamp_ns"], 1_000_000_000)

    def test_service_sockets_are_latest_only_and_bounded(self) -> None:
        context = zmq.Context()
        self.addCleanup(context.term)
        subscriber, publisher = create_service_sockets(
            context,
            request_endpoint=f"inproc://requests-{self._testMethodName}",
            result_endpoint=f"inproc://results-{self._testMethodName}",
        )
        self.addCleanup(subscriber.close, 0)
        self.addCleanup(publisher.close, 0)

        self.assertEqual(subscriber.getsockopt(zmq.RCVHWM), 1)
        self.assertEqual(subscriber.getsockopt(zmq.CONFLATE), 1)
        self.assertEqual(publisher.getsockopt(zmq.SNDHWM), 1)


class OnlineInferenceFallbackTests(unittest.TestCase):
    def test_oom_clears_allocator_and_retries_at_frozen_fallback_resolution(self) -> None:
        class OomError(RuntimeError):
            pass

        class Detector:
            image_long_side = 800

        class Models:
            detector = Detector()

        class Config:
            image_long_side = 800

        calls: list[int] = []
        clears: list[bool] = []
        events: list[tuple[str, dict[str, object]]] = []

        def infer(_image, _prompt, models, _cfg):
            calls.append(models.detector.image_long_side)
            if len(calls) == 1:
                raise OomError("out of memory")
            return ()

        wrapped = make_online_infer(
            "person .", Models(), Config(), Path("unused.jsonl"),
            infer_fn=infer, oom_type=OomError,
            empty_cache=lambda: clears.append(True),
            record=lambda _path, state, **details: events.append((state, details)),
        )

        self.assertEqual(wrapped(np.zeros((2, 3, 3), dtype=np.uint8)), ())
        self.assertEqual(calls, [800, 640])
        self.assertEqual(clears, [True, True])
        self.assertEqual(events[0][0], "RESOLUTION_FALLBACK")
        self.assertEqual(events[0][1]["from_long_side"], 800)
        self.assertEqual(events[0][1]["to_long_side"], 640)


class ServiceProcessFailureTests(unittest.TestCase):
    def test_fatal_failure_is_persisted_when_event_path_is_known(self) -> None:
        event_log = Path(self._testMethodName + ".jsonl")
        self.addCleanup(event_log.unlink, missing_ok=True)
        args = SimpleNamespace(event_log=event_log)

        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            run_with_failure_event(
                args, run=lambda _args: (_ for _ in ()).throw(
                    RuntimeError("startup failed")
                ),
            )

        event = json.loads(event_log.read_text(encoding="utf-8"))
        self.assertEqual(event["state"], "SERVICE_FAILED")
        self.assertEqual(event["error"], "startup failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
