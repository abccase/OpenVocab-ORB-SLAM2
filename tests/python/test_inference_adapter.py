from __future__ import annotations

import unittest

import numpy as np

from semantic_py.openvocab_slam.config import InferenceConfig, normalize_formal_prompt
from semantic_py.openvocab_slam.inference import (
    GroundingDinoDetector,
    ModelBundle,
    SamSegmenter,
    infer_instances,
)
from semantic_py.openvocab_slam.schemas import decode_binary_mask_rle


FORMAL_PROMPT = "person . chair . office chair ."


class FakeDetector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, float]] = []

    def predict(self, image_bgr, prompt, box_threshold, text_threshold):
        self.calls.append((prompt, box_threshold, text_threshold))
        return [
            {"box_cxcywh_norm": (0.25, 0.50, 0.50, 1.00), "score": 0.90, "label": " Person. "},
            {"box_cxcywh_norm": (1.20, 0.50, 0.10, 0.20), "score": 0.99, "label": "outside"},
            {"box_cxcywh_norm": (0.75, 0.50, 0.50, 0.50), "score": 0.80, "label": "CHAIR (0.80)"},
            {"box_cxcywh_norm": (0.50, 0.50, 0.00, 0.25), "score": 0.95, "label": "empty"},
        ]


class FakeSegmenter:
    def __init__(self) -> None:
        self.boxes: list[tuple[float, float, float, float]] = []

    def predict_masks(self, image_bgr, boxes_xyxy):
        self.boxes = list(boxes_xyxy)
        masks = np.zeros((len(boxes_xyxy), 4, 8), dtype=np.float32)
        masks[0, :, :4] = 0.75
        masks[1, 1:3, 4:8] = 0.75
        return masks


class InferenceAdapterTests(unittest.TestCase):
    def test_grounding_dino_adapter_preserves_normalized_boxes_and_scores(self) -> None:
        calls = []

        def preprocess(image_bgr, image_long_side):
            calls.append((image_bgr.copy(), image_long_side))
            return "tensor"

        def predict(**kwargs):
            self.assertEqual(kwargs["image"], "tensor")
            self.assertEqual(kwargs["caption"], FORMAL_PROMPT)
            self.assertEqual(kwargs["box_threshold"], 0.35)
            self.assertEqual(kwargs["text_threshold"], 0.25)
            self.assertEqual(kwargs["device"], "cuda")
            self.assertTrue(kwargs["remove_combined"])
            return (
                np.asarray([[0.25, 0.50, 0.50, 1.00]], dtype=np.float32),
                np.asarray([0.875], dtype=np.float32),
                ["person"],
            )

        image = np.zeros((4, 8, 3), dtype=np.uint8)
        detector = GroundingDinoDetector(
            model=object(),
            device="cuda",
            image_long_side=640,
            preprocess=preprocess,
            predict_fn=predict,
        )

        observed = detector.predict(image, FORMAL_PROMPT, 0.35, 0.25)

        self.assertEqual(calls[0][1], 640)
        np.testing.assert_array_equal(calls[0][0], image)
        self.assertEqual(observed[0]["label"], "person")
        self.assertAlmostEqual(observed[0]["score"], 0.875)
        np.testing.assert_allclose(observed[0]["box_cxcywh_norm"], [0.25, 0.5, 0.5, 1.0])

    def test_sam_adapter_batches_boxes_and_returns_probabilities(self) -> None:
        class FakeTransform:
            def apply_boxes_torch(self, boxes, image_shape):
                self.args = (boxes.clone(), image_shape)
                return boxes + 1.0

        class FakePredictor:
            def __init__(self):
                self.transform = FakeTransform()
                self.device = "cpu"
                self.image = None

            def set_image(self, image, image_format):
                self.image = (image.copy(), image_format)

            def predict_torch(self, **kwargs):
                self.kwargs = kwargs
                logits = np.asarray(
                    [[[[0.0, 2.0], [-2.0, 0.0]]], [[[1.0, -1.0], [0.0, 3.0]]]],
                    dtype=np.float32,
                )
                import torch

                return torch.from_numpy(logits), None, None

        image = np.zeros((2, 2, 3), dtype=np.uint8)
        predictor = FakePredictor()
        observed = SamSegmenter(predictor).predict_masks(
            image,
            [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 2.0, 2.0)],
        )

        self.assertEqual(predictor.image[1], "BGR")
        self.assertEqual(predictor.transform.args[1], (2, 2))
        self.assertEqual(tuple(predictor.kwargs["boxes"].shape), (2, 4))
        self.assertFalse(predictor.kwargs["multimask_output"])
        self.assertTrue(predictor.kwargs["return_logits"])
        self.assertEqual(observed.shape, (2, 2, 2))
        np.testing.assert_allclose(observed[0, 0, 0], 0.5, atol=1e-6)
        self.assertGreater(observed[0, 0, 1], 0.5)

    def test_prompt_normalization_is_stable_and_deduplicates_terms(self) -> None:
        self.assertEqual(
            normalize_formal_prompt(" Person. chair . person . Office Chair "),
            "person . chair . office chair .",
        )

    def test_inference_filters_boxes_then_segments_survivors_in_one_batch(self) -> None:
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        detector = FakeDetector()
        segmenter = FakeSegmenter()
        cfg = InferenceConfig(
            schema="ovorb.semantic-cache.v1",
            image_long_side=800,
            box_threshold=0.35,
            text_threshold=0.25,
            mask_threshold=0.5,
        )

        output = infer_instances(image, FORMAL_PROMPT, ModelBundle(detector, segmenter), cfg)

        self.assertEqual(detector.calls, [(FORMAL_PROMPT, 0.35, 0.25)])
        self.assertEqual([item.label for item in output], ["person", "chair"])
        self.assertEqual([item.local_id for item in output], [0, 1])
        self.assertEqual(segmenter.boxes, [(0.0, 0.0, 4.0, 4.0), (4.0, 1.0, 8.0, 3.0)])
        self.assertEqual([item.box_xyxy for item in output], segmenter.boxes)
        self.assertEqual([item.score for item in output], [0.9, 0.8])
        self.assertEqual(int(decode_binary_mask_rle(output[0].mask_rle).sum()), 16)
        self.assertEqual(int(decode_binary_mask_rle(output[1].mask_rle).sum()), 8)

    def test_inference_rejects_wrong_image_and_segmenter_shape(self) -> None:
        cfg = InferenceConfig("ovorb.semantic-cache.v1", 800, 0.35, 0.25, 0.5)
        detector = FakeDetector()
        segmenter = FakeSegmenter()
        with self.assertRaises(ValueError):
            infer_instances(np.zeros((4, 8), dtype=np.uint8), FORMAL_PROMPT, ModelBundle(detector, segmenter), cfg)
        segmenter.predict_masks = lambda image, boxes: np.zeros((1, 4, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            infer_instances(np.zeros((4, 8, 3), dtype=np.uint8), FORMAL_PROMPT, ModelBundle(detector, segmenter), cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
