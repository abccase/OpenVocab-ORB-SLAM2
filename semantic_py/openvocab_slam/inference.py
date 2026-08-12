from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

from .schemas import InstanceObservation, encode_binary_mask_rle


@dataclass(frozen=True)
class ModelBundle:
    detector: object
    segmenter: object


class GroundingDinoDetector:
    """Small stable adapter around the pinned GroundingDINO inference API."""

    def __init__(self, model, device, image_long_side, *, preprocess=None, predict_fn=None):
        self.model = model
        self.device = str(device)
        self.image_long_side = int(image_long_side)
        self._preprocess = preprocess or _preprocess_grounding_dino_image
        if predict_fn is None:
            from groundingdino.util.inference import predict as predict_fn
        self._predict = predict_fn

    def predict(self, image_bgr, prompt, box_threshold, text_threshold):
        transformed = self._preprocess(image_bgr, self.image_long_side)
        boxes, scores, phrases = self._predict(
            model=self.model,
            image=transformed,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
            remove_combined=True,
        )
        boxes_array = _as_numpy(boxes)
        scores_array = _as_numpy(scores)
        if boxes_array.ndim != 2 or boxes_array.shape[1:] != (4,):
            raise ValueError("GroundingDINO boxes must have shape Nx4")
        if scores_array.shape != (boxes_array.shape[0],) or len(phrases) != boxes_array.shape[0]:
            raise ValueError("GroundingDINO outputs have inconsistent lengths")
        return [
            {
                "box_cxcywh_norm": tuple(float(value) for value in box),
                "score": float(score),
                "label": str(phrase),
            }
            for box, score, phrase in zip(boxes_array, scores_array, phrases)
        ]


class SamSegmenter:
    """Batch box-prompt adapter around the pinned SAM predictor."""

    def __init__(self, predictor):
        self.predictor = predictor

    def predict_masks(self, image_bgr, boxes_xyxy):
        import torch

        image = np.asarray(image_bgr)
        self.predictor.set_image(image, image_format="BGR")
        boxes = torch.as_tensor(
            np.asarray(boxes_xyxy, dtype=np.float32),
            dtype=torch.float32,
            device=self.predictor.device,
        )
        transformed = self.predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        logits, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            mask_input=None,
            multimask_output=False,
            return_logits=True,
        )
        probabilities = torch.sigmoid(logits[:, 0])
        return probabilities.detach().cpu().numpy()


def load_model_bundle(
    dino_config,
    dino_weights,
    sam_weights,
    *,
    bert_directory,
    device,
    image_long_side,
):
    """Load the exact GroundingDINO Swin-T and SAM ViT-B models used by P03."""
    import torch
    from groundingdino.models import build_model
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig
    from segment_anything import SamPredictor, sam_model_registry

    args = SLConfig.fromfile(str(dino_config))
    args.device = str(device)
    args.text_encoder_type = str(bert_directory)
    dino = build_model(args)
    checkpoint = torch.load(str(dino_weights), map_location="cpu", weights_only=True)
    dino.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    dino.eval().to(device)
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_weights)).to(device)
    return ModelBundle(
        detector=GroundingDinoDetector(dino, device, image_long_side),
        segmenter=SamSegmenter(SamPredictor(sam)),
    )


def infer_instances(image_bgr, prompt, models, cfg):
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be an HxWx3 uint8 array")
    height, width = image.shape[:2]
    predictions = models.detector.predict(
        image,
        prompt,
        cfg.box_threshold,
        cfg.text_threshold,
    )
    survivors: list[tuple[tuple[float, float, float, float], float, str]] = []
    for prediction in predictions:
        score = float(prediction["score"])
        if not math.isfinite(score) or score < cfg.box_threshold or score > 1.0:
            continue
        cx, cy, box_width, box_height = (
            float(value) for value in prediction["box_cxcywh_norm"]
        )
        if not all(math.isfinite(value) for value in (cx, cy, box_width, box_height)):
            continue
        x1 = min(float(width), max(0.0, (cx - box_width / 2.0) * width))
        y1 = min(float(height), max(0.0, (cy - box_height / 2.0) * height))
        x2 = min(float(width), max(0.0, (cx + box_width / 2.0) * width))
        y2 = min(float(height), max(0.0, (cy + box_height / 2.0) * height))
        if x2 <= x1 or y2 <= y1:
            continue
        label = _normalize_phrase(str(prediction["label"]))
        if not label:
            continue
        survivors.append(((x1, y1, x2, y2), score, label))
    if not survivors:
        return []
    boxes = [item[0] for item in survivors]
    masks = np.asarray(models.segmenter.predict_masks(image, boxes))
    if masks.shape != (len(survivors), height, width):
        raise ValueError(
            f"segmenter returned shape {masks.shape}, expected {(len(survivors), height, width)}"
        )
    output = []
    for local_id, ((box, score, label), mask) in enumerate(zip(survivors, masks)):
        binary_mask = np.asarray(mask >= cfg.mask_threshold, dtype=np.bool_)
        output.append(
            InstanceObservation(
                local_id=local_id,
                label=label,
                score=score,
                box_xyxy=box,
                mask_rle=encode_binary_mask_rle(binary_mask),
            )
        )
    return output


def _normalize_phrase(phrase: str) -> str:
    without_score = re.sub(r"\s*\([^)]*\)\s*$", "", phrase)
    return " ".join(without_score.strip().strip(".").lower().split())


def _preprocess_grounding_dino_image(image_bgr, image_long_side):
    import cv2
    from PIL import Image
    import groundingdino.datasets.transforms as transforms

    image_array = np.asarray(image_bgr)
    height, width = image_array.shape[:2]
    scale = image_long_side / max(height, width)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    transform = transforms.Compose(
        [
            transforms.RandomResize([(target_width, target_height)]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    transformed, _ = transform(Image.fromarray(image_rgb), None)
    return transformed


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
