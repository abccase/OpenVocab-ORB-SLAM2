from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class InferenceConfig:
    schema: str
    image_long_side: int
    box_threshold: float
    text_threshold: float
    mask_threshold: float

    def __post_init__(self) -> None:
        if not self.schema:
            raise ValueError("schema must not be empty")
        if self.image_long_side <= 0:
            raise ValueError("image_long_side must be positive")
        for name in ("box_threshold", "text_threshold", "mask_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    def to_primitive(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "image_long_side": self.image_long_side,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "mask_threshold": self.mask_threshold,
        }

    def sha256(self) -> str:
        payload = json.dumps(self.to_primitive(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def normalize_formal_prompt(prompt: str) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in prompt.replace("\n", " ").split("."):
        term = " ".join(raw_term.strip().lower().split())
        if term and term not in seen:
            terms.append(term)
            seen.add(term)
    if not terms:
        raise ValueError("formal prompt has no terms")
    return " . ".join(terms) + " ."
