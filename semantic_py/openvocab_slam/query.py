"""Deterministic text query over exported semantic objects."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Mapping


@dataclass(frozen=True)
class QueryResult:
    object_id: str
    match_kind: str
    confidence: float
    record: Mapping[str, object]


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("query text must be a string")
    normalized = re.sub(r"(?:[^\w]+|_+)", " ", value.lower(), flags=re.UNICODE)
    return " ".join(normalized.split())


def query_objects(
    objects: Iterable[Mapping[str, object]], text: str
) -> list[QueryResult]:
    query = normalize_text(text)
    if not query:
        raise ValueError("query text must not be empty")
    query_tokens = frozenset(query.split())
    matches: list[QueryResult] = []
    for record in objects:
        if not isinstance(record, Mapping):
            raise ValueError("object record must be a mapping")
        object_id = record.get("object_id")
        label = record.get("normalized_label")
        aliases = record.get("aliases", [])
        confidence = record.get("confidence")
        if (not isinstance(object_id, str) or not object_id or
                not isinstance(label, str) or not label or
                not isinstance(aliases, (list, tuple)) or
                not all(isinstance(alias, str) for alias in aliases) or
                type(confidence) not in (int, float) or
                not math.isfinite(float(confidence)) or
                not 0.0 <= float(confidence) <= 1.0):
            raise ValueError("invalid object query record")
        candidates = [normalize_text(label)] + [normalize_text(alias) for alias in aliases]
        if query in candidates:
            kind = "exact"
        elif any(
            query_tokens <= frozenset(candidate.split()) or
            frozenset(candidate.split()) <= query_tokens
            for candidate in candidates if candidate
        ):
            kind = "token"
        else:
            continue
        matches.append(QueryResult(object_id, kind, float(confidence), record))
    matches.sort(key=lambda item: (
        0 if item.match_kind == "exact" else 1,
        -item.confidence,
        item.object_id,
    ))
    return matches
