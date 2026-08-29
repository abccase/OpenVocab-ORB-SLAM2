from __future__ import annotations

from semantic_py.openvocab_slam.query import query_objects


def object_record(object_id: str, label: str, confidence: float, aliases=()):
    return {
        "object_id": object_id,
        "normalized_label": label,
        "aliases": list(aliases),
        "confidence": confidence,
    }


def test_query_ranks_exact_before_token_even_when_token_has_higher_confidence() -> None:
    objects = [
        object_record("obj-0007", "chair", 0.99),
        object_record("obj-0003", "office chair", 0.70),
        object_record("obj-0001", "monitor", 1.0),
    ]

    result = query_objects(objects, "  Office,   Chair!! ")

    assert [item.object_id for item in result] == ["obj-0003", "obj-0007"]
    assert [item.match_kind for item in result] == ["exact", "token"]


def test_query_uses_aliases_then_stable_confidence_and_id_order() -> None:
    objects = [
        object_record("obj-0009", "seat", 0.80, aliases=("desk chair",)),
        object_record("obj-0002", "chair", 0.90),
        object_record("obj-0001", "chair", 0.90),
    ]

    result = query_objects(objects, "chair")

    assert [item.object_id for item in result] == ["obj-0001", "obj-0002", "obj-0009"]
    assert [item.match_kind for item in result] == ["exact", "exact", "token"]
