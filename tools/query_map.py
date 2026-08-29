#!/usr/bin/env python3
"""Query exported semantic objects with deterministic ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_py.openvocab_slam.query import query_objects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_root", type=Path)
    parser.add_argument("query")
    args = parser.parse_args()
    objects_path = args.map_root / "objects.json"
    objects = json.loads(objects_path.read_text(encoding="utf-8"))
    if not isinstance(objects, list):
        raise ValueError("objects.json must contain a list")
    results = query_objects(objects, args.query)
    print(json.dumps([
        {
            "object_id": result.object_id,
            "match_kind": result.match_kind,
            "confidence": result.confidence,
            "object": result.record,
        }
        for result in results
    ], indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
