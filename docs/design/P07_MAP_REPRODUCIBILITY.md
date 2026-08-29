# P07 static map reproducibility

P07 consumes an exact RGB-D association, one validated offline
`semantic-feedback` trajectory at seed `23011`, and the frozen semantic and
dynamic caches. Generated maps are intentionally ignored by Git; all code,
parameters, and dependency versions required to regenerate them are tracked.

Install `requirements/semantic.lock` in a Python 3.10 environment. The map
configuration is frozen in `config/P07_MAP.json`. Open3D performs masked TSDF
integration on CPU; screenshots use a deterministic headless projection.

Create exactly one registered source trajectory for each formal map sequence:

```bash
venv/semantic-gpu/bin/python tools/run_orb_tum.py \
  --mode semantic-feedback --study smoke \
  --sequence fr1_desk --seed 23011

venv/semantic-gpu/bin/python tools/run_orb_tum.py \
  --mode semantic-feedback --study smoke \
  --sequence fr3_walking_xyz --seed 23011
```

Then build the maps from a clean product tree:

```bash
venv/semantic-gpu/bin/python tools/build_map.py --sequence fr1_desk
venv/semantic-gpu/bin/python tools/build_map.py --sequence fr3_walking_xyz

venv/semantic-gpu/bin/python tools/validate_map.py \
  artifacts/maps/<run_id> --report artifacts/maps/<run_id>-integrity.json
```

`tools/build_map.py` fails closed when zero or multiple registered runs match a
sequence, mode, and seed. It also rechecks the selected trajectory, dataset
tree, semantic cache, dynamic cache, and per-frame payload hashes. Each map is
written atomically below `artifacts/maps/<run_id>/` and includes:

- `static_mesh.ply` and `static_cloud.ply`;
- `objects.json` and `objects/*.ply`;
- `dynamic_tracks.jsonl`;
- fixed `screenshots/front.png` and `screenshots/top.png`;
- `map_manifest.json`, which binds every input, parameter, count, and output.

Query an exported map with exact-then-token deterministic ranking:

```bash
venv/semantic-gpu/bin/python tools/query_map.py \
  artifacts/maps/<run_id> "office chair"
```
