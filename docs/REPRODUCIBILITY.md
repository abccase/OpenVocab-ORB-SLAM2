# Reproducibility

Tracked source is intentionally separate from ignored H01 data, model weights, caches, runs, maps, figures, reports, and local control-plane state. A clean checkout must name an existing controlled asset root:

```bash
python3 tools/reproduce.py --asset-root /abs/primary/OpenVocab-ORB-SLAM2 \
  --output-root /tmp/openvocab-reproduction --validate-existing --smoke
```

The command fails closed if required data/cache/run/map/report directories or their manifest hashes are absent. For a separate clean checkout it links the ignored `external/` dependency tree from the explicit asset root so the full unit contract can validate the pinned compiled extension; it never copies or downloads that tree. It never downloads datasets or weights. It writes `reproduction_manifest.json` and per-stage command logs atomically under the output root.

`--validate-existing` checks the six H01 dataset manifests, frozen semantic/dynamic cache completions, neutral P08 summary/artifact hashes, and P07 map validators. `--smoke` additionally configures/builds the checkout, runs the Python unit suite, and executes exactly one `fr3_walking_xyz` baseline plus one frozen-cache semantic-feedback smoke. Formal frozen-cache mismatch is fatal. Online IPC remains a demonstration path and is not substituted for a smoke semantic result.

Source, cache, run, map, and report identity fields are machine-readable and bound by SHA256 manifests. Reproduction logs and all generated material are ignored by Git; do not copy ignored assets into a clean checkout.
