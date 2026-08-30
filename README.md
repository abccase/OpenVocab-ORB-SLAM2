# OpenVocab-ORB-SLAM2 reconstruction

This repository is a reproducible research reconstruction of a paper framework, not the paper's executable ground truth and not a claim that semantic feedback improves localization. It is a derivative of [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) (upstream `f2e6f51cdc8d067655d90a78c06261378e07e8f3`), retaining its GPLv3 obligations and upstream attribution. The reconstruction adds frozen-cache semantic feedback, asynchronous online demonstration plumbing, and static semantic-map tooling around an ORB-SLAM2 RGB-D compatibility baseline.

The approved six-sequence, 60-run P08 study result is **neutral**: median paired semantic-feedback minus baseline ATE was `+0.000407814 m` (95% paired-bootstrap CI `[-0.000586581, +0.005119730] m`). This does not support a positive localization-improvement claim.

## Modes and evidence boundary

- `baseline` is the compatibility ORB-SLAM2 RGB-D path. It does not read semantic inputs.
- `semantic-feedback` is the formal path. It consumes only a valid frozen causal dynamic-score cache and fails closed on missing or mismatched input.
- `online` is an asynchronous ZeroMQ demonstration. It may degrade frame-by-frame to baseline when a packet is missing, invalid, or older than 250 ms. It is not formal localization evidence.

See [architecture](docs/ARCHITECTURE.md), the [experiment card](docs/EXPERIMENT_CARD.md), [limitations](docs/LIMITATIONS.md), and the canonical [design](docs/design/DESIGN_SPEC.md).

## Build and test

Ubuntu 22.04, C++11, CMake, OpenCV, Eigen, Python 3, NumPy, Open3D, and the Python packages in `requirements/semantic.lock` are the compatibility target. H01 datasets, model weights, caches, runs, maps, and reports are deliberately ignored and never fetched by these commands.

```bash
./build.sh
python3 -m pytest tests/python -q
ctest --test-dir build --output-on-failure
```

## Reproduce approved artifacts

Use a clean checkout and explicitly point it at a controlled existing asset root. The command has no H01 download path and writes atomic logs/manifests outside Git. Run the executed reproduction first, render the delivery from its validated evidence, then run the delivery-validation pass:

```bash
venv/semantic-gpu/bin/python tools/reproduce.py \
  --asset-root /absolute/path/to/primary/OpenVocab-ORB-SLAM2 \
  --output-root /absolute/path/to/primary/OpenVocab-ORB-SLAM2/reports/final/reproduction-<commit> \
  --validate-existing --smoke

venv/semantic-gpu/bin/python tools/render_visual_acceptance.py \
  --asset-root /absolute/path/to/primary/OpenVocab-ORB-SLAM2 \
  --output /absolute/path/to/primary/OpenVocab-ORB-SLAM2/reports/final

venv/semantic-gpu/bin/python tools/reproduce.py \
  --asset-root /absolute/path/to/primary/OpenVocab-ORB-SLAM2 \
  --output-root /tmp/openvocab-delivery-validation \
  --validate-existing
```

The stage order is fixed: `preflight`, `build`, `unit`, `data-validate`, `cache-validate`, `smoke`, `metrics`, `map-validate`. `--smoke` runs one bounded `fr3_walking_xyz` baseline and one frozen-cache `semantic-feedback` run; it never treats online IPC as formal semantics. The final validate-only pass does not pretend to rerun those three stages: it requires their exact-commit logs and smoke manifests, then checks `FINAL_REPORT.md`, the self-contained H02 sheet, its source manifest and every listed source, and the delivery-manifest hashes. [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) describes identities and failure behavior.

## Provenance and licensing

Pinned sources are ORB-SLAM2 `f2e6f51…`, Grounding DINO `856dde20…`, and Segment Anything `dca509fe…`; see [SOURCE_PINS.md](docs/design/SOURCE_PINS.md). ORB-SLAM2 is GPLv3; Grounding DINO and Segment Anything are Apache-2.0 at their pinned revisions; TUM RGB-D remains subject to the provider's stated terms. The derivative-work notice and dependency inventory are in [LICENSES.md](docs/LICENSES.md). Preserve `License-gpl.txt`, `LICENSE.txt`, and third-party notices when redistributing.
