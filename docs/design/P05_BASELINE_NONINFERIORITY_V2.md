# P05 Baseline Noninferiority V2 Design

## Status and authority

This design implements the user-approved P05 protocol revision dated
2026-08-24. It replaces the failed five-repeat, two-sided baseline-equivalence
gate only. It does not change the primary P08 study matrix, semantic prompt,
dynamic-cache identity, feature-filter thresholds, trajectory alignment, or
metric association rules.

The new engineering study ID is
`ovorb2_p05_baseline_noninferiority_v2`. Results from the earlier
`equivalence` attempts remain preserved and invalid for closing P05. They are
design-time pilot evidence only and must not be pooled with this study.

## Why a new study is required

The original gate compared five-run medians with a valid-pose margin of 0.005
and an ATE margin derived from two pooled MADs. Identical candidate executable
hashes changed pass/fail state across independent five-run batches on three of
six sequences. The gate therefore conflated ORB-SLAM2 mapping-thread scheduling
variance with semantic-source-path drift. A real baseline-path drift was found
and fixed, but the five-run gate remained unstable afterward.

Because formal results already exist, the old study is not rewritten. This
revision creates a separately identified, pre-registered engineering gate.

## Acceptance model

P05 baseline acceptance has two layers:

1. Deterministic path-equivalence requirements prove that candidate baseline
   mode does not consume semantics and preserves the legacy RGB-D feature path.
2. A paired noninferiority study rejects material performance regression while
   reporting all two-sided distribution shifts without treating improvement or
   harmless scheduling drift as failure.

Both layers are mandatory. Passing the statistical layer cannot excuse a
deterministic identity, access, telemetry, or artifact failure.

## Frozen study matrix

The study uses the six sequences already frozen in
`config/EXPERIMENT_MANIFEST.yaml` and repetition identifiers `23011` through
`23025`, inclusive.

For each sequence and repetition identifier, one oracle run and one candidate
run form a block. The complete matrix is:

- 6 sequences;
- 15 paired blocks per sequence;
- 2 implementations per block;
- 180 valid runs total.

The oracle is built from producer commit
`58014b7c1f2b73427b67b4e80a8cf334127f48ea`. The candidate is built from the
final implementation commit that also contains the frozen v2 verifier. The
tracked protocol manifest pins the oracle commit and requires the candidate to
resolve to `HEAD_AT_REGISTRATION`; it cannot contain its own final Git hash
without creating a circular identity. The immutable run-registry batch record
freezes that resolved candidate commit plus both executable SHA-256 values
before the first formal run.

Within every block, oracle-first versus candidate-first is generated from
Python's `random.Random(23010)`. Each sequence has either seven or eight blocks
in each order, and which implementation receives the eighth first position is
also determined before runs begin. The complete order is stored in the tracked
protocol manifest and hashed into every run registration. Formal execution is
sequential; conditions are not parallelized.

The repetition identifier is an identity and pairing key. Baseline ORB-SLAM2
does not claim algorithmic determinism from `ORB_SLAM2_RUN_SEED`.

## Deterministic hard gates

Before statistics are evaluated, all of the following must pass:

- current C++ and Python product tests;
- exact oracle and candidate producer commits and executable hashes;
- compatibility tag, vocabulary, settings, association, dataset manifest,
  extracted source tree, experiment manifest, and protocol-manifest hashes;
- exactly one complete valid attempt for each expected producer, sequence,
  repetition, implementation, and block;
- complete frame telemetry and parseable camera/keyframe trajectories;
- candidate baseline telemetry has `semantic_accessed == 0`, baseline semantic
  state, `raw_keypoints == used_keypoints`, zero removal counts, and zero
  cache/policy timing for every frame;
- candidate registrations contain no cache root or cache identity;
- legacy/optional-null frame-invariant and RGB-D API tests remain green;
- one separately registered `strace -ff -e trace=file` candidate-baseline probe
  per sequence shows no access beneath semantic/dynamic cache roots and no
  access to prompt, inference-model, or semantic-identity files.

Strace probes are access audits, not members of the 180-run statistical matrix.
Their timing and trajectories are excluded from numerical comparison.

## Metrics and paired bootstrap

ATE continues to use rigid SE(3) alignment without scale fitting and TUM
nearest one-to-one association with maximum timestamp difference 0.02 seconds.
Valid-pose fraction continues to come from complete per-frame telemetry.

For sequence `s` and paired block `i`:

```text
pose_delta[s,i] = candidate_valid_pose_fraction - oracle_valid_pose_fraction
ate_log_ratio[s,i] = log(candidate_ate_rmse_m / oracle_ate_rmse_m)
```

ATE values must be finite and strictly positive. Each sequence is evaluated
from exactly 15 pairs.

The verifier performs 100,000 paired bootstrap resamples of the 15 block
indices with replacement. It uses NumPy `Generator(PCG64(23010))`; the
generator is re-created per sequence and the same resampled block-index matrix
is used for both metrics. Quantiles use `numpy.quantile(..., method="linear")`.
The report records NumPy version, algorithm, seed, resample count, and tool hash.

For valid-pose fraction, the estimand is the arithmetic mean of the 15 paired
deltas. The one-sided 95% lower confidence bound is the fifth percentile of
bootstrap mean deltas. The sequence passes when:

```text
pose_lower_95 >= -0.10
```

For ATE, the estimand is the geometric-mean candidate/oracle ratio, computed by
exponentiating the mean paired log ratio. The one-sided 95% upper confidence
bound is the exponential of the 95th percentile bootstrap mean log ratio. The
sequence passes when:

```text
ate_ratio_upper_95 <= 1.25
```

All six sequences must pass both noninferiority criteria. This is an
intersection-union gate, so no multiplicity correction is added. The report
also includes unpaired summaries, observed two-sided shifts, all 15 paired
values, and confidence intervals; those descriptive values cannot be promoted
to new blocking criteria after results are visible.

## Components and data flow

### Tracked protocol manifest

`config/P05_BASELINE_NONINFERIORITY_V2.json` owns the study ID, six sequence
IDs, 15 repetition IDs, exact 180-condition order, oracle producer identity,
candidate policy `HEAD_AT_REGISTRATION`, bootstrap configuration,
noninferiority margins, metric rules, and references to the existing experiment
manifest. It is canonical JSON and is committed before any formal run.

### Reproducible oracle builder

`tools/build_p05_oracle.py` creates an isolated detached worktree at the frozen
oracle commit beneath an explicitly supplied ignored build root. It performs a
headless Release build with the repository's pinned build procedure and writes
an atomic build manifest containing source commit, compiler/CMake/OpenCV/Eigen/
OpenSSL versions, configure arguments, executable path, and executable SHA-256.
It refuses a dirty or wrong-commit worktree. The binary and build manifest are
formal ignored artifacts, not Git content; the builder and build recipe are
tracked.

### Formal runner

`tools/run_p05_baseline_noninferiority.py` validates the protocol manifest,
oracle build manifest, candidate `HEAD`, data identities, and registry path
before launch. It runs the frozen order sequentially and delegates artifact
creation/validation to the existing baseline and OV run functions. Oracle
manifests record the oracle producer, build-manifest hash, and executable;
candidate manifests record the resolved final candidate producer and
executable. Every run records `study_id`, `block_id`, `implementation`, and
protocol-manifest SHA-256.

The runner supports validation-only and resume. Resume ignores complete runs
from other producers or study IDs. It rejects more than one complete valid run
for the same expected producer and condition instead of selecting whichever is
convenient.

### Noninferiority verifier

`tools/verify_baseline_noninferiority.py` independently revalidates all inputs,
manifests, artifact hashes, access state, telemetry, pair membership, and
producer identities. It then computes the frozen statistics and writes an
atomic report. The verifier exits zero only when every deterministic gate and
all 12 sequence/metric noninferiority tests pass.

The existing `tools/verify_baseline_equivalence.py` and its reports remain
historical evidence; v2 does not reinterpret or overwrite them.

### Protocol amendment

`docs/design/amendments/2026-08-24-p05-baseline-noninferiority-v2.md` records
the old gate, new gate, approval, reason, affected phase, preserved invalid
artifacts, new study ID, and the explicit statement that P08's primary study is
unchanged.

## Failure and invalidation rules

- A missing, corrupt, identity-mismatched, incomplete, or semantically accessed
  run is invalid, preserved, and rerun only for the same frozen condition.
- A code change after candidate registration invalidates all candidate runs for
  that producer. A verifier-only code change still changes the trusted producer
  and requires re-registration unless made before the first formal run.
- An oracle rebuild with a different executable hash invalidates all oracle
  runs for the study even if the source commit is unchanged.
- Duplicate valid attempts for the expected producer and condition fail closed.
- A deterministic hard-gate failure blocks statistics and semantic smoke.
- A statistical noninferiority failure blocks semantic smoke. Results are not
  rerun to seek a favorable sample; another change requires a new approved
  amendment and study ID.

## Testing strategy

Development follows TDD. Synthetic tests cover:

- exact 15-pair requirement and repetition pairing;
- frozen balanced order and order-hash validation;
- missing, duplicate, stale-producer, wrong-study, and wrong-block attempts;
- candidate semantic/cache telemetry rejection;
- paired bootstrap reproducibility and shared resample indices;
- exact boundary acceptance at `-0.10` and `1.25`;
- rejection immediately outside each boundary;
- finite/positive ATE and `[0,1]` valid-pose validation;
- atomic complete report publication and producer binding;
- resume behavior without cherry-picking attempts.

After unit and integration tests pass, the six strace probes run first. The
180-run matrix is registered once, executed once, independently verified, and
recorded in the ignored task report and run registry. If v2 passes, run the
already specified single semantic-feedback smoke on `fr3_sitting_xyz`, seed
`23011`, then execute P05 phase-closure verification.

## Reproducibility and Git scope

Git commits contain only tracked protocol/configuration, source, tests, design,
and build instructions required to reproduce the study. They exclude agent
reports, run registries, trajectories, telemetry, logs, caches, datasets,
models, environments, build trees, binaries, maps, and temporary worktrees.
No push is part of P05 execution.
