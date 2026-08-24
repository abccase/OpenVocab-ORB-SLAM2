# P05 Baseline Gate Amendment: Noninferiority V2

## Approval and scope

The user approved this amendment on 2026-08-24 with the exact token
`APPROVE_P05_V2_SPEC`. It replaces only the P05 engineering gate that checks
candidate baseline behavior against the legacy RGB-D oracle. The primary P08
study, its hypotheses, modes, sequences, metrics, and analysis remain
unchanged.

## Superseded gate

The superseded gate used five repetitions per sequence, a two-sided median
valid-pose tolerance of 0.005, and an ATE tolerance derived from two pooled
MADs. Three sequences changed pass/fail state across complete batches made
with the same candidate executable. The old gate was therefore too sensitive
to ORB-SLAM2 mapping-thread scheduling variance for its intended purpose.

All completed old `equivalence` attempts and reports are preserved as invalid
pilot evidence. They must not be selected, pooled, or reinterpreted by this
amendment.

## Replacement gate

The replacement study ID is `ovorb2_p05_baseline_noninferiority_v2`. It uses
the six frozen TUM sequences and repetition IDs 23011 through 23025: 15 paired
oracle/candidate blocks per sequence and 180 statistical runs total. The exact
within-block order is frozen in
`config/P05_BASELINE_NONINFERIORITY_V2.json` from `random.Random(23010)`.

The oracle producer remains
`58014b7c1f2b73427b67b4e80a8cf334127f48ea`. The candidate resolves
`HEAD_AT_REGISTRATION`; the ignored immutable batch registration records that
resolved commit and both executable SHA-256 values before any formal run.

For each sequence, the verifier performs 100,000 paired bootstrap resamples
with a per-sequence `numpy.random.Generator(numpy.random.PCG64(23010))`. Both
metrics use the same resampled index matrix and NumPy linear quantiles. The
one-sided acceptance bounds are:

- fifth percentile of mean candidate-minus-oracle valid-pose fraction must be
  at least -0.10;
- exponentiated 95th percentile of mean log candidate/oracle ATE ratio must be
  at most 1.25.

All six sequences must pass both bounds. Deterministic identity, artifact,
telemetry, no-semantic-access, and file-access gates remain hard requirements.
Six separate candidate-baseline `strace` probes are excluded from statistics.

## Failure policy

A failed or invalid condition is preserved and may be retried only for the
same frozen condition when it has no valid completed expected-producer
attempt. A valid completed condition is never rerun. Statistical failure is
not rerun under this study ID to seek a favorable sample; another product
change or performance experiment requires a separately approved amendment and
new study identity.

## Reproducibility and Git boundary

Git records only protocol/configuration, implementation source, tests, design,
and build instructions needed to reproduce the gate. Agent files, phase
reports, run registries, traces, trajectories, telemetry, logs, datasets,
models, caches, build trees, binaries, temporary worktrees, environments, and
maps remain local ignored artifacts. This amendment authorizes no push.
