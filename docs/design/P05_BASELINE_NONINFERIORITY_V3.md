# P05 Baseline Noninferiority V3 Specification

V3 is the approved successor to V2. The complete V2 design remains normative
except where this document and the V3 amendment explicitly replace it.

## Canonical identities

- Study ID: `ovorb2_p05_baseline_noninferiority_v3`
- Protocol manifest: `config/P05_BASELINE_NONINFERIORITY_V3.json`
- Oracle producer: `58014b7c1f2b73427b67b4e80a8cf334127f48ea`
- Candidate producer policy: `HEAD_AT_REGISTRATION`
- Sequences, repetitions, order, statistics, margins, and deterministic gates:
  identical to V2 and frozen by the canonical V3 manifest
- Oracle telemetry timestamp tolerance: absolute difference `<= 1e-5` seconds
- Candidate telemetry timestamp rule: exact numeric equality

## Oracle telemetry rule

For row `i`, parse the association timestamp and oracle JSONL timestamp as
finite binary floating-point values and require:

```text
abs(oracle_timestamp[i] - association_timestamp[i]) <= 0.00001
```

The verifier must also require contiguous zero-based frame indices, exact frame
coverage, row-order preservation, finite nonnegative tracking time, a Boolean
pose flag, and strictly increasing association timestamps. The tolerance does
not permit nearest-neighbor row reassignment.

## Invalidation boundary

V2 artifacts are preserved as an incomplete verifier outcome. V3 must build or
attest executables at its registered candidate commit, create a new immutable
registration, run six new access probes, and execute a fresh 180-run matrix.
No V2 metric result may be reused in the V3 report.

All remaining requirements, including report atomicity, access-trace reparsing,
paired statistics, smoke ordering, phase closure, and Git exclusions, are those
of `docs/design/P05_BASELINE_NONINFERIORITY_V2.md`.
