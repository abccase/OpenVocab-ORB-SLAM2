# P05 Baseline Noninferiority V3 Implementation Plan

1. Preserve all V2 tracked and ignored evidence without reinterpretation.
2. Add the canonical V3 manifest and fail-closed validation of its study ID and
   `1e-5` oracle telemetry tolerance.
3. Add regression coverage that accepts the observed historical rounding,
   rejects values beyond the bound, and keeps candidate matching exact.
4. Update runner/verifier identities and default documentation from V2 to V3
   without changing order, statistics, margins, oracle commit, or run count.
5. Run focused tests, the complete Python suite, compile checks, Release build,
   and CTest before registration.
6. Commit only reproducibility-critical tracked files, rebuild/attest oracle
   and candidate executables, and create a new immutable V3 registration.
7. Run six V3 access audits, validate 0/180, execute the fresh 180-run matrix,
   run the independent verifier, and preserve a complete pass or failure report.
8. Only after a verifier pass, run semantic smoke, update ignored Agent Pack
   closure evidence, verify P05 closure, and request `APPROVE_P05_TO_P06`.
