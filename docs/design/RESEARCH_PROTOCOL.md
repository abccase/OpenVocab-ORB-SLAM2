# Pre-registered Research Protocol

## Research question

On frozen TUM RGB-D sequences, does causally confirmed open-vocabulary dynamic-region feedback change ORB-SLAM2 localization and tracking behavior relative to the compatibility baseline, and under what conditions?

## Primary outcome

Paired difference in ATE translation RMSE in meters:

\[
\Delta\mathrm{ATE}_{s,r}=\mathrm{ATE}_{semantic,s,r}-\mathrm{ATE}_{baseline,s,r},
\]

for sequence `s` and seed/repetition `r`. Negative values favor semantic feedback. Report each pair, per-sequence summaries, overall median paired difference, and paired bootstrap 95% confidence interval.

## Secondary outcomes

RPE translation and rotation at one second, valid pose fraction, lost-frame fraction, relocalization count, tracking time, raw/used/removed keypoints, semantic cache coverage, label stability, map outputs, and online IPC timing. Secondary outcomes are descriptive; no selective promotion to primary is allowed after results are visible.

## Dataset matrix

| Sequence | Role | Why included |
|---|---|---|
| `fr1_desk` | Static control | Checks over-filtering in a conventional office scene |
| `fr1_room` | Static control / loop | Checks long trajectory and loop-closure behavior |
| `fr3_sitting_xyz` | Moderate dynamics | Moving camera with lower human motion |
| `fr3_sitting_halfsphere` | Moderate dynamics | Rotation and translation with lower human motion |
| `fr3_walking_xyz` | High dynamics | Standard walking-person dynamic challenge |
| `fr3_walking_halfsphere` | High dynamics | Dynamic people plus complex camera motion |

H01 archive identity and extracted tree hashes are frozen before any formal cache or experiment is accepted.

## Formal run matrix

Two modes x six sequences x five paired seeds equals 60 valid formal runs. Runs are timestamp-paced. Run order is generated once from seed `23010`, balanced within each sequence, saved before the first formal result, and never regenerated. Bootstrap mask-generation runs are separately named and excluded.

## Frozen semantics

All formal semantic-feedback runs consume the same immutable cache version. Cache generation completes before comparison runs and binds all inputs listed in the design. Runtime prompt changes are allowed only in the online demonstration and never enter formal tables.

## Alignment and metrics

ATE uses rigid SE(3) alignment because RGB-D scale is observable. RPE uses a fixed one-second delta. Association uses the TUM timestamp convention and a maximum timestamp difference of 0.02 s. Metric tool versions, arguments, trajectory hashes, and associated-pose counts are saved.

## Valid run criteria

A valid run must have the expected study ID, sequence, mode, seed, source tree hash, code commit, compatibility baseline tag, cache identity when applicable, prompt hash, configuration hash, complete process exit, trajectory parse success, telemetry coverage, no undeclared degradation, and metric output. Runs failing any criterion remain preserved with `valid=false` and a reason, then are rerun with the same condition.

## Engineering acceptance

Engineering acceptance requires every mode and module to satisfy its tests and artifacts regardless of scientific direction. The online path must demonstrate non-blocking behavior, explicit degradation, and actual telemetry; it is not required to achieve 5 Hz on the laptop.

## Scientific interpretation

- Improvement: paired evidence favors semantic feedback on dynamic sequences without material regression on controls.
- Mixed: benefits depend on sequence or metric, or controls regress.
- Neutral: confidence intervals and effect sizes show no meaningful change.
- Negative: semantic feedback degrades tracking or mapping.

All four outcomes satisfy the research-layer completion rule when the protocol is complete. The report must diagnose feature starvation, mask error, motion-confirmation lag, stale semantics, loop-closure interaction, and timing as applicable.

## Protocol amendments

Before formal runs, a necessary amendment requires user approval and a signed amendment record containing old value, new value, reason, affected phases, and invalidated artifacts. After any formal result exists, changes to primary sequence/mode/repetition/prompt/threshold/alignment definitions create a new study ID rather than rewriting this study.
