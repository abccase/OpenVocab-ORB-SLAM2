# Experiment card

**Purpose.** Evaluate a reconstructed semantic-feedback policy against an ORB-SLAM2 compatibility baseline; engineering completion is separate from a scientific improvement claim.

**Formal protocol.** Six TUM RGB-D sequences, two modes, five paired seeds (`23011`–`23015`), 60 valid conditions. Metrics use timestamp association within 0.02 s, rigid SE(3) ATE, and 1.0 s RPE. The P08 runner rejects missing, duplicate, unpaired, degraded, cache-mismatched, or post-protocol attempts.

**Outcome.** The approved matrix contained 60/60 valid runs and zero invalid attempts. The outcome classification is neutral; the median paired ATE delta is `+0.000407814 m`, with CI `[-0.000586581, +0.005119730] m`. Do not interpret individual runs, P06 online timing, or P07 maps as evidence of positive localization improvement.

**Artifacts.** Ignored machine-readable identities reside under `runs/ovorb2_tum_v1`, `reports/final`, and `artifacts/maps`; use `tools/reproduce.py --asset-root … --validate-existing` to verify them. Protocol details: [P08_STUDY.md](design/P08_STUDY.md).
