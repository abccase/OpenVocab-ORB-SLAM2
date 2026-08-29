# P08 pre-registered study execution

`tools/run_study.py` is the only formal launcher for study `ovorb2_tum_v1`.
It freezes one balanced 60-row order with Python `random.Random(23010)`, hashes
the CSV before registration, binds the current Git commit, executable,
vocabulary, datasets, prompt, and immutable caches, and preserves every attempt.

Formal execution is sequential and resumable:

```bash
python3 tools/run_study.py --manifest config/EXPERIMENT_MANIFEST.yaml --freeze-order
python3 tools/run_study.py --manifest config/EXPERIMENT_MANIFEST.yaml --resume
python3 tools/validate_runs.py --expect 60 --strict
python3 tools/analyze_study.py --runs runs/ovorb2_tum_v1 --output reports/final
```

Resume skips an attempt only after replaying artifact hashes, telemetry coverage,
trajectory parsing, and metric output. ATE uses rigid SE(3) alignment with scale
fixed to one, timestamp association at no more than 0.02 seconds, and RPE at a
one-second delta. The primary contrast is semantic-feedback minus baseline ATE
RMSE for each sequence and seed. The two-sided 95% interval is a deterministic
paired bootstrap of the median using NumPy PCG64, seed 23010, 100000 resamples,
and linear quantiles.

Outcome classification is fixed in the analyzer: improvement requires an
overall interval below zero, negative requires an overall interval above zero,
neutral requires the overall and every sequence interval to contain zero, and
all other outcomes are mixed. Representative P07 map integrity metrics are
reported explicitly as separate mapping evidence; the P08 localization runner
does not silently rebuild or relabel TSDF maps.

All generated runs, registries, tables, figures, maps, logs, and reports remain
outside Git. Only the launcher, validator, analyzer, protocol documentation, and
tests are reproducibility-critical tracked content.
