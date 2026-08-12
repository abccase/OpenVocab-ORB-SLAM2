# Paper-to-Implementation Traceability

| Paper statement or element | Rebuild interpretation | Verification | Claim rule |
|---|---|---|---|
| ORB-SLAM2 geometric thread | Official RGB-D ORB-SLAM2 at the frozen commit with an isolated Ubuntu 22.04 compatibility layer | Build, API smoke, baseline oracle | May claim official-base secondary development |
| “Unmodified ORB-SLAM2” | Applies only to `baseline`; semantic feedback necessarily adds a mask-aware path | Golden baseline equivalence and mode tests | Do not describe feedback mode as unmodified |
| Grounding DINO + SAM | Swin-T detector followed by box-prompted SAM ViT-B | Fake-model unit tests, real smoke, cache manifests | May claim open-vocabulary candidate generation, not universal recognition |
| Asynchronous semantic thread at 5 Hz | Non-blocking latest-frame IPC with request rate capped at 5 Hz; previously confirmed 3D tracks provide causal predicted feedback; actual achieved rate reported | Queue/fault/online-state tests and online telemetry | Never claim 5 Hz achieved unless measured; formal accuracy conclusions use the frozen cache |
| Semantic packet `(label, mask, score)` | Versioned packet with frame/time/model/prompt identity and RLE mask | JSON schema and cross-language fixtures | May claim packet compatibility only for schema v1 |
| Reprojection and label propagation | RGB-D back-projection, explicit coordinate adapters, track IDs, TSDF/object export | Synthetic projection and round-trip tests | Do not claim 3D accuracy without ground truth |
| DBSCAN `eps=0.1`, `min_samples=5` | Used only as an optional object-point cleanup initialized at these paper values; robust boxes use track-associated points | Parameter manifest and ablation note | Do not present the paper values as validated optimum |
| Kalman handling of dynamic objects | Constant-velocity 3D track prediction with Hungarian association | Crossing, occlusion, and lifecycle tests | May claim implemented temporal tracking |
| Dynamic feature filtering | Open-vocabulary instance candidates plus depth/pose-based multi-frame motion confirmation | Static-person and moving-person synthetic/real diagnostics | May claim measured effect, not guaranteed improvement |
| Structured cuboids | Robust oriented boxes with explicit fallback | Synthetic cuboid error test and map manifest | Call them fitted boxes, not exact object geometry |
| TUM RGB-D experiments | Six frozen TUM sequences, including four dynamic and two control sequences | H01 inventory and 60-run matrix | Primary empirical evidence |
| EuRoC experiments | Excluded because the paper's RGB-D semantic fusion and EuRoC stereo setup are not closed technically | Scope audit | Never imply EuRoC was reproduced |
| Centimeter-level table values | Reference claims only; not success thresholds | Rebuild's own SE(3) ATE/RPE tables | Never reuse paper numbers as measured results |
| Sim(3) wording in APE figure | RGB-D primary evaluation uses metric SE(3), no scale fitting | Metric manifest and command logs | Sim(3) may appear only as labeled sensitivity analysis |
| Real-time structured map | Tracking stays non-blocking; TSDF/object fusion may run offline or quasi-online | Separate timing reports | Do not combine cached/offline time with online rate |

## Paper gaps resolved by this design

The paper does not uniquely specify dynamic classes, cross-frame IDs, mask-to-frame timing, failure semantics, prompt versioning, feature-filter insertion, cache identity, coordinate conventions, statistical repetitions, or baseline equivalence. This package makes each item explicit and testable. The chosen behavior is a reconstruction decision, not a quotation of the paper.
