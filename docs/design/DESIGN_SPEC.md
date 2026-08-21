# OpenVocab-ORB-SLAM2 Rebuild Design

## 1. Goal and claim boundary

Build a reproducible RGB-D research prototype that reconstructs the paper's three functional modules: ORB-SLAM2 geometry, asynchronous Grounding DINO + SAM semantics, and semantic-geometric 3D fusion. The rebuild adds a second, explicitly tested semantic-feedback mode so the proposition “semantics can improve tracking in dynamic scenes” can be evaluated rather than assumed.

The paper is a requirements source. It is not treated as executable ground truth. Its numerical results, “unmodified ORB-SLAM2” statement, EuRoC/RGB-D mismatch, realtime wording, and underspecified dynamic logic are documented in `PAPER_TRACEABILITY.md` and are not inherited as verified facts.

## 2. System modes

### `baseline`

The Ubuntu 22.04 compatibility version of official ORB-SLAM2 RGB-D. It receives the original RGB and depth frames and no semantic data. All semantic code paths are bypassed. A golden oracle verifies trajectory format, frame-state telemetry, and representative trajectories against the compatibility tag.

### `semantic-feedback`

The same geometric pipeline receives a causal dynamic-score mask before feature use. Features in high-confidence dynamic regions are removed. Features in uncertain regions are deterministically retained at the frozen fraction using a stable hash of sequence ID, frame timestamp, integer pixel coordinate, and seed. Static/low-score features are unchanged. This realizes downweighting without changing ORB descriptor distance or bundle-adjustment residual definitions.

Formal `semantic-feedback` runs require a valid frozen cache and fail closed. Online demonstration runs may explicitly degrade per frame to baseline behavior.

## 3. End-to-end data flow

1. H01 supplies six TUM archives. The ingestion tool safely extracts and creates deterministic RGB-depth associations.
2. The compatibility baseline produces a bootstrap trajectory for each sequence. These bootstrap runs are inputs to dynamic-mask generation and are excluded from the baseline-versus-feedback result matrix.
3. The Python semantic cache tool applies the frozen prompt to every associated RGB frame, runs Grounding DINO Swin-T, then SAM ViT-B, and writes atomic, versioned instance packets.
4. The dynamic reasoning pass uses the RGB-D frame, camera intrinsics, bootstrap pose, and instance masks to estimate robust 3D centroids, associate instances, update Kalman states, and produce a causal dynamic probability.
5. A second-pass frozen dynamic-score cache stores per-instance tracks and per-pixel score maps. Its manifest binds dataset tree hash, source trajectory hash, model hashes, prompt hash, thresholds, code commit, and schema.
6. The ORB-SLAM2 semantic runner loads the score map matching the source timestamp, filters or partially retains features, estimates pose, and writes trajectory plus frame telemetry.
7. The mapping tool uses selected valid trajectory output, original RGB-D frames, instance tracks, and dynamic scores to fuse a static TSDF and object records.
8. The experiment harness validates run manifests and computes trajectory, tracking, runtime, semantic, IPC, and map metrics.

## 4. Module boundaries and interfaces

### 4.1 ORB-SLAM2 compatibility layer

P01 changes only build and API compatibility required by Ubuntu 22.04/OpenCV 4 and optional headless operation. It must not change thresholds, feature extraction, matching, keyframe decisions, local mapping, loop closure, or optimization. This commit is tagged `baseline/ubuntu22`.

### 4.2 C++ semantic mask policy

Create:

```cpp
namespace ORB_SLAM2::semantic {
enum class SemanticState { BASELINE, CACHE_VALID, ONLINE_VALID, DEGRADED_TO_BASELINE };
struct DynamicScoreMap {
  double source_timestamp;
  cv::Mat scores_f32;  // CV_32FC1, [0,1], same resolution as RGB
  std::string manifest_sha256;
};
struct FeatureDecision {
  bool keep;
  float semantic_weight;
  const char* reason;
};
FeatureDecision decideFeature(float score, int x, int y,
                              std::uint64_t frame_key,
                              const PolicyConfig& config);
}
```

Add optional score-map overloads to `System::TrackRGBD`, `Tracking::GrabImageRGBD`, and the RGB-D `Frame` constructor. The original signatures remain on their dedicated compatibility implementation; the offline baseline runner calls those signatures directly and obtains passive feature counts only after tracking returns. The semantic overload dispatches a null map to the same compatibility `Frame` path. The baseline path must not allocate, load, or inspect semantic assets. This separation is deliberate: routing the baseline hot path through semantic-aware overloads measurably perturbs ORB-SLAM2's nondeterministic local-mapping schedule.

Filter keypoints and aligned descriptor rows after ORB extraction and before depth association, grid assignment, matching, and map-point creation. Preserve vector/descriptor index invariants. Raw and used counts are written to telemetry.

### 4.3 Python semantic cache

Package root: `semantic_py/openvocab_slam/`.

```python
@dataclass(frozen=True)
class InstanceObservation:
    local_id: int
    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask_rle: dict[str, object]

def infer_instances(image_bgr: np.ndarray, prompt: str,
                    models: ModelBundle, cfg: InferenceConfig
                    ) -> list[InstanceObservation]: ...

def write_cache_frame(path: Path, packet: SemanticFramePacket) -> str: ...
def validate_cache(root: Path, expected: CacheManifest) -> CacheValidation: ...
```

Frame files are compressed MessagePack with COCO-style RLE masks. An index JSONL maps `frame_id` and exact RGB timestamp to file path and SHA256. Writes use `.<name>.partial`, fsync, atomic rename, then index append. Resume skips only entries whose identity and hash validate.

### 4.4 Instance association and motion confirmation

For each mask, valid depth pixels are back-projected and transformed to the world frame with the bootstrap pose. Use the component-wise median as centroid and median absolute deviation as uncertainty. Observations with fewer than 100 valid depth pixels cannot generate strong dynamic evidence.

Association cost is:

\[
C_{ij}=0.55\,\hat d_{3D}+0.30(1-\mathrm{IoU})+0.15(1-s_{label}),
\]

where 3D distance is normalized by the 1.0 m gate and label similarity is one for normalized exact label matches, zero otherwise. Gated pairs are solved by Hungarian assignment. A constant-velocity Kalman filter tracks position and velocity. Tracks survive five misses.

Motion evidence compares measured centroid with the predicted world centroid after camera-motion compensation. The motion threshold is the larger of 0.10 m and three robust sigmas. New tracks require three confirming observations before strong filtering. A probability accumulator enters dynamic at 0.70 and exits at 0.40. This hysteresis prevents frame-to-frame state chatter.

The frozen mask used at frame `t` is based only on observations at or before `t`; no future frame may influence formal feedback. Unit tests construct crossing objects, occlusion, missing depth, camera-only motion, static persons, and truly moving persons.

### 4.5 Online IPC

C++ and Python are separate processes using ZeroMQ PUB/SUB with one MessagePack blob per message. The C++ publisher caps requests at 5 Hz and uses high-water mark one with conflation. It never waits for send or receive. The Python service consumes the latest available frame and publishes a packet conforming to `config/PROTOCOL_SCHEMA.json`.

A packet is usable only when protocol version, run ID, prompt hash, model manifest hash, image dimensions, source timestamp, and maximum mask age validate. Raw online instance masks are candidates, not automatically dynamic. After ORB-SLAM2 completes the packet's source frame, an `OnlineDynamicState` receives its depth and `T_world_camera`, updates the same causal centroid/Kalman/hysteresis logic, and stores confirmed 3D tracks. Before a later frame, it projects only previously confirmed tracks with the last completed pose to create a predicted score map. New or unconfirmed instances carry score 0.25 and do not trigger strong filtering. Predictions older than 250 ms are unusable.

Invalid or absent packets set `DEGRADED_TO_BASELINE`, keep all baseline features, and log the reason. A service crash cannot terminate or block SLAM. The online demonstration verifies functional asynchronous feedback and failure behavior; formal localization conclusions come only from the frozen-cache study because online model timing is hardware-dependent.

### 4.6 Static TSDF and object map

The Python mapping process uses Open3D TSDF with frozen configuration. Pixels whose dynamic score is at least 0.70 are not integrated into the static volume. Uncertain pixels are excluded from object geometry but may enter the static TSDF with their measured confidence recorded. Camera poses are explicit `T_world_camera`; adapters and synthetic tests prevent inversion errors.

Outputs:

- `static_mesh.ply` and `static_cloud.ply`;
- per-object point clouds;
- `objects.json` containing ID, normalized label, aliases, confidence history, observation range, centroid, orientation, extent, point count, and source track;
- `dynamic_tracks.jsonl` for moving objects that are not fused permanently;
- `map_manifest.json` binding all inputs and parameters.

A robust oriented box is fit from trimmed object points; degenerate objects fall back to an axis-aligned box and record the fallback. Query uses normalized exact match followed by token containment and returns stable sort by confidence then object ID.

## 5. Data and artifact layout

```text
data/tum/inbox/                 H01 archives, ignored
data/tum/raw/                   safely extracted datasets, ignored
external/                       pinned source/model checkouts, ignored
weights/                        model weights and checksums, ignored
cache/semantic/v1/              immutable instance cache, ignored
cache/dynamic/v1/               immutable tracks and score maps, ignored
runs/<study>/<run_id>/          manifests, logs, trajectories, telemetry
artifacts/maps/<run_id>/        TSDF and object map exports
reports/phases/                 phase evidence
reports/final/                  final tables, figures, and acceptance sheet
```

Every generated directory contains a manifest with schema, producer commit, exact command, input hashes, output hashes, creation time, hostname, and validity state. Formal results never depend on an unmanifested file.

## 6. Failure behavior

| Failure | Formal offline behavior | Online demonstration behavior |
|---|---|---|
| Cache missing/hash mismatch | Abort run; invalid manifest | Degrade to baseline and log |
| Prompt/model/schema mismatch | Abort run | Reject packet; degrade |
| Stale mask | Abort if cache association is ambiguous | Degrade for that frame |
| Semantic service crash | Not applicable | SLAM continues; restart service separately |
| Depth missing in an instance | Keep as unknown; no strong filtering | Same |
| Tracking lost | Record lost frame; do not invent pose | Same |
| GPU out of memory | Apply one frozen resolution fallback and regenerate entire affected cache | Record; service may CPU-fallback only for functional demo |
| Partial long run | Preserve registry and resume from validated atomic outputs | Preserve logs and restart |

## 7. Test architecture

C++ uses GoogleTest for policy, frame invariants, cache provider, IPC client, and coordinate adapters. Python uses pytest for schemas, cache atomicity, inference adapter with fakes, centroid geometry, association, Kalman lifecycle, dynamic hysteresis, TSDF masking, query, manifests, and statistics. Integration tests use tiny synthetic RGB-D fixtures checked into `tests/fixtures/`; no model or TUM download is required for unit tests.

P05 includes a baseline equivalence test against artifacts from `baseline/ubuntu22`. P06 fault-injects absent, stale, corrupt, and wrong-version packets and kills the semantic service while SLAM is active. P08 rejects missing, duplicate, unpaired, degraded, cache-mismatched, or post-protocol runs.

## 8. Scope exclusions

The first version does not include EuRoC, live cameras, ROS or ROS2, embedded Python, TensorRT/ONNX conversion, model training, appearance ReID, OctoMap, Gaussian Splatting, dense learned SLAM, interactive relabeling, or manual ground-truth annotation.
