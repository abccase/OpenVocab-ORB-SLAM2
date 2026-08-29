# Architecture

The system has three separated paths. The baseline calls the original RGB-D tracking signature and does not allocate or inspect semantic assets. Formal semantic-feedback loads an immutable per-frame dynamic-score cache, removes high-confidence dynamic features before depth association/matching, and deterministically retains uncertain features. A cache mismatch is fatal.

Semantic cache generation uses pinned Grounding DINO boxes and SAM masks. Causal motion confirmation combines RGB-D centroids, bootstrap poses, association, Kalman state, and hysteresis before writing frozen score maps. Static TSDF fusion excludes pixels with score at least 0.70 and exports object records and dynamic tracks in the explicit `T_world_camera` frame.

The online path uses nonblocking PUB/SUB MessagePack messages at a maximum 5 Hz. It is isolated from the formal result: absent/stale/invalid packets cause `DEGRADED_TO_BASELINE`, never blocking SLAM. Module contracts and exact interfaces are in [DESIGN_SPEC.md](design/DESIGN_SPEC.md).
