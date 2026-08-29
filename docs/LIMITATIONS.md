# Limitations

- P05 establishes compatibility/noninferiority controls, not semantic benefit.
- P06 online evidence is hardware-dependent. The measured P06 demonstration was degraded-only because no packet met the 250 ms causal-age limit; it does not support a realtime or localization claim.
- P07 maps are representative static TSDF/object exports, not ground-truth semantic-map accuracy measurements. Dynamic objects are intentionally not permanently fused.
- P08 is restricted to six TUM RGB-D sequences, frozen prompts/models/caches, paired seeds, and SE(3) metrics. Its neutral result does not generalize to other datasets, cameras, prompts, hardware, or online scheduling.
- The paper leaves several dynamic-scene decisions unspecified. This implementation's score thresholds, association, confirmation, and retention policy are reconstruction choices documented in the design, not recovered facts.
- No model training, live cameras, ROS, TensorRT/ONNX, EuRoC evaluation, or manual ground-truth annotation is included.
