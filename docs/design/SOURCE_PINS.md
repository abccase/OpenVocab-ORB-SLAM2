# Official Source Pins

| Component | Official source | Frozen identity | Use |
|---|---|---|---|
| ORB-SLAM2 | `https://github.com/raulmur/ORB_SLAM2.git` | `f2e6f51cdc8d067655d90a78c06261378e07e8f3` | RGB-D geometry baseline and fork base |
| Grounding DINO | `https://github.com/IDEA-Research/GroundingDINO.git` | `856dde20aee659246248e20734ef9ba5214f5e44` | Open-vocabulary boxes and phrase scores |
| Segment Anything | `https://github.com/facebookresearch/segment-anything.git` | `dca509fe793f601edb92606367a655c15ac00fdf` | Box-prompted instance masks |
| TUM RGB-D | `https://cvg.cit.tum.de/data/datasets/rgbd-dataset` | Six H01 archive basenames plus recorded SHA256 | RGB-D, timestamps, ground truth |

ORB-SLAM2 is GPLv3. Grounding DINO and Segment Anything repositories are Apache-2.0 at the pins above. TUM RGB-D data is attributed and cited under the terms stated by its official site. The Agent must copy upstream license files, add a derivative-work notice, and generate a dependency license inventory before P09 closure.

A source pin may change only when the exact asset is unavailable or incompatible and the user approves a protocol amendment before results are generated. The old and new identities, reason, validation, and affected caches must be recorded; all affected artifacts must be regenerated.
