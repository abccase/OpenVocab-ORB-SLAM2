# Licenses and derivative-work notice

This is a derivative work of the official ORB-SLAM2 repository at `f2e6f51cdc8d067655d90a78c06261378e07e8f3`. ORB-SLAM2 is `GPL-3.0-or-later`; its upstream license text is retained as `License-gpl.txt` and project license material as `LICENSE.txt`. Redistribution of this derivative must comply with GPLv3 or a later GPL version, preserve the original notices, and make corresponding source available under the applicable GPL terms.

Included ORB-SLAM2 third-party components retain their notices: DBoW2 and g2o license files are in `Thirdparty/DBoW2/LICENSE.txt` and `Thirdparty/g2o/license-bsd.txt`. Pangolin, OpenCV, Eigen, and other build dependencies are independently licensed and must be installed/distributed under their own terms.

The reconstruction records Grounding DINO `856dde20aee659246248e20734ef9ba5214f5e44` and Segment Anything `dca509fe793f601edb92606367a655c15ac00fdf` as Apache-2.0 repositories at those pins. TUM RGB-D is attributed to its official provider and governed by the provider's dataset terms; it is not bundled here. See [SOURCE_PINS.md](design/SOURCE_PINS.md) for the authoritative pin table.

The exact Python package inventory is the tracked, hashable lockfile `requirements/semantic.lock` (SHA256 is recorded by the delivery manifest); it covers the semantic/cache/map/renderer environment. The build inventory is the tracked `CMakeLists.txt`, `Thirdparty/DBoW2/LICENSE.txt`, and `Thirdparty/g2o/license-bsd.txt`; CMake finds system OpenCV, Eigen, OpenSSL, ZeroMQ, and GTest rather than vendoring them. This document intentionally does not claim license versions for locally installed system packages that are not pinned by this repository.
