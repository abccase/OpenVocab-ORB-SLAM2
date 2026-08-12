#!/usr/bin/env python3
from __future__ import annotations

import unittest

from tools.audit_compatibility import audit_patch


class CompatibilityAuditTests(unittest.TestCase):
    def test_allows_opencv_namespace_compatibility_in_tracking(self) -> None:
        patch = """diff --git a/src/Tracking.cc b/src/Tracking.cc
--- a/src/Tracking.cc
+++ b/src/Tracking.cc
@@ -1 +1 @@
-            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
+            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
"""
        self.assertEqual(audit_patch(patch), [])

    def test_rejects_tracking_threshold_change(self) -> None:
        patch = """diff --git a/src/Tracking.cc b/src/Tracking.cc
--- a/src/Tracking.cc
+++ b/src/Tracking.cc
@@ -1 +1 @@
-    const int minMatches = 15;
+    const int minMatches = 10;
"""
        violations = audit_patch(patch)
        self.assertEqual(len(violations), 2)
        self.assertTrue(all("src/Tracking.cc" in item for item in violations))

    def test_rejects_any_optimizer_change(self) -> None:
        patch = """diff --git a/src/Optimizer.cc b/src/Optimizer.cc
--- a/src/Optimizer.cc
+++ b/src/Optimizer.cc
@@ -1 +1 @@
-    optimizer.initializeOptimization();
+    optimizer.initializeOptimization(0);
"""
        violations = audit_patch(patch)
        self.assertEqual(len(violations), 2)
        self.assertTrue(all("src/Optimizer.cc" in item for item in violations))


if __name__ == "__main__":
    unittest.main(verbosity=2)
