#include <gtest/gtest.h>

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

namespace {

TEST(G2oAbiTest, OptimizesUnaryProjectionEdgesAcrossSharedLibraryBoundary) {
  g2o::SparseOptimizer optimizer;
  g2o::BlockSolver_6_3::LinearSolverType* linear_solver =
      new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
  g2o::BlockSolver_6_3* block_solver =
      new g2o::BlockSolver_6_3(linear_solver);
  optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(block_solver));

  g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
  pose->setId(0);
  pose->setEstimate(g2o::SE3Quat());
  ASSERT_TRUE(optimizer.addVertex(pose));

  for (int i = 0; i < 32; ++i) {
    g2o::EdgeSE3ProjectXYZOnlyPose* edge =
        new g2o::EdgeSE3ProjectXYZOnlyPose();
    edge->setVertex(0, pose);
    edge->fx = 535.4;
    edge->fy = 539.2;
    edge->cx = 320.1;
    edge->cy = 247.6;
    edge->Xw = Eigen::Vector3d(
        -0.8 + 0.05 * i, -0.4 + 0.025 * i, 2.0 + 0.02 * i);
    edge->setMeasurement(edge->cam_project(edge->Xw));
    edge->setInformation(Eigen::Matrix2d::Identity());
    ASSERT_TRUE(optimizer.addEdge(edge));
  }

  ASSERT_TRUE(optimizer.initializeOptimization());
  EXPECT_GE(optimizer.optimize(2), 1);
}

}  // namespace
