/** No-op Viewer implementation for Pangolin-free builds. */

#include "Viewer.h"

namespace ORB_SLAM2 {

Viewer::Viewer(System* pSystem, FrameDrawer* pFrameDrawer,
               MapDrawer* pMapDrawer, Tracking* pTracking, const string&)
    : mpSystem(pSystem),
      mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer),
      mpTracker(pTracking),
      mT(0.0),
      mImageWidth(0.0f),
      mImageHeight(0.0f),
      mViewpointX(0.0f),
      mViewpointY(0.0f),
      mViewpointZ(0.0f),
      mViewpointF(0.0f),
      mbFinishRequested(false),
      mbFinished(true),
      mbStopped(true),
      mbStopRequested(false) {}

void Viewer::Run() { SetFinish(); }

void Viewer::RequestFinish() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
    mbFinished = true;
}

void Viewer::RequestStop() {
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = true;
    mbStopped = true;
}

bool Viewer::isFinished() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

bool Viewer::isStopped() {
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

void Viewer::Release() {
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}

bool Viewer::Stop() { return true; }

bool Viewer::CheckFinish() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}

}  // namespace ORB_SLAM2
