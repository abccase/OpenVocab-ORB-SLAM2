/** Headless MapDrawer implementation preserving tracking pose updates. */

#include "MapDrawer.h"

namespace ORB_SLAM2 {

MapDrawer::MapDrawer(Map* pMap, const string&): mpMap(pMap) {}

void MapDrawer::SetCurrentCameraPose(const cv::Mat& Tcw) {
    std::unique_lock<std::mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::SetReferenceKeyFrame(KeyFrame*) {}

}  // namespace ORB_SLAM2
