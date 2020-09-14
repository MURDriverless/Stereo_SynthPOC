#pragma once

#include "ConeColorID.hpp"
#include <opencv2/core.hpp>

struct ConeROI {
    cv::Rect roiRect;
    float x, y, w, h;
    std::vector<cv::Point2f> keypoints;

    ConeColorID colorID;
};