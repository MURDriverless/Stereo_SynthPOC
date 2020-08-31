#pragma once

#include <opencv2/core.hpp>

struct PreviewArgs {
    const bool valid;
    cv::Mat* rFrameBBoxMatPtr;
    cv::Mat* matchesMatPtr;

    PreviewArgs() : valid(false) {};
    PreviewArgs(cv::Mat& rFrameBBox, cv::Mat& matchesMat) : valid(true) {
        rFrameBBoxMatPtr = &rFrameBBox;
        matchesMatPtr = &matchesMat;
    };
};