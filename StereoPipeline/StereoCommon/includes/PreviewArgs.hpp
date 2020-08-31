#pragma once

#include <opencv2/core.hpp>

struct PreviewArgs {
    const bool valid;
    cv::Mat* lFrameBBoxMatPtr;
    cv::Mat* rFrameBBoxMatPtr;
    cv::Mat* matchesMatPtr;

    PreviewArgs() : valid(false) {};
    PreviewArgs(cv::Mat& lFrameBBox, cv::Mat& rFrameBBox, cv::Mat& matchesMat) : valid(true) {
        lFrameBBoxMatPtr = &lFrameBBox;
        rFrameBBoxMatPtr = &rFrameBBox;
        matchesMatPtr = &matchesMat;
    };
};