#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "tkDNN/Yolo3Detection.h"
#include "KeypointDetector.hpp"

struct ConeROI {
    cv::Rect roiRect;
    float x, y, w, h;
    std::vector<cv::Point2f> keypoints;
};

class Detectors {
    private:
        std::unique_ptr<tk::dnn::DetectionNN> detNN;
        std::unique_ptr<KeypointDetector> keypointDetector;
        std::vector<tk::dnn::box> bbox;

        // yolo setup
        static const int n_classes = 3;
        static const int n_batch = 1;

        // tkdnn setup
        static const int keypointsW = 80;
        static const int keypointsH = 80;
        static const int maxBatch = 100;
    public:
        Detectors();
        ~Detectors();
        void initialize(std::string objectModel, std::string featureModel);
        void detectFrame(const cv::Mat &imageFrame, std::vector<ConeROI> &coneROIs);
};