#include <iostream>

#include <ros/ros.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "Detectors.hpp"
#include "ClassicalStereo.hpp"

#ifndef SRC_ROOT_PATH
#define SRC_ROOT_PATH "../"
#endif

int main(int argc, char** argv) {
    ros::init(argc, argv, "mur_stereo_ros_node");

    std::string lCalibPath = std::string(SRC_ROOT_PATH).append("../calibration.xml");
    std::string rCalibPath = std::string(SRC_ROOT_PATH).append("../calibration.xml");

    const double baseline = 200.00;
    cv::Ptr<cv::Feature2D> featureDetector = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    ClassicalStereo classical(lCalibPath, rCalibPath, baseline, featureDetector, descriptorMatcher);

    // Prep detectors
    std::string coneRT = std::string(SRC_ROOT_PATH).append("../models/yolo4_cones_int8.rt");
    std::string keyPtsONNX = std::string(SRC_ROOT_PATH).append("../models/keypoints.onnx");

    Detectors detectors;
    detectors.initialize(coneRT, keyPtsONNX);

    // use 2 threads with thread args? Assume always synced

    return 0;
}