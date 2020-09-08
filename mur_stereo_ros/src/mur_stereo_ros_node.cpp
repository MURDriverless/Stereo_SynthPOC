#include <iostream>
#include <chrono>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

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
#include "PreviewArgs.hpp"

#ifndef SRC_ROOT_PATH
#define SRC_ROOT_PATH "../"
#endif

#define PREVIEW

struct GlobalFrames {
    cv::Mat lFrame;
    cv::Mat rFrame;

    int frameFlag = 0b00;
} globalFrames;

void imageCallback(const sensor_msgs::ImageConstPtr& msg, bool left) {
    try
    {
        if (left) {
            globalFrames.lFrame = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
            globalFrames.frameFlag |= 0b01;
        }
        else {
            globalFrames.rFrame = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
            globalFrames.frameFlag |= 0b10;
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void imageCallbackL(const sensor_msgs::ImageConstPtr& msg) {
    imageCallback(msg, true);
}

void imageCallbackR(const sensor_msgs::ImageConstPtr& msg) {
    imageCallback(msg, false);
}

int main(int argc, char** argv) {
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

    #ifdef PREVIEW
    cv::Mat rFrameBBox;
    cv::Mat lFrameBBox;
    cv::Mat matchesPreview;
    PreviewArgs previewArgs(lFrameBBox, rFrameBBox, matchesPreview);
    #else
    PreviewArgs previewArgs();
    #endif /* PREVIEW */

    // use 2 threads with thread args? Assume always synced
    ros::init(argc, argv, "mur_stereo_ros_node");

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber subL = it.subscribe("mur/stereo_cam/left_image", 1, imageCallbackL);
    image_transport::Subscriber subR = it.subscribe("mur/stereo_cam/right_image", 1, imageCallbackR);

    auto then = std::chrono::high_resolution_clock::now();
    while (ros::ok()) {
        ros::spinOnce();
        if (globalFrames.frameFlag == 0b11) {
            auto now = std::chrono::high_resolution_clock::now();
            double frameTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
            then = now;

            std::cout << "FPS: " << 1000.0/frameTime << std::endl;

            cv::Mat lUnDist, rUnDist;
            classical.preprocessFramePair(globalFrames.lFrame, globalFrames.rFrame, lUnDist, rUnDist);

            std::vector<ConeROI> coneROIs;
            detectors.detectFrame(lUnDist, coneROIs, previewArgs);

            std::vector<ConeEst> coneEsts;

            classical.estConePos(lUnDist, rUnDist, coneROIs, coneEsts, 0, previewArgs);

            #ifdef PREVIEW
            cv::imshow("Camera_Undist1", lFrameBBox);
            cv::resizeWindow("Camera_Undist1", 1000, 600);
            cv::waitKey(1);

            cv::imshow("Camera_Undist2", rFrameBBox);
            cv::resizeWindow("Camera_Undist2", 1000, 600);
            cv::waitKey(1);

            cv::imshow("Matches", matchesPreview);
            cv::resizeWindow("Camera_Undist2", 1000, 600);
            cv::waitKey(1);
            #endif /* PREVIEW */

            globalFrames.frameFlag = 0b00;
        }
    }


    return 0;
}