#include <iostream>

#include <opencv2/core.hpp>

#include "Detectors.hpp"
#include "ClassicalStereo.hpp"
#include "PreviewArgs.hpp"

#define PREVIEW

#ifndef SRC_ROOT_PATH
#define SRC_ROOT_PATH "./"
#endif

int main(int argc, char** argv) {
    #ifdef PREVIEW
    cv::Mat rFrameBBox;
    cv::Mat lFrameBBox;
    cv::Mat matchesPreview;
    PreviewArgs previewArgs(lFrameBBox, rFrameBBox, matchesPreview);
    #else
    PreviewArgs previewArgs = PreviewArgs();
    #endif /* PREVIEW */

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

    const char imFolder[] = "../Output1/Blue";
    char pathBuf[256];
    int frameNum = 1;

    while (true) {
        std::snprintf(pathBuf, 256, "%s/Image%04d_L.png", imFolder, frameNum);
        cv::Mat lFrame = cv::imread(pathBuf);

        if (lFrame.data == nullptr) {
            break;
        }

        std::snprintf(pathBuf, 256, "%s/Image%04d_R.png", imFolder, frameNum);
        cv::Mat rFrame = cv::imread(pathBuf);

        // Start Proc Block
        cv::Mat lUnDist, rUnDist;
        classical.preprocessFramePair(lFrame, rFrame, lUnDist, rUnDist);

        std::vector<ConeROI> coneROIs;
        detectors.detectFrame(lUnDist, coneROIs, previewArgs);

        std::vector<ConeEst> coneEsts;
        classical.estConePos(lUnDist, rUnDist, coneROIs, coneEsts, frameNum, previewArgs);

        #ifdef PREVIEW
        cv::imshow("Camera_Undist1", lFrameBBox);
        cv::resizeWindow("Camera_Undist1", 1000, 600);
        cv::waitKey(1);

        cv::imshow("Camera_Undist2", rFrameBBox);
        cv::resizeWindow("Camera_Undist2", 1000, 600);
        cv::waitKey(500);

        // cv::imshow("Matches", matchesPreview);
        // cv::resizeWindow("Camera_Undist2", 1000, 600);
        // cv::waitKey(1);
        #endif /* PREVIEW */
        // End Proc Block

        frameNum++;
    }

    return 0;
}