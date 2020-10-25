#include <iostream>

#include <opencv2/core.hpp>

#include "Detectors.hpp"
#include "ClassicalStereo.hpp"
#include "PreviewArgs.hpp"

#include "GeniWrap.hpp"

#define PREVIEW

#ifndef SRC_ROOT_PATH
#define SRC_ROOT_PATH "./"
#endif

const std::string CAMARA_NAME_L = "CameraLeft (40068492)";
const std::string CAMARA_NAME_R = "CameraLeft (40068492)";

unsigned int grabCount = 6000;

int main(int argc, char** argv) {
    #ifdef PREVIEW
    cv::Mat rFrameBBox;
    cv::Mat lFrameBBox;
    cv::Mat matchesPreview;
    PreviewArgs previewArgs(lFrameBBox, rFrameBBox, matchesPreview);
    #else
    PreviewArgs previewArgs = PreviewArgs();
    #endif /* PREVIEW */

    // Prep Classical
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

    // Prep real cameras
    std::unique_ptr<IGeniCam> camera1;
    std::unique_ptr<IGeniCam> camera2;

    camera1.reset(IGeniCam::create(GeniImpl::Pylon_i));
    camera2.reset(IGeniCam::create(GeniImpl::Pylon_i));
    
    camera1->initializeLibrary();
    
    int exitCode = 0;
    try {
        camera1->setup(CAMARA_NAME_L);
        // camera2->setup(CAMARA_NAME_R);

        camera1->startGrabbing(grabCount);
        unsigned long imageCount = 0;

        while (imageCount < grabCount && camera1->isGrabbing()) {
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            int height1;
            int width1;
            uint8_t* buffer1;

            int height2;
            int width2;
            uint8_t* buffer2;

            bool ret1 = camera1->retreiveResult(height1, width1, buffer1);
            // bool ret2 = camera2->retreiveResult(height2, width2, buffer2);

            if (ret1) {
                cv::Mat inMat = cv::Mat(height1, width1, CV_8UC1, buffer1);
                cv::imshow("ehh", inMat);
                cv::waitKey(1);
            }

            imageCount++;
        }
    }
    catch (const std::exception &e)
    {
        // Error handling.
        std::cerr << "An exception occurred." << std::endl
        << e.what() << std::endl;
        exitCode = 1;
    }

    camera1->finalizeLibrary();
    delete camera1.release();
    delete camera2.release();

    return exitCode;
}