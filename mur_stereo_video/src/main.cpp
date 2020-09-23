#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>

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

#define PREVIEW

struct FrameBuffer {
    std::mutex mutexLock;
    cv::Mat lFrame;
    cv::Mat rFrame;

    int frameNum = -1;
    bool eof = false;
};

struct ProducerArgs {
    FrameBuffer* frameBuffer;
    std::string lFilePath;
    std::string rFilePath;
};

void frameProducer(ProducerArgs *producerArgs);

int main(int argc, char** argv) {
    // Prep Window
    cv::namedWindow("Camera_Undist1", 0);
    cv::namedWindow("Camera_Undist2", 0);
    cv::namedWindow("Matches", 0);

    // Prep Producer
    FrameBuffer frameBuffer;
    ProducerArgs producerArgs;
    producerArgs.frameBuffer = &frameBuffer;
    producerArgs.lFilePath = "../track3_L.mp4";
    producerArgs.rFilePath = "../track3_R.mp4";

    std::thread producerL(frameProducer, &producerArgs);

    const double baseline = 200.00;
    cv::Ptr<cv::Feature2D> featureDetector = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    ClassicalStereo classical("../calibration.xml", "../calibration.xml", baseline, featureDetector, descriptorMatcher);

    // Prep detectors
    Detectors detectors;
    detectors.initialize("../models/yolo4_cones_int8.rt", "../models/keypoints.onnx");

    // Actual Stuff
    int lastFrame = -1;
    cv::Mat lFrameCopy, lUnDist;
    cv::Mat rFrameCopy, rUnDist;

    auto then = std::chrono::high_resolution_clock::now();

    #ifdef PREVIEW
    cv::Mat rFrameBBox;
    cv::Mat lFrameBBox;
    cv::Mat matchesPreview;
    PreviewArgs previewArgs(lFrameBBox, rFrameBBox, matchesPreview);
    #else
    PreviewArgs previewArgs = PreviewArgs();
    #endif /* PREVIEW */

    while (true) {

        auto now = std::chrono::high_resolution_clock::now();
        double frameTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
        then = now;

        std::cout << "FPS: " << 1000.0/frameTime << std::endl;

        bool valid = false;
        frameBuffer.mutexLock.lock();
        if (lastFrame < frameBuffer.frameNum) {
            lFrameCopy = frameBuffer.lFrame.clone();
            rFrameCopy = frameBuffer.rFrame.clone();
            lastFrame = frameBuffer.frameNum;
            valid = true;
        }
        frameBuffer.mutexLock.unlock();

        if (frameBuffer.eof) {
            break;
        }

        if (lFrameCopy.empty() || rFrameCopy.empty() || !valid) {
            continue;
        }

        cv::Mat lUnDist, rUnDist;
        classical.preprocessFramePair(lFrameCopy, rFrameCopy, lUnDist, rUnDist);

        std::vector<ConeROI> coneROIs;
        detectors.detectFrame(lUnDist, coneROIs, previewArgs);

        std::vector<ConeEst> coneEsts;

        classical.estConePos(lUnDist, rUnDist, coneROIs, coneEsts, lastFrame, previewArgs);

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

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    cv::destroyAllWindows();
    producerL.join();

    return 0;
}

void frameProducer(ProducerArgs *producerArgs) {
    FrameBuffer *frameBuffer = producerArgs->frameBuffer;
    cv::VideoCapture lVideoFile(producerArgs->lFilePath);
    cv::VideoCapture rVideoFile(producerArgs->rFilePath);

    auto frameTime = std::chrono::milliseconds(static_cast<int64_t>(1000.0/lVideoFile.get(cv::CAP_PROP_FPS)));
    // auto frameTime = std::chrono::milliseconds(100);

    auto startTime = std::chrono::high_resolution_clock::now();

    std::cout << frameTime.count() << std::endl;

    while (true) {
        auto now = std::chrono::high_resolution_clock::now();

        frameBuffer->mutexLock.lock();
        lVideoFile.read(frameBuffer->lFrame);
        rVideoFile.read(frameBuffer->rFrame);
        frameBuffer->frameNum++;
        frameBuffer->mutexLock.unlock();

        if (frameBuffer->lFrame.empty() || frameBuffer->rFrame.empty()) {
            frameBuffer->eof = true;
            return;
        }

        auto now2 = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(frameTime - (now2 - now));
    }
}