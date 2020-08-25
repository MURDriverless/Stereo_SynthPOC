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
    cv::namedWindow("Cam1_crop", 0);
    cv::namedWindow("Cam2_crop", 0);
    cv::namedWindow("Matches", 0);

    // Prep Producer
    FrameBuffer frameBuffer;
    ProducerArgs producerArgs;
    producerArgs.frameBuffer = &frameBuffer;
    producerArgs.lFilePath = "../track3_L.mp4";
    producerArgs.rFilePath = "../track3_R.mp4";

    std::thread producerL(frameProducer, &producerArgs);

    // Load Calib
    cv::FileStorage fs;
    fs.open("../calibration.xml", cv::FileStorage::READ);

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Size calibSize;

    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    fs["calibImageSize"] >> calibSize;

    fs.release();

    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, calibSize, 0.0);
    cv::Mat map1, map2;
    cv::cuda::GpuMat map1_cuda, map2_cuda;

    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, cv::Size(1920, 1200), CV_32FC1, map1, map2);
    map1_cuda.upload(map1);
    map2_cuda.upload(map2);

    // Prep detectors
    Detectors detectors;
    detectors.initialize("../models/yolo4_cones_int8.rt", "../models/keypoints.onnx");

    // Prep cone model
    std::vector<cv::Point3f> conePoints; // Real world mm
    conePoints.push_back(cv::Point3f(0, 0, 0));

    for (int i = 1; i <= 3; i++) {
        float x = -77.5/3.0f * i;
        float y = 300.0f/3.0f * i;

        conePoints.push_back(cv::Point3f( x, y, 0));
        conePoints.push_back(cv::Point3f(-x, y, 0));
    }

    // Prep feature matching
    cv::Ptr<cv::Feature2D> featureDetector = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    // Actual Stuff
    int lastFrame = -1;
    cv::Mat lFrameCopy, lUnDist;
    cv::Mat rFrameCopy, rUnDist;

    auto then = std::chrono::high_resolution_clock::now();

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

        cv::cuda::GpuMat lFrame_GPU, lUnDist_GPU;
        cv::cuda::GpuMat rFrame_GPU, rUnDist_GPU;

        lFrame_GPU.upload(lFrameCopy);
        rFrame_GPU.upload(rFrameCopy);
        
        cv::cuda::remap(lFrame_GPU, lUnDist_GPU, map1_cuda, map2_cuda, cv::INTER_LINEAR);
        cv::cuda::remap(rFrame_GPU, rUnDist_GPU, map1_cuda, map2_cuda, cv::INTER_LINEAR);

        lUnDist_GPU.download(lUnDist);
        rUnDist_GPU.download(rUnDist);

        cv::Mat lUnDist_clean = lUnDist.clone();
        cv::Mat rUnDist_clean = rUnDist.clone();

        std::vector<ConeROI> coneROIs;
        detectors.detectFrame(lUnDist, coneROIs);

        for (int i = 0; i < coneROIs.size(); i++) {
            ConeROI &coneROI = coneROIs[i];
            cv::Mat tvec;
            cv::Mat rvec;

            double est_depth;

            std::vector<cv::Point3f> cone4 (conePoints.begin()+1, conePoints.end()-2);
            std::vector<cv::Point2f> key4  (coneROI.keypoints.begin()+1, coneROI.keypoints.end()-2);

            bool ret = cv::solvePnP(cone4, key4, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SolvePnPMethod::SOLVEPNP_IPPE);
            if (true) {
                est_depth = tvec.at<double>(2, 0);

                const double f1 = 1449;
                const double f2 = 1449;
                const double B = 200;

                const double width1 = 1920;
                const double width2 = 1920;

                std::cout << "Est Depth: " << est_depth << std::endl;

                float border = 0.0f;
                coneROI.roiRect -= cv::Point2i(border * coneROI.roiRect.width, border * coneROI.roiRect.height);
                coneROI.roiRect += cv::Size2i(2*border * coneROI.roiRect.width, 2*border * coneROI.roiRect.height);

                cv::Rect projRect(coneROI.roiRect);

                int x_p = coneROI.roiRect.x - 1920.0/2.0;
                int x_pp = (f2/est_depth) * (est_depth/f1 * x_p - B);

                projRect.x = x_pp + 1920.0/2.0;

                // projRect.x -= coneROI.roiRect.width * border;
                // projRect.y -= coneROI.roiRect.height * border;
                // projRect.width *= 1.0f + 2*border;
                // projRect.height *= 1.0f + 2*border;

                cv::rectangle(rUnDist, projRect, cv::Scalar(255, 255, 255));

                std::vector<cv::Mat> imgPair;
                imgPair.push_back(lUnDist_clean);
                imgPair.push_back(rUnDist_clean);

                std::vector<cv::Rect> roiMask;
                roiMask.push_back(coneROI.roiRect);
                roiMask.push_back(projRect);

                try {
                    cv::Mat unDist1_cropped = lUnDist_clean(coneROI.roiRect);
                    cv::Mat unDist2_cropped = rUnDist_clean(projRect);

                    // cv::imshow("Cam1_crop", unDist1_cropped);
                    // cv::resizeWindow("Cam1_crop", 1000, 600);

                    // cv::imshow("Cam2_crop", unDist2_cropped);
                    // cv::resizeWindow("Cam2_crop", 1000, 600);
                    // cv::waitKey(1);

                    std::vector<cv::KeyPoint> featureKeypoints1;
                    std::vector<cv::KeyPoint> featureKeypoints2;
                    cv::Mat descriptors1;
                    cv::Mat descriptors2;

                    std::vector<cv::DMatch> matches;
                    std::vector<cv::DMatch> matchesFilt;

                    featureDetector->detectAndCompute(unDist1_cropped, cv::noArray(), featureKeypoints1, descriptors1);
                    featureDetector->detectAndCompute(unDist2_cropped, cv::noArray(), featureKeypoints2, descriptors2);

                    descriptorMatcher->match(descriptors1, descriptors2, matches);

                    uint32_t yDelta = projRect.height * 0.1;

                    for (const cv::DMatch &match : matches) {
                        if (abs(featureKeypoints1[match.queryIdx].pt.y - featureKeypoints2[match.trainIdx].pt.y) < yDelta) {
                            matchesFilt.push_back(match);
                        }
                    }

                    std::vector<float> disparity;
                    for (const cv::DMatch &match : matchesFilt) {
                        float x1 = featureKeypoints1[match.queryIdx].pt.x;
                        x1 += coneROI.roiRect.x;
                        x1 -= width1/2;

                        float x2 = featureKeypoints2[match.trainIdx].pt.x;
                        x2 += projRect.x;
                        x2 -= width2/2;

                        disparity.push_back(x1*f2/f1 - x2);
                    }

                    std::sort(disparity.begin(), disparity.end());

                    if (disparity.size() < 1) {
                        continue;
                    }

                    float medDisp = disparity[(int) disparity.size()/2];
                    float zEst = B*f2/medDisp;
                    float xEst = zEst*(coneROI.roiRect.x + coneROI.roiRect.width/2 - width1/2)/f1;
                    float yEst = zEst*(coneROI.roiRect.y + coneROI.roiRect.height - 1200/2)/f1;

                    std::cout << "Refined Pos (t, x, y, z): (" << lastFrame << ", " << xEst << ", "
                    << yEst << ", " <<  zEst << ")" << std::endl;

                    // std::cout << "Median Disp: " << disparity[(int) disparity.size()/2] << std::endl; 

                    cv::Mat imgMatch;
                    cv::drawMatches(unDist1_cropped, featureKeypoints1, unDist2_cropped, featureKeypoints2, matchesFilt, imgMatch);
                    // cv::drawKeypoints(unDist1_cropped, featureKeypoints1, imgMatch);

                    // cv::imshow("Matches", imgMatch);
                    // cv::resizeWindow("Matches", 1000, 600);
                    // cv::waitKey(1);
                }
                catch (const std::exception &e) {
                    // Error handling.
                    std::cerr << "An exception occurred." << std::endl
                    << e.what() << std::endl;
                }
                
            }
        }

        cv::imshow("Camera_Undist2", rUnDist);
        cv::resizeWindow("Camera_Undist2", 1000, 600);
        cv::waitKey(1);

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