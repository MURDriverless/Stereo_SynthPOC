#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

struct FrameBuffer {
    std::mutex mutexLock;
    cv::Mat frame;
    int frameNum = -1;
    bool eof = false;
};

struct ProducerArgs {
    FrameBuffer* frameBuffer;
    std::string filePath;
};

void frameProducer(ProducerArgs *producerArgs);

int main(int argc, char** argv) {
    FrameBuffer lBuffer;
    ProducerArgs leftArgs;
    leftArgs.frameBuffer = &lBuffer;
    leftArgs.filePath = "../output_R.mp4";

    std::thread producerL(frameProducer, &leftArgs);

    cv::FileStorage fs;
    fs.open("../calibration.xml", cv::FileStorage::READ);

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Size calibSize;

    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    fs["calibImageSize"] >> calibSize;

    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, calibSize, 0.0);

    int lastFrame = -1;
    cv::Mat frameCopy, unDist;
    cv::namedWindow("Frame", 0);
    while (true) {
        bool valid = false;
        lBuffer.mutexLock.lock();
        if (lastFrame < lBuffer.frameNum) {
            frameCopy = lBuffer.frame.clone();
            lastFrame = lBuffer.frameNum;
            valid = true;
        }
        lBuffer.mutexLock.unlock();

        if (lBuffer.eof) {
            break;
        }

        if (frameCopy.empty() || !valid) {
            continue;
        }

        cv::undistort(frameCopy, unDist, cameraMatrix, distCoeffs, newCameraMatrix);

        cv::imshow("Frame", unDist);
        cv::resizeWindow("Frame", 1000, 600);
        cv::waitKey(1);

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    cv::destroyAllWindows();
    producerL.join();

    return 0;
}

void frameProducer(ProducerArgs *producerArgs) {
    FrameBuffer *frameBuffer = producerArgs->frameBuffer;
    cv::VideoCapture videoFile(producerArgs->filePath);
    auto frameTime = std::chrono::milliseconds(static_cast<int64_t>(1000.0/videoFile.get(cv::CAP_PROP_FPS)));
    auto startTime = std::chrono::high_resolution_clock::now();

    std::cout << frameTime.count() << std::endl;

    while (true) {
        auto now = std::chrono::high_resolution_clock::now();

        frameBuffer->mutexLock.lock();
        videoFile.read(frameBuffer->frame);
        frameBuffer->frameNum++;
        frameBuffer->mutexLock.unlock();

        if (frameBuffer->frame.empty()) {
            frameBuffer->eof = true;
            return;
        }

        auto now2 = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(frameTime - (now2 - now));
    }
}