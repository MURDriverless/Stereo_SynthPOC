#include <iostream>
#include <mutex>
#include <pthread.h>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

class StereoCam {
    private:
        cv::VideoCapture fileL;
        cv::VideoCapture fileR;
        
        int frameRate;
        int lastFrame = -1;
        std::chrono::system_clock::time_point startTime;
    public:
        StereoCam(std::string pathL, std::string pathR) {
            fileL = cv::VideoCapture(pathL);
            fileR = cv::VideoCapture(pathR);

            frameRate = fileL.get(cv::CAP_PROP_FPS);
            startTime = std::chrono::high_resolution_clock::now();
        }

        bool grabFrame(cv::Mat &leftFrame, cv::Mat &rightFrame) {
            auto now = std::chrono::high_resolution_clock::now();
            int nowFrame = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count() * frameRate / 1000;

            if (nowFrame <= lastFrame) {
                return false;
            }

            if (lastFrame + 1 != nowFrame) {
                // fileL.set(cv::CAP_PROP_POS_FRAMES, nowFrame);
                // fileR.set(cv::CAP_PROP_POS_FRAMES, nowFrame);
            }


            fileL.read(leftFrame);
            fileR.read(rightFrame);

            // lastFrame = nowFrame;
            return true;
        }
};

int main(int argc, char** argv) {
    StereoCam stereoCam("../output_L.mp4", "../output_R.mp4");

    cv::Mat lFrame, rFrame;

    auto now = std::chrono::high_resolution_clock::now();
    auto then = std::chrono::high_resolution_clock::now();
    cv::namedWindow("Frame", 0);
    while (true) {
        if (stereoCam.grabFrame(lFrame, rFrame)) {
            cv::imshow("Frame", lFrame);
            cv::resizeWindow("Frame", 600, 300);
            cv::waitKey(1);
        }
        auto now = std::chrono::high_resolution_clock::now();
        float deltaT = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
        std::cout << 1000.0 / (deltaT) << std::endl;
        then = now;
    }

    return 0;
}