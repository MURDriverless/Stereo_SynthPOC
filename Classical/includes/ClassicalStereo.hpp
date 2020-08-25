#pragma once
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

class ClassicalStereo {
    private:
        class CameraParams {
            private:
                CameraParams() {};
            public:
                cv::Mat cameraMatrix;
                cv::Mat distCoeffs;
                cv::Size calibSize;

                double imgCenter_x;
                double imgCenter_y;
                double focal_px_x;
                double focal_px_y;

                cv::Mat newCameraMatrix;
                cv::Mat xmap, ymap;
                cv::cuda::GpuMat xmap_CUDA, ymap_CUDA;

                CameraParams(const std::string& calibrationFile);
                void preprocessFrame(const cv::Mat& frame, cv::Mat& frameOut);
        };

        ClassicalStereo::CameraParams lCamParams;
        ClassicalStereo::CameraParams rCamParams;


    public:
        ClassicalStereo(std::string lCalibrationFile, std::string rCalibrationFile);
        void preprocessFramePair(const cv::Mat& lFrame, const cv::Mat& rFrame, cv::Mat& lFrameOut, cv::Mat& rFrameOut);
};