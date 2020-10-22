#pragma once
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "ConeROI.hpp"
#include "ConeEst.hpp"
#include "PreviewArgs.hpp"

// #define CONE4 // Use 4 Point cone pnp rather than 7

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
        const double _baseline;

        std::vector<std::vector<cv::Point3f>> conePointsVec; // Real world mm

        cv::Ptr<cv::Feature2D> featureDetector;
        cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;

    public:
        ClassicalStereo(std::string lCalibrationFile, std::string rCalibrationFile, double baseline, cv::Ptr<cv::Feature2D>& featureDetector, cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher);
        void preprocessFramePair(const cv::Mat& lFrame, const cv::Mat& rFrame, cv::Mat& lFrameOut, cv::Mat& rFrameOut);
        void estConePos(const cv::Mat& lFrame, const cv::Mat& rFrame, const std::vector<ConeROI>& coneROIs, std::vector<ConeEst>& coneEsts, int lastFrame = -1, const PreviewArgs& previewArgs = PreviewArgs());
};