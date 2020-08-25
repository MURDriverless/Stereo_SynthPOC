#include "ClassicalStereo.hpp"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

ClassicalStereo::CameraParams::CameraParams(const std::string& calibrationFile) {
    cv::FileStorage fs;
    fs.open(calibrationFile, cv::FileStorage::READ);
    fs["cameraMatrix"]   >> cameraMatrix;
    fs["distCoeffs"]     >> distCoeffs;
    fs["calibImageSize"] >> calibSize;

    imgCenter_x = cameraMatrix.at<double>(0, 2);
    imgCenter_y = cameraMatrix.at<double>(1, 2);

    focal_px_x = cameraMatrix.at<double>(0, 0);
    focal_px_y = cameraMatrix.at<double>(1, 1);

    fs.release();

    newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, calibSize, 0.0);
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, cv::Size(1920, 1200), CV_32FC1, xmap, ymap);
    xmap_CUDA.upload(xmap);
    ymap_CUDA.upload(ymap);
}

inline void ClassicalStereo::CameraParams::preprocessFrame(const cv::Mat& frame, cv::Mat& frameOut) {
    cv::cuda::GpuMat frame_CUDA(frame);
    cv::cuda::GpuMat frame_RGB_CUDA, frame_Undist_CUDA;

    // TODO: Fix up color correction
    // cv::cuda::cvtColor(frame_CUDA, frame_RGB_CUDA, cv::CV_BAYer)
    cv::cuda::remap(frame_CUDA, frame_Undist_CUDA, xmap_CUDA, ymap_CUDA, cv::InterpolationFlags::INTER_LINEAR);

    frame_Undist_CUDA.download(frameOut);
}

ClassicalStereo::ClassicalStereo(std::string lCalibrationFile, std::string rCalibrationFile) : 
    lCamParams(lCalibrationFile), rCamParams(rCalibrationFile) {
}

void ClassicalStereo::preprocessFramePair(const cv::Mat& lFrame, const cv::Mat& rFrame, cv::Mat& lFrameOut, cv::Mat& rFrameOut) {
    lCamParams.preprocessFrame(lFrame, lFrameOut);
    rCamParams.preprocessFrame(rFrame, rFrameOut);
}