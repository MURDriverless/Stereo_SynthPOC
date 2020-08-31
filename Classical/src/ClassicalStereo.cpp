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

ClassicalStereo::ClassicalStereo(std::string lCalibrationFile, std::string rCalibrationFile, double baseline, cv::Ptr<cv::Feature2D>& featureDetector, cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher) : 
    lCamParams(lCalibrationFile), rCamParams(rCalibrationFile), _baseline(baseline) {

    conePoints.push_back(cv::Point3f(0, 0, 0));
    for (int i = 1; i <= 3; i++) {
        float x = -77.5/3.0f * i;
        float y = 300.0f/3.0f * i;

        conePoints.push_back(cv::Point3f( x, y, 0));
        conePoints.push_back(cv::Point3f(-x, y, 0));
    }

    this->featureDetector   = featureDetector;
    this->descriptorMatcher = descriptorMatcher;
}

void ClassicalStereo::preprocessFramePair(const cv::Mat& lFrame, const cv::Mat& rFrame, cv::Mat& lFrameOut, cv::Mat& rFrameOut) {
    lCamParams.preprocessFrame(lFrame, lFrameOut);
    rCamParams.preprocessFrame(rFrame, rFrameOut);
}

void ClassicalStereo::estConePos(const cv::Mat& lFrame, const cv::Mat& rFrame, const std::vector<ConeROI>& coneROIs, std::vector<ConeEst> coneEsts, int lastFrame, const PreviewArgs& previewArgs) {
    if (previewArgs.valid) {
        *(previewArgs.rFrameBBoxMatPtr) = rFrame.clone();
    }

    for (int i = 0; i < coneROIs.size(); i++) {
        const ConeROI& coneROI = coneROIs[i];
        cv::Mat tvec;
        cv::Mat rvec;

        double est_depth;

        #ifdef CONE4
            std::vector<cv::Point3f> conePts (conePoints.begin()+1, conePoints.end()-2);
            std::vector<cv::Point2f> keyPts  (coneROI.keypoints.begin()+1, coneROI.keypoints.end()-2);
        #else
            std::vector<cv::Point3f> &conePts = conePoints;
            std::vector<cv::Point2f> &keyPts  = coneROI.keypoints;
        #endif

        // TODO: Need to process the return
        bool ret = cv::solvePnP(conePts, keyPts, lCamParams.cameraMatrix, lCamParams.distCoeffs, rvec, tvec, false, cv::SolvePnPMethod::SOLVEPNP_IPPE);
        if (true) {
            est_depth = tvec.at<double>(2, 0);

            // Using reference shouldnt cause performance degredation?
            const double &f1 = lCamParams.focal_px_x;
            const double &f2 = rCamParams.focal_px_x;

            const double &lImgCenter_x = lCamParams.imgCenter_x;
            const double &rImgCenter_x = rCamParams.imgCenter_x;

            const double &rImgCenter_y = rCamParams.imgCenter_y;

            // TODO? : Enlarge borders around cones?
            // float border = 0.0f;
            // coneROI.roiRect -= cv::Point2i(border * coneROI.roiRect.width, border * coneROI.roiRect.height);
            // coneROI.roiRect += cv::Size2i(2*border * coneROI.roiRect.width, 2*border * coneROI.roiRect.height);

            cv::Rect projRect(coneROI.roiRect);

            int x_p = coneROI.roiRect.x - lImgCenter_x;
            int x_pp = (f2/est_depth) * (est_depth/f1 * x_p - _baseline);

            projRect.x = x_pp + rImgCenter_x;

            // projRect.x -= coneROI.roiRect.width * border;
            // projRect.y -= coneROI.roiRect.height * border;
            // projRect.width *= 1.0f + 2*border;
            // projRect.height *= 1.0f + 2*border;

            // Bounds checking
            // Should we implement reshaping if out of bounds?
            if (!(0 <= projRect.x && projRect.x + projRect.width < rFrame.cols)) {
                continue;
            }

            cv::Mat unDist1_cropped = lFrame(coneROI.roiRect);
            cv::Mat unDist2_cropped = rFrame(projRect);

            std::vector<cv::KeyPoint> featureKeypoints1;
            std::vector<cv::KeyPoint> featureKeypoints2;
            cv::Mat descriptors1;
            cv::Mat descriptors2;

            std::vector<cv::DMatch> matches;
            std::vector<cv::DMatch> matchesFilt;

            featureDetector->detectAndCompute(unDist1_cropped, cv::noArray(), featureKeypoints1, descriptors1);
            featureDetector->detectAndCompute(unDist2_cropped, cv::noArray(), featureKeypoints2, descriptors2);

            /*
            No descriptor in left or right frame, either due to insufficient light.
            Or plain ground textures, expecially in synthetic data
            */
            if (descriptors1.empty() || descriptors2.empty()) {
                continue;
            }

            descriptorMatcher->match(descriptors1, descriptors2, matches);

            // Filters for horizontal-ish matches only
            uint32_t yDelta = projRect.height * 0.1;
            for (const cv::DMatch &match : matches) {
                if (abs(featureKeypoints1[match.queryIdx].pt.y - featureKeypoints2[match.trainIdx].pt.y) < yDelta) {
                    matchesFilt.push_back(match);
                }
            }

            // Check if no valid matches
            if (matchesFilt.size() == 0) {
                continue;
            }

            std::vector<float> disparity;
            for (const cv::DMatch &match : matchesFilt) {
                float x1 = featureKeypoints1[match.queryIdx].pt.x;
                x1 += coneROI.roiRect.x;
                x1 -= lImgCenter_x;

                float x2 = featureKeypoints2[match.trainIdx].pt.x;
                x2 += projRect.x;
                x2 -= rImgCenter_x;

                disparity.push_back(x1*f2/f1 - x2);
            }

            // Performance loss from not using a sorted heap should be negligable
            std::sort(disparity.begin(), disparity.end());

            float medDisp = disparity[(int) disparity.size()/2];
            float zEst = _baseline*f2/medDisp;
            float xEst = zEst*(coneROI.roiRect.x + coneROI.roiRect.width/2 - rImgCenter_x)/f1;
            float yEst = zEst*(coneROI.roiRect.y + coneROI.roiRect.height - rImgCenter_y)/f1;

            ConeEst coneEst;
            coneEst.pos.x = xEst;
            coneEst.pos.y = yEst;
            coneEst.pos.z = zEst;

            coneEsts.push_back(coneEst);

            if (previewArgs.valid) {
                cv::rectangle(*(previewArgs.rFrameBBoxMatPtr), projRect, cv::Scalar(255, 255, 255));

                if (i == 0) {
                    // Draw matches here.
                }
            }

            if (lastFrame >= 0) {
                std::cout << "Est Depth: " << est_depth << std::endl;
                std::cout << "Refined Pos (t, x, y, z): (" << lastFrame << ", " << xEst << ", "
                << yEst << ", " <<  zEst << ")" << std::endl;
            }
        }
    }
}