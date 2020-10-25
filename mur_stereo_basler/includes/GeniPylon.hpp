#pragma once
#include "GeniWrap.hpp"

#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h>
#include <pylon/ParameterIncludes.h>
#include <pylon/EnumParameterT.h>
#include <GenApi/GenApi.h>

class PylonCam: public IGeniCam {
    private:
        Pylon::CBaslerUniversalInstantCamera camera;
        Pylon::CGrabResultPtr ptrGrabResult;

    public:
        PylonCam() {

        }
        
        void initializeLibrary() {
            Pylon::PylonInitialize();
        }

        void finalizeLibrary() {
            Pylon::PylonTerminate();
        }

        void setup(const std::string cameraName) {
            Pylon::CTlFactory& TlFactory = Pylon::CTlFactory::GetInstance();
            Pylon::CDeviceInfo di;
            di.SetFriendlyName(cameraName.c_str());
            camera.Attach(TlFactory.CreateDevice(di));

            // Print the model name of the camera.
            std::cout << "Using device " << camera.GetDeviceInfo().GetModelName() << std::endl;

            // The parameter MaxNumBuffer can be used to control the count of buffers
            // allocated for grabbing. The default value of this parameter is 10.
            // camera.MaxNumBuffer = 1;

            camera.Open();
            GenApi::INodeMap& nodemap = camera.GetNodeMap();
            // Pylon::CEnumParameter(nodemap, "PixelFormat").Se'tValue("BayerBG8");
            Pylon::CBooleanParameter(nodemap, "AcquisitionFrameRateEnable").SetValue(false);
            // camera.LightSourcePreset.SetValue(Basler_UniversalCameraParams::LightSourcePresetEnums::LightSourcePreset_Daylight6500K);

            // // Select auto function ROI 2
            // Pylon::CEnumParameter(nodemap, "AutoFunctionROISelector").SetValue("ROI2");
            // // Enable the Balance White Auto auto function
            // // for the auto function ROI selected
            // Pylon::CBooleanParameter(nodemap, "AutoFunctionROIUseWhiteBalance").SetValue(true);
            // // Enable Balance White Auto by setting the operating mode to Continuous
            // Pylon::CEnumParameter(nodemap, "BalanceWhiteAuto").SetValue("Continuous");
            camera.Close();
        }

        void startGrabbing(uint32_t numImages = UINT32_MAX) {
            // Start the grabbing of c_countOfImagesToGrab images.
            // The camera device is parameterized with a default configuration which
            // sets up free-running continuous acquisition.
            camera.StartGrabbing(numImages, Pylon::GrabStrategy_LatestImageOnly);
        }

        bool isGrabbing() {
            return camera.IsGrabbing();
        }

        bool retreiveResult(int &height, int &width, uint8_t* &buffer) {
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            camera.RetrieveResult( 5000, ptrGrabResult, Pylon::TimeoutHandling_ThrowException);

            height = ptrGrabResult->GetHeight();
            width  = ptrGrabResult->GetWidth();
            buffer = static_cast<uint8_t *>(ptrGrabResult->GetBuffer());

            return ptrGrabResult->GrabSucceeded();
        }

        void clearResult() {
            return;
        }
};