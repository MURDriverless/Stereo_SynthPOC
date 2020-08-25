# Stereo_SynthPOC
Currently used as a playground for testing and benchmarking of the stereo pipeline, as well as clean up of code.

**Will be removed in the future once integrated with original repo**

## Requirements/Compiled with
 - `CUDA` 10.0
 - `CUDNN` 7.6
 - `OpenCV` 4.1.1, Compiled with CUDA, CUDNN and Non-free addons
 - `tkDNN` 0.5

### Installing `tkDNN`
```
git clone https://github.com/ceccocats/tkDNN.git
cd tkDNN
git checkout v0.5
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig
```

## Usage
 - Currently left and right video streams are hard coded, need to modify `src/main.cpp` to change the video stream targets
 - Place models in `models/` for `keypoints.onnx` and `yolo4_cones_int8.rt`