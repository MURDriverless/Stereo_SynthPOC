#include <iostream>
#include <chrono>

#include "StereoBench.hpp"

const uint8_t NUM_TIMEMARKS = 5;

#ifdef STEREOBENCH
struct {
    int64_t timeMarks[NUM_TIMEMARKS];
} benchTimes;

#endif // ifdef STEREOBENCH

void StereoBenchAddTime(SB_TIME_IDX timeIdx) {
#ifndef STEREOBENCH
    return;
#else
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    benchTimes.timeMarks[static_cast<uint8_t>(timeIdx)] = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
#endif // ifndef STEREOBENCH
}

void StereoBenchPrintInfo() {
#ifndef STEREOBENCH
    return
#else
    std::cout << "STEREO_BENCH :: (";
    std::cout << "FRAME_INGEST" << ", ";
    std::cout << "FRAME_UNDIST" << ", ";
    std::cout << "FRAME_DETECT" << ", ";
    std::cout << "FRAME_KEYPT"  << ", ";
    std::cout << "FRAME_SIFT"   ;
    std::cout << ")" << std::endl;
#endif // ifndef STEREOBENCH
}

void StereoBenchPrintTimes() {
#ifndef STEREOBENCH
    return;
#else
    std::cout << "STEREO_BENCH :: (" << benchTimes.timeMarks[0];

    for (int i = 1; i < NUM_TIMEMARKS; i++) {
        std::cout << ", " << benchTimes.timeMarks[i];
    }

    std::cout << ")" << std::endl;
#endif // ifndef STEREOBENCH
}