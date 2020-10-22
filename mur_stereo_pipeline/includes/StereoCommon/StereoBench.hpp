#pragma once

#include <iostream>
#include <stdint.h>

#define STEREOBENCH

enum class SB_TIME_IDX : uint8_t {
    FRAME_INGEST = 0,
    FRAME_UNDIST = 1,
    FRAME_DETECT = 2,
    FRAME_KEYPT  = 3,
    FRAME_SIFT   = 4
};

void StereoBenchAddTime(SB_TIME_IDX timeIdx);
void StereoBenchPrintInfo();
void StereoBenchPrintTimes();