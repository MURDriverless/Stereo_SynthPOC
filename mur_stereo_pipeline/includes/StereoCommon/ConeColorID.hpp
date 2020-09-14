#pragma once
#include <iostream>

enum class ConeColorID {
    Blue = 0,
    Orange = 1,
    Yellow = 2
};

inline std::string ConeColorID2str(ConeColorID coneColorID) {
    switch (coneColorID) {
        case (ConeColorID::Blue)    : return "Blue";
        case (ConeColorID::Orange)  : return "Orange";
        case (ConeColorID::Yellow)  : return "Yellow";
    }
}