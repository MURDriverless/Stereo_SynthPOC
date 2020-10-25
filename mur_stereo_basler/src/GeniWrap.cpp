#include "GeniWrap.hpp"

#include "GeniPylon.hpp"

IGeniCam* IGeniCam::create(GeniImpl geniImpl) {
    switch(geniImpl) {
        case GeniImpl::Pylon_i :
            return new PylonCam();
    }

    return nullptr;
}