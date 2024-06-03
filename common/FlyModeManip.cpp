#include "FlyModeManip.h"

#include "CameraFrame.h"

#include <glm/gtc/matrix_transform.hpp>

#include <numbers>
using namespace std::numbers;


FlyModeManip::FlyModeManip(CameraFrame* cameraFrame)
    : CameraFrameManip(cameraFrame) {
}

void FlyModeManip::rotate(float deg_u, float deg_v) {
    float rad_u = -pi / 180.f * deg_u;
    float rad_v = -pi / 180.f * deg_v;

    CameraFrame& fc = *cameraFrame;

    fc.frame = glm::mat3(glm::rotate(glm::mat4(fc.frame), rad_u, fc.frame[1]) *
                         glm::rotate(glm::mat4(fc.frame), rad_v, fc.frame[0]) *
                         glm::mat4(fc.frame));

    if (fc.forceUp)
    {
        fc.forceUpFrame();
    }

    fc.modified = true;
}

/*! helper function: move forward/backwards by given multiple of
      motion speed, then make sure the frame, poidistance etc are
      all properly set, the widget gets notified, etc */
void FlyModeManip::move(float step) {
    cameraFrame->position += step * cameraFrame->frame[2];
    cameraFrame->modified = true;
}
