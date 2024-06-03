#include "InspectModeManip.h"

#include "CameraFrame.h"

#include <glm/gtc/matrix_transform.hpp>

#include <numbers>
using namespace std::numbers;

InspectModeManip::InspectModeManip(CameraFrame* cameraFrame)
    : CameraFrameManip(cameraFrame) {
}

void InspectModeManip::rotate(const float deg_u, const float deg_v) {
    float rad_u = -pi / 180.f * deg_u;
    float rad_v = -pi / 180.f * deg_v;

    CameraFrame& fc = *cameraFrame;

    const glm::vec3 poi = fc.getPOI();
    // fc.frame = linear3f::rotate(fc.frame.vy, rad_u) *
    //     linear3f::rotate(fc.frame.vx, rad_v) * fc.frame;
    fc.frame = glm::mat3(glm::rotate(glm::mat4(fc.frame), rad_u, fc.frame[1]) *
                         glm::rotate(glm::mat4(fc.frame), rad_v, fc.frame[0]) *
                         glm::mat4(fc.frame));

    if (fc.forceUp) {
        fc.forceUpFrame();
    }

    fc.position = poi + fc.poiDistance * fc.frame[2];
    fc.modified = true;
}

/*! helper function: move forward/backwards by given multiple of
    motion speed, then make sure the frame, poidistance etc are
    all properly set, the widget gets notified, etc */
void InspectModeManip::move(float step) {
    glm::vec3 poi = cameraFrame->getPOI();
    // inspectmode can't get 'beyond' the look-at point:
    float minReqDistance = 0.1f * cameraFrame->motionSpeed;
    cameraFrame->poiDistance =
        std::fmax(minReqDistance, cameraFrame->poiDistance - step);
    cameraFrame->position =
        poi + cameraFrame->poiDistance * cameraFrame->frame[2];
    cameraFrame->modified = true;
}
