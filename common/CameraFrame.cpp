#include "CameraFrame.h"

CameraFrame::CameraFrame(float worldScale)
    : motionSpeed(worldScale) {
}

glm::vec3 CameraFrame::getPOI() const {
    return position - poiDistance * frame[2];
}

void CameraFrame::setOrientation(
    /* camera origin    : */ glm::vec3 origin,
    /* point of interest: */ glm::vec3 poi,
    /* up-vector        : */ glm::vec3 up) {
    /*! re-compute all orientation related fields from given 'user-style' camera parameters */
    position = origin;
    upVector = up;
    frame[2] = (poi == origin)
        ? glm::vec3(0, 0, 1)
        : /* negative because we use NEGATIZE z axis */ -normalize(poi -
                                                                   origin);
    frame[0] = cross(up, frame[2]);
    if (dot(frame[0], frame[0]) < 1e-8f)
        frame[0] = glm::vec3(0, 1, 0);
    else
        frame[0] = normalize(frame[0]);
    // frame[0]
    //   = (fabs(dot(up,frame[2])) < 1e-6f)
    //   ? glm::vec3(0,1,0)
    //   : normalize(cross(up,frame[2]));
    frame[1] = normalize(cross(frame[2], frame[0]));
    poiDistance = glm::length(poi - origin);
    forceUpFrame();
}

void CameraFrame::forceUpFrame() {
    /*! tilt the frame around the z axis such that the y axis is "facing upwards" */
    // frame[2] remains unchanged
    if (fabsf(dot(frame[2], upVector)) < 1e-6f)
        // looking along upvector; not much we can do here ...
        return;
    frame[0] = normalize(cross(upVector, frame[2]));
    frame[1] = normalize(cross(frame[2], frame[0]));
    modified = true;
}

void CameraFrame::setUpVector(glm::vec3 up) {
    upVector = up;
    forceUpFrame();
}

float CameraFrame::computeStableEpsilon(float f) const {
    return std::fabs(f) * float(1. / (1 << 21));
}

float CameraFrame::computeStableEpsilon(const glm::vec3 v) const {
    return std::fmax(
        std::fmax(computeStableEpsilon(v.x), computeStableEpsilon(v.y)),
        computeStableEpsilon(v.z));
}

glm::vec3 CameraFrame::get_from() const {
    return position;
}

glm::vec3 CameraFrame::get_at() const {
    return getPOI();
}

glm::vec3 CameraFrame::get_up() const {
    return upVector;
}
