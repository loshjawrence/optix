#include "CameraFrameManip.h"

#include <spdlog/spdlog.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

#include "CameraFrame.h"

CameraFrameManip::CameraFrameManip(CameraFrame* cameraFrame)
    : cameraFrame(cameraFrame) {
}

/*! this gets called when the user presses a key on the keyboard ... */
void CameraFrameManip::key(int key, int mods) {
    CameraFrame& fc = *cameraFrame;

    switch (key) {
    case '+':
    case '=':
        fc.motionSpeed *= 1.5f;
        spdlog::info("VIEWER: new motion speed is {}", fc.motionSpeed);
        break;
    case '-':
    case '_':
        fc.motionSpeed /= 1.5f;
        spdlog::info("VIEWER: new motion speed is ", fc.motionSpeed);
        break;
    case 'C':
        spdlog::info(
            "(C)urrent camera: - from :{} - poi  :{} - upVec:{} - frame:{}",
            glm::to_string(fc.position),
            glm::to_string(fc.getPOI()),
            glm::to_string(fc.upVector),
            glm::to_string(fc.frame));
        break;
    case 'x':
    case 'X':
        fc.setUpVector(fc.upVector == glm::vec3(1, 0, 0) ? glm::vec3(-1, 0, 0)
                                                         : glm::vec3(1, 0, 0));
        break;
    case 'y':
    case 'Y':
        fc.setUpVector(fc.upVector == glm::vec3(0, 1, 0) ? glm::vec3(0, -1, 0)
                                                         : glm::vec3(0, 1, 0));
        break;
    case 'z':
    case 'Z':
        fc.setUpVector(fc.upVector == glm::vec3(0, 0, 1) ? glm::vec3(0, 0, -1)
                                                         : glm::vec3(0, 0, 1));
        break;
    default: break;
    }
}

void CameraFrameManip::strafe(glm::vec3 howMuch) {
    cameraFrame->position += howMuch;
    cameraFrame->modified = true;
}

/*! strafe, in screen space */
void CameraFrameManip::strafe(glm::vec2 howMuch) {
    strafe(howMuch.x * cameraFrame->frame[0] -
           howMuch.y * cameraFrame->frame[1]);
}

// /*! this gets called when the user presses a key on the keyboard ... */
// void special(int key, const vec2i &where) { };

/*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
void CameraFrameManip::mouseDragLeft(glm::vec2 delta) {
    rotate(delta.x * degrees_per_drag_fraction,
           delta.y * degrees_per_drag_fraction);
}

/*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
void CameraFrameManip::mouseDragMiddle(glm::vec2 delta) {
    strafe(delta * pixels_per_move * cameraFrame->motionSpeed);
}

/*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
void CameraFrameManip::mouseDragRight(glm::vec2 delta) {
    move(delta.y * pixels_per_move * cameraFrame->motionSpeed);
}
