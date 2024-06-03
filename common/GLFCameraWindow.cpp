#include "GLFCameraWindow.h"

#include <spdlog/spdlog.h>

#include "CameraFrameManip.h"
#include "FlyModeManip.h"
#include "InspectModeManip.h"

#include <GLFW/glfw3.h>

GLFCameraWindow::GLFCameraWindow(const std::string& title,
                                 glm::vec3 camera_from,
                                 glm::vec3 camera_at,
                                 glm::vec3 camera_up,
                                 float worldScale)
    : GLFWindow(title)
    , cameraFrame(worldScale) {
    cameraFrame.setOrientation(camera_from, camera_at, camera_up);
    enableFlyMode();
    enableInspectMode();
}

void GLFCameraWindow::key(int key, int mods) {
    switch (key) {
    case 'f':
    case 'F':
        spdlog::info("Entering 'fly' mode");
        if (flyModeManip)
        {
            cameraFrameManip = flyModeManip;
        }
        break;
    case 'i':
    case 'I':
        spdlog::info("Entering 'inspect' mode");
        if (inspectModeManip) {
            cameraFrameManip = inspectModeManip;
        }
        break;
    default:
        if (cameraFrameManip)
        {
            cameraFrameManip->key(key, mods);
        }
    }
}

/*! callback that window got resized */
void GLFCameraWindow::mouseMotion(glm::ivec2 newPos) {
    glm::ivec2 windowSize;
    glfwGetWindowSize(handle, &windowSize.x, &windowSize.y);

    if (isPressed.leftButton && cameraFrameManip)
        cameraFrameManip->mouseDragLeft(glm::vec2(newPos - lastMousePos) /
                                        glm::vec2(windowSize));
    if (isPressed.rightButton && cameraFrameManip)
        cameraFrameManip->mouseDragRight(glm::vec2(newPos - lastMousePos) /
                                         glm::vec2(windowSize));
    if (isPressed.middleButton && cameraFrameManip)
        cameraFrameManip->mouseDragMiddle(glm::vec2(newPos - lastMousePos) /
                                          glm::vec2(windowSize));
    lastMousePos = newPos;
    /* empty - to be subclassed by user */
}

/*! callback that window got resized */
void GLFCameraWindow::mouseButton(int button, int action, int mods) {
    const bool pressed = (action == GLFW_PRESS);
    switch (button) {
    case GLFW_MOUSE_BUTTON_LEFT: isPressed.leftButton = pressed; break;
    case GLFW_MOUSE_BUTTON_MIDDLE: isPressed.middleButton = pressed; break;
    case GLFW_MOUSE_BUTTON_RIGHT: isPressed.rightButton = pressed; break;
    }
    lastMousePos = getMousePos();
}

void GLFCameraWindow::enableFlyMode() {
    flyModeManip = std::make_shared<FlyModeManip>(&cameraFrame);
    cameraFrameManip = flyModeManip;
}

void GLFCameraWindow::enableInspectMode() {
    inspectModeManip = std::make_shared<InspectModeManip>(&cameraFrame);
    cameraFrameManip = inspectModeManip;
}
