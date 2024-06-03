#pragma once

#include "CameraFrame.h"
#include "GLFWindow.h"

#include <memory>

struct GLFCameraWindow : public GLFWindow {
    GLFCameraWindow(const std::string& title,
                    glm::vec3 camera_from,
                    glm::vec3 camera_at,
                    glm::vec3 camera_up,
                    float worldScale);

    void enableFlyMode();
    void enableInspectMode();

    // /*! put pixels on the screen ... */
    // virtual void draw()
    // { /* empty - to be subclassed by user */ }

    // /*! callback that window got resized */
    // virtual void resize(glm::ivec2 newSize)
    // { /* empty - to be subclassed by user */ }

    virtual void key(int key, int mods) override;

    /*! callback that window got resized */
    virtual void mouseMotion(glm::ivec2 newPos) override;

    /*! callback that window got resized */
    virtual void mouseButton(int button, int action, int mods) override;

    // /*! mouse got dragged with left button pressedn, by 'delta'
    //   pixels, at last position where */
    // virtual void mouseDragLeft  (glm::ivec2 where, glm::ivec2 delta) {}

    // /*! mouse got dragged with left button pressedn, by 'delta'
    //   pixels, at last position where */
    // virtual void mouseDragRight (glm::ivec2 where, glm::ivec2 delta) {}

    // /*! mouse got dragged with left button pressedn, by 'delta'
    //   pixels, at last position where */
    // virtual void mouseDragMiddle(glm::ivec2 where, glm::ivec2 delta) {}

    /*! a (global) pointer to the currently active window, so we can
      route glfw callbacks to the right GLFWindow instance (in this
      simplified library we only allow on window at any time) */
    // static GLFWindow *current;

    struct {
        bool leftButton{false}, middleButton{false}, rightButton{false};
    } isPressed;
    glm::ivec2 lastMousePos = {-1, -1};

    friend struct CameraFrameManip;

    CameraFrame cameraFrame;
    std::shared_ptr<CameraFrameManip> cameraFrameManip;
    std::shared_ptr<CameraFrameManip> inspectModeManip;
    std::shared_ptr<CameraFrameManip> flyModeManip;
};
