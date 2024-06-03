#pragma once

#include <glm/glm.hpp>

struct CameraFrame;

// ------------------------------------------------------------------
/*! abstract base class that allows to manipulate a renderable
    camera */
struct CameraFrameManip {
    CameraFrameManip(CameraFrame* cameraFrame);

    /*! this gets called when the user presses a key on the keyboard ... */
    virtual void key(int key, int mods);

    virtual void strafe(glm::vec3 howMuch);

    /*! strafe, in screen space */
    virtual void strafe(glm::vec2 howMuch);

    virtual void move(float step) = 0;
    virtual void rotate(float dx, float dy) = 0;

    // /*! this gets called when the user presses a key on the keyboard ... */
    // virtual void special(int key, const vec2i &where) { };

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragLeft(glm::vec2 delta);

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragMiddle(glm::vec2 delta);

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragRight(glm::vec2 delta);

protected:
    CameraFrame* cameraFrame{};
    float kbd_rotate_degrees{10.f};
    float degrees_per_drag_fraction{150.f};
    float pixels_per_move{10.f};
};

