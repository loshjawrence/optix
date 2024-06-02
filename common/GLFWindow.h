#pragma once

#include <glm/glm.hpp>
#include <string>

struct GLFWwindow;
struct GLFWmonitor;

struct GLFWindow {
    GLFWindow(const std::string& title);
    ~GLFWindow();

    /*! put pixels on the screen ... */
    virtual void draw() { /* empty - to be subclassed by user */
    }

    /*! callback that window got resized */
    virtual void resize(
        glm::ivec2 newSize) { /* empty - to be subclassed by user */
    }

    virtual void key(int key, int mods) {
    }

    /*! callback that window got resized */
    virtual void mouseMotion(glm::ivec2 newPos) {
    }

    /*! callback that window got resized */
    virtual void mouseButton(int button, int action, int mods) {
    }

    glm::ivec2 getMousePos() const;

    /*! re-render the frame - typically part of draw(), but we keep
      this a separate function so render() can focus on optix
      rendering, and now have to deal with opengl pixel copies
      etc */
    virtual void render() { /* empty - to be subclassed by user */
    }

    /*! opens the actual window, and runs the window's events to
      completion. This function will only return once the window
      gets closed */
    void run();

    /*! the glfw window handle */
    GLFWwindow* handle{};
};
