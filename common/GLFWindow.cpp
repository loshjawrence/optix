#include "GLFWindow.h"

#include <assert.h>

#include <spdlog/spdlog.h>

#include <GLFW/glfw3.h>
// #include <glad/glad.h>

static void glfw_error_callback([[maybe_unused]] int error,
                                const char* description) {
    spdlog::error("{}", description);
}

GLFWindow::~GLFWindow() {
    glfwDestroyWindow(handle);
    glfwTerminate();
}

GLFWindow::GLFWindow(const std::string& title) {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) {
        spdlog::error("GLFW init FAILED.");
        glfwTerminate();
        exit(1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    handle = glfwCreateWindow(1200, 800, title.c_str(), NULL, NULL);
    if (!handle) {
        glfwTerminate();
        exit(1);
    }

    glfwSetWindowUserPointer(handle, this);
    glfwMakeContextCurrent(handle);
    glfwSwapInterval(1);
}

static GLFWindow* getGLFWindow(GLFWwindow* window) {
    GLFWindow* gw = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);
    return gw;
}

/*! callback for a window resizing event */
static void glfwindow_reshape_cb(GLFWwindow* window, int width, int height) {
    getGLFWindow(window)->resize({width, height});
}

/*! callback for a key press */
static void glfwindow_key_cb(
    GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        getGLFWindow(window)->key(key, mods);
    }
}

/*! callback for _moving_ the mouse to a new position */
static void glfwindow_mouseMotion_cb(GLFWwindow* window, double x, double y) {
    getGLFWindow(window)->mouseMotion({int(x), int(y)});
}

/*! callback for pressing _or_ releasing a mouse button*/
static void glfwindow_mouseButton_cb(GLFWwindow* window,
                                     int button,
                                     int action,
                                     int mods) {
    getGLFWindow(window)->mouseButton(button, action, mods);
}

void GLFWindow::run() {
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize({width, height});

    glfwSetFramebufferSizeCallback(handle, glfwindow_reshape_cb);
    glfwSetMouseButtonCallback(handle, glfwindow_mouseButton_cb);
    glfwSetKeyCallback(handle, glfwindow_key_cb);
    glfwSetCursorPosCallback(handle, glfwindow_mouseMotion_cb);

    while (!glfwWindowShouldClose(handle)) {
        render();
        draw();
        glfwSwapBuffers(handle);
        glfwPollEvents();
    }
}

glm::ivec2 GLFWindow::getMousePos() const {
    double x, y;
    glfwGetCursorPos(handle, &x, &y);
    return {int(x), int(y)};
}
