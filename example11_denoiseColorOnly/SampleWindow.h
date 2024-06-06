#pragma once

#include "GLFCameraWindow.h"
#include "SampleRenderer.h"
#include <vector>

#include <glm/glm.hpp>
#include <glad/glad.h>

struct Model;
struct Camera;
struct light;

struct SampleWindow : public GLFCameraWindow {
    SampleWindow(const std::string& title,
                 const Model* model,
                 const Camera& camera,
                 const QuadLight& light,
                 const float worldScale);

    virtual void render() override;
    virtual void draw() override;
    virtual void resize(glm::ivec2 newSize) override;
    virtual void key(int key, int mods) override;

    glm::ivec2 fbSize{};
    GLuint fbTexture{};
    SampleRenderer sample;
    std::vector<glm::vec4> pixels{};
};
