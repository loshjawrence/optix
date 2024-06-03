#pragma once

#include "GLFCameraWindow.h"
#include "SampleRenderer.h"
#include "TriangleMesh.h"
#include "Camera.h"

#include <vector>

#include <glm/glm.hpp>
#include <glad/glad.h>


struct SampleWindow : public GLFCameraWindow {
    SampleWindow(const std::string& title,
                 const TriangleMesh& model,
                 const Camera& camera,
                 const float worldScale);

    virtual void render() override;
    virtual void draw() override;
    virtual void resize(glm::ivec2 newSize) override;

    glm::ivec2 fbSize{};
    GLuint fbTexture{};
    SampleRenderer sample;
    std::vector<uint32_t> pixels{};
};
