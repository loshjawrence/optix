#pragma once

#include "GLFWindow.h"
#include "SampleRenderer.h"

#include <vector>

#include <glm/glm.hpp>
#include <glad/glad.h>


struct SampleWindow : public GLFWindow {
    SampleWindow(const std::string& title);

    virtual void render() override;
    virtual void draw() override;
    virtual void resize(glm::ivec2 newSize) override;

    glm::ivec2 fbSize{};
    GLuint fbTexture{0};
    SampleRenderer sample{};
    std::vector<uint32_t> pixels{};
};
