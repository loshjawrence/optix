#pragma once

#include <glm/glm.hpp>

struct LaunchParams
{
    int frameID{};
    uint32_t* colorBuffer{};
    glm::ivec2 fbSize{};
};
