#pragma once

#include <glm/glm.hpp>

struct QuadLight {
    glm::vec3 origin{};
    glm::vec3 du{};
    glm::vec3 dv{};
    glm::vec3 power{};
};
