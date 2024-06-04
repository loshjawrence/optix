#pragma once

#include <glm/glm.hpp>

struct Box
{
    void extend(const glm::vec3& point);
    glm::vec3 diag() const;
    glm::vec3 center() const;

    glm::vec3 min = glm::vec3(FLT_MAX);
    glm::vec3 max = glm::vec3(-FLT_MAX);
};
