#pragma once

#include <glm/glm.hpp>

struct TriangleMeshSBTData {
    glm::vec3 color{};
    glm::vec3* vertex{};
    glm::ivec3* index{};
};
