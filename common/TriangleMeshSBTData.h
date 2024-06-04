#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct TriangleMeshSBTData {
    glm::vec3 diffuse{};
    glm::vec3* vertex{};
    glm::vec3* normal{};
    glm::vec2* texcoord{};
    glm::ivec3* index{};
    bool hasTexture{};
    cudaTextureObject_t texture;
};
