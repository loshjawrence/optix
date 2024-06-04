#include "TriangleMesh.h"

#define GLM_FORCE_CTOR_INIT GLM_CTOR_INITIALIZER_LIST
#include <array>
#include <glm/gtc/matrix_transform.hpp>

void TriangleMesh::addCube(glm::vec3 center, glm::vec3 size) {
    //! add aligned cube with front-lower-left corner and size
    const glm::mat4 identity = glm::mat4(1.0f);

    const glm::vec3 halfSize = 0.5f * size;
    const glm::vec3 translate = center - halfSize;

    // TRS in that order (even though the typical matrix transform math would suggest translate(rotate(scale(..)))
    glm::mat4 xfm = glm::translate(identity, translate);
    xfm = glm::scale(xfm, size);
    addUnitCube(xfm);
}

void TriangleMesh::addUnitCube(glm::mat4 xfm) {
    /*! add a unit cube (subject to given xfm matrix) to the current
      triangleMesh */
    int firstVertexID = (int)vertex.size();
    vertex.push_back(glm::vec3(xfm * glm::vec4(0.f, 0.f, 0.f, 1.0f)));
    vertex.push_back(glm::vec3(xfm * glm::vec4(1.f, 0.f, 0.f, 1.0f)));
    vertex.push_back(glm::vec3(xfm * glm::vec4(0.f, 1.f, 0.f, 1.0f)));
    vertex.push_back(glm::vec3(xfm * glm::vec4(1.f, 1.f, 0.f, 1.0f)));
    vertex.push_back(glm::vec3(xfm * glm::vec4(0.f, 0.f, 1.f, 1.0f)));
    vertex.push_back(glm::vec3(xfm * glm::vec4(1.f, 0.f, 1.f, 1.0f)));
    vertex.push_back(glm::vec3(xfm * glm::vec4(0.f, 1.f, 1.f, 1.0f)));
    vertex.push_back(glm::vec3(xfm * glm::vec4(1.f, 1.f, 1.f, 1.0f)));

    std::array<glm::ivec3, 12> relativeIndices{glm::ivec3{0, 1, 3},
                                               glm::ivec3{2, 0, 3},
                                               glm::ivec3{5, 7, 6},
                                               glm::ivec3{5, 6, 4},
                                               glm::ivec3{0, 4, 5},
                                               glm::ivec3{0, 5, 1},
                                               glm::ivec3{2, 3, 7},
                                               glm::ivec3{2, 7, 6},
                                               glm::ivec3{1, 5, 7},
                                               glm::ivec3{1, 7, 3},
                                               glm::ivec3{4, 0, 2},
                                               glm::ivec3{4, 2, 6}};
    for (const auto& v : relativeIndices) {
        index.push_back(firstVertexID + v);
    }
}
