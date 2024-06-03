#include "TriangleMesh.h"

#include <array>
#include <glm/gtc/matrix_transform.hpp>

void TriangleMesh::addCube(glm::vec3 center, glm::vec3 size) {
    //! add aligned cube with front-lower-left corner and size

    glm::mat4 xfm = glm::scale(glm::identity<glm::mat4>(), size);
    xfm = glm::translate(xfm, center - (0.5f * size));
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
