#pragma once

#include <vector>

#include <glm/glm.hpp>

struct TriangleMesh {
/*! a simple indexed triangle mesh that our sample renderer will
      render */
    /*! add a unit cube (subject to given xfm matrix) to the current
        triangleMesh */
    void addUnitCube(glm::mat4 xfm);

    //! add aligned cube aith front-lower-left corner and size
    void addCube(glm::vec3 center, glm::vec3 size);

    std::vector<glm::vec3> vertex{};
    std::vector<glm::ivec3> index{};
    glm::vec3 color{1.0f, 0.0f, 0.0f};
};
