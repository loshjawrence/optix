#pragma once

#include <glm/glm.hpp>

struct Camera {
    /*! camera position - *from* where we are looking */
    glm::vec3 from{};
    /*! which point we are looking *at* */
    glm::vec3 at{};
    /*! general up-vector */
    glm::vec3 up{};
};
