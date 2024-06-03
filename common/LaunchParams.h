#pragma once

#include <glm/glm.hpp>
#include <optix_types.h>

struct LaunchParams
{
    //int frameID{};
    //uint32_t* colorBuffer{};
    //glm::ivec2 fbSize{};

    struct {
      uint32_t *colorBuffer;
      glm::ivec2     size;
    } frame;
    
    struct {
      glm::vec3 position;
      glm::vec3 direction;
      glm::vec3 horizontal;
      glm::vec3 vertical;
    } camera;

    OptixTraversableHandle traversable;
};
