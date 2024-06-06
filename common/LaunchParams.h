#pragma once

#include <glm/glm.hpp>
#include <optix_types.h>

struct LaunchParams {
    int numPixelSamples = 1;

    struct {
        int frameID{};
        uint32_t* colorBuffer{};
        glm::vec4* renderBuffer{};
        glm::vec4* normalBuffer{};
        glm::vec4* albedoBuffer{};
        glm::ivec2 size{};
        int accumID{};
    } frame;

    struct {
        glm::vec3 position{};
        glm::vec3 direction{};
        glm::vec3 horizontal{};
        glm::vec3 vertical{};
    } camera;

    struct {
        glm::vec3 origin{};
        glm::vec3 du{};
        glm::vec3 dv{};
        glm::vec3 power{};

    } light;

    OptixTraversableHandle traversable{};
};
