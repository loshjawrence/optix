#define NOMINMAX

#include "SampleRenderer.h"
#include <cuda.h>

__device__ glm::vec4 clamp(glm::vec4 a, float min, float max)
{
    return glm::vec4{
        glm::min(glm::max(a.x, min), max),
        glm::min(glm::max(a.y, min), max),
        glm::min(glm::max(a.z, min), max),
        glm::min(glm::max(a.w, min), max)
    };
}

__device__ glm::vec4 tonemapACESFilm(float4 X)
{
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    // for the orginal aces curve, mult x by 0.6 first as mentioned in the post.
    // x *= 0.6;
    const float a{2.51};
    const float b{0.03};
    const float c{2.43};
    const float d{0.59};
    const float e{0.14};
    glm::vec3 x = glm::vec3(X.x, X.y, X.z);
    glm::vec3 y = (x * (a * x + b)) / (x * (c * x + d) + e);
    return clamp(glm::vec4{y.x, y.y, y.z, 1.0f}, 0.0f, 1.0f);
}

__global__ void computeFinalPixelColorsKernal(glm::vec4* finalColorBuffer,
                                              float4* denoisedBuffer,
                                              glm::ivec2 size) {
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;

    if (pixelX >= size.x || pixelY >= size.y)
    {
        return;
    }

    int pixelID = pixelX + pixelY * size.x;
    float4 raw = denoisedBuffer[pixelID];
    finalColorBuffer[pixelID] = tonemapACESFilm(raw);
}

void SampleRenderer::computeFinalPixelColors() {
    glm::ivec2 fbSize = launchParams.frame.size;
    glm::ivec2 blockSize{32, 32};
    glm::ivec2 numBlocks{(fbSize.x + blockSize.x - 1.0f) / blockSize.x,
                         (fbSize.y + blockSize.y - 1.0f) / blockSize.y};
    computeFinalPixelColorsKernal<<<
        dim3(numBlocks.x, numBlocks.y), dim3(blockSize.x, blockSize.y)>>>(
        (glm::vec4*)finalColorBuffer.d_pointer(),
        (float4*)denoisedBuffer.d_pointer(),
        fbSize);
}
