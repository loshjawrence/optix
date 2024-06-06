#pragma once

#include "CUDABuffer.h"
#include "Camera.h"
#include "LaunchParams.h"

#include <cuda_runtime.h> // need for cudaDeviceProp
#include <optix_stubs.h>  // needed for the rest

struct Model;
struct QuadLight;

class SampleRenderer {
public:
    // performs all setup, including initializing optix, creates module, pipeline, programs, SBT, etc.
    SampleRenderer(const Model* model, const QuadLight& light);

    // render one frame
    void render();

    void saveFramebuffer();

    // resize frame buffer
    void resizeFramebuffer(const glm::ivec2& newSize);
    void downloadFramebuffer(std::vector<glm::vec4>& outPayload);

    void setCamera(const Camera& camera);

    bool denoiserOn = true;
    bool accumulate = true;

    LaunchParams launchParams;

protected:
    /*! runs a cuda kernel that performs gamma correction and float4-to-rgba conversion */
    void computeFinalPixelColors();

    /*! creates and configures a optix device context (in this simple
        example, only for the primary GPU device) */
    void createContext();

    /*! creates the module that contains all the programs we are going
        to use. in this simple example, we use a single module from a
        single .cu file, using a single embedded ptx string */
    void createModule();

    /*! does all setup for the raygen program(s) we are going to use */
    void createRaygenPrograms();

    /*! does all setup for the miss program(s) we are going to use */
    void createMissPrograms();

    /*! does all setup for the hitgroup program(s) we are going to use */
    void createHitgroupPrograms();

    /*! assembles the full pipeline of all programs */
    void createPipeline();
    void createTextures();

    /*! constructs the shader binding table */
    void buildSBT();

    /*! build an acceleration structure for the given triangle mesh */
    OptixTraversableHandle buildAccel();

protected:
    /*! @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext cudaContext{};
    CUstream stream{};
    cudaDeviceProp deviceProps{};
    /*! @} */

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext{};

    /*! @{ the pipeline we're building */
    OptixPipeline pipeline{};
    OptixPipelineCompileOptions pipelineCompileOptions{};
    OptixPipelineLinkOptions pipelineLinkOptions{};
    /*! @} */

    /*! @{ the module that contains out device programs */
    OptixModule module{};
    OptixModuleCompileOptions moduleCompileOptions{};
    /* @} */

    /*! vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs{};
    CUDABuffer raygenRecordsBuffer{};
    std::vector<OptixProgramGroup> missPGs{};
    CUDABuffer missRecordsBuffer{};
    std::vector<OptixProgramGroup> hitgroupPGs{};
    CUDABuffer hitgroupRecordsBuffer{};
    OptixShaderBindingTable sbt{};

    CUDABuffer launchParamsBuffer{};
    /*! @} */

    /*! the color buffer we use during _rendering_, which is a bit
        larger than the actual displayed frame buffer (to account for
        the border), and in float4 format (the denoiser requires
        floats) */
    CUDABuffer fbRender{};
    CUDABuffer fbNormal{};
    CUDABuffer fbAlbedo{};

    /*! output of the denoiser pass, in float4 */
    CUDABuffer denoisedBuffer{};

    /*! the actual final color buffer used for display, in rgba8 */
    CUDABuffer finalColorBuffer{};

    OptixDenoiser denoiser{};
    CUDABuffer denoiserScratch{};
    CUDABuffer denoiserState{};
    CUDABuffer denoiserIntensity{};

    /*! the camera we are to render with. */
    Camera lastSetCamera{};

    /*! the model we are going to trace rays against */
    const Model* model{};
    /*! one buffer per input mesh */
    std::vector<CUDABuffer> vertexBuffer{};
    std::vector<CUDABuffer> normalBuffer{};
    std::vector<CUDABuffer> texcoordBuffer{};
    std::vector<CUDABuffer> indexBuffer{};

    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer{};

    // one texture per object and pixel array per used texture
    std::vector<cudaArray_t> textureArrays{};
    std::vector<cudaTextureObject_t> textureObjects{};
};
