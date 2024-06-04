#include "SampleRenderer.h"

#include <filesystem>
#include <format>
#include <source_location>
namespace fs = std::filesystem;

#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// NOTE: this include may only appear in a single .cpp
// TODO: Should probably wrap it better then...
// #include <optix_function_table_definition.h>

#include "CMakeGenerated.h"
#include "CudaUtil.h"
#include "IOUtil.h"
#include "OptixUtil.h"
#include "TriangleMeshSBTData.h"
#include "Model.h"
#include "TriangleMesh.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

static void context_log_cb(unsigned int level,
                           const char* tag,
                           const char* message,
                           [[maybe_unused]] void*) {
    spdlog::warn("\n[{}][{}]: {}", level, tag, message);
}

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    // sbt record for raygen program
    // TODO: probably want to extract these two to SBTRecord
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    void* data{};
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    // sbt record for miss program
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    void* data{};
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
    // sbt record for miss program
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    TriangleMeshSBTData data{};
};

static void printSuccess(
    std::source_location sl = std::source_location::current()) {
    spdlog::info("{} successfully ran", sl.function_name());
}

SampleRenderer::SampleRenderer(const Model* model)
    : model(model) {
    initOptix();

    resizeFramebuffer({1200, 1024});

    createContext();
    createModule();
    createRaygenPrograms();
    createMissPrograms();
    createHitgroupPrograms();

    launchParams.traversable = buildAccel();

    createPipeline();
    buildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));
    printSuccess();
}

void SampleRenderer::createContext() {
    const int deviceID{};
    cudaCheck(cudaSetDevice(deviceID));
    cudaCheck(cudaStreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    spdlog::info("deviceProps.name: {}", deviceProps.name);

    cudaCheck(cuCtxGetCurrent(&cudaContext));
    optixCheck(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    optixCheck(optixDeviceContextSetLogCallback(optixContext,
                                                context_log_cb,
                                                nullptr,
                                                4));
    printSuccess();
}

void SampleRenderer::createModule() {
    moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.pipelineLaunchParamsVariableName =
        "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;

#if DEBUG
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipelineCompileOptions.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_USER;
#else
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif

    const fs::path moduleFilename = g_optixIRPath / "devicePrograms.optixir";
    const std::vector<char> ptxCode = getBinaryDataFromFile(moduleFilename);
    optixCheck(optixModuleCreate(optixContext,
                                 &moduleCompileOptions,
                                 &pipelineCompileOptions,
                                 ptxCode.data(),
                                 ptxCode.size(),
                                 nullptr,
                                 nullptr,
                                 &module));
    printSuccess();
}

void SampleRenderer::createRaygenPrograms() {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions{};
    OptixProgramGroupDesc pgDesc{};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    optixCheck(optixProgramGroupCreate(optixContext,
                                       &pgDesc,
                                       raygenPGs.size(),
                                       &pgOptions,
                                       nullptr,
                                       nullptr,
                                       &raygenPGs[0]));
    printSuccess();
}

void SampleRenderer::createMissPrograms() {
    // we do a single ray gen program in this example:
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions{};
    OptixProgramGroupDesc pgDesc{};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    optixCheck(optixProgramGroupCreate(optixContext,
                                       &pgDesc,
                                       missPGs.size(),
                                       &pgOptions,
                                       nullptr,
                                       nullptr,
                                       &missPGs[0]));
    printSuccess();
}

void SampleRenderer::createHitgroupPrograms() {
    // we do a single ray gen program in this example:
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions{};
    OptixProgramGroupDesc pgDesc{};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    optixCheck(optixProgramGroupCreate(optixContext,
                                       &pgDesc,
                                       hitgroupPGs.size(),
                                       &pgOptions,
                                       nullptr,
                                       nullptr,
                                       &hitgroupPGs[0]));
    printSuccess();
}

void SampleRenderer::createPipeline() {
    std::vector<OptixProgramGroup> programGroups;
    programGroups.insert(programGroups.end(),
                         raygenPGs.begin(),
                         raygenPGs.end());
    programGroups.insert(programGroups.end(), missPGs.begin(), missPGs.end());
    programGroups.insert(programGroups.end(),
                         hitgroupPGs.begin(),
                         hitgroupPGs.end());

    optixCheck(optixPipelineCreate(optixContext,
                                   &pipelineCompileOptions,
                                   &pipelineLinkOptions,
                                   programGroups.data(),
                                   int(programGroups.size()),
                                   nullptr,
                                   nullptr,
                                   &pipeline));
    constexpr int KB = 1024;
    constexpr int TWO_KB = 2 * KB;
    optixCheck(optixPipelineSetStackSize(
        // [in] The pipeline to configure the stack size for
        pipeline,
        // [in] The direct stack size requirement for direct callables invoked from IS(intersection) or AH(anyhit).
        TWO_KB,
        // [in] The direct stack size requirement for direct callables invoked from RG(raygen), MS(miss), or CH(closesthit).
        TWO_KB,
        // [in] The continuation stack requirement.
        TWO_KB,
        // [in] The maximum depth of a traversable graph passed to trace.
        1));

    printSuccess();
}

void SampleRenderer::buildSBT() {
    // RAYGEN RECORDS
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); ++i) {
        RaygenRecord rec{};
        optixCheck(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        //rec.data = ...later;
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // MISS RECORDS
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); ++i) {
        MissRecord rec{};
        optixCheck(optixSbtRecordPackHeader(missPGs[i], &rec));
        //rec.data = ...later;
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = int(missRecords.size());

    // HITGROUP RECORDS
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID = 0; meshID < model->meshes.size(); meshID++) {
        HitgroupRecord rec;
        // all meshes use the same code, so all same hit group
        optixCheck(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
        rec.data.diffuse = model->meshes[meshID]->diffuse;
        rec.data.vertex = (glm::vec3*)vertexBuffer[meshID].d_pointer();
        rec.data.index = (glm::ivec3*)indexBuffer[meshID].d_pointer();
        hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();

    printSuccess();
}

OptixTraversableHandle SampleRenderer::buildAccel() {

    vertexBuffer.resize(model->meshes.size());
    indexBuffer.resize(model->meshes.size());


    // triangle inputs
    std::vector<OptixBuildInput> triangleInput(model->meshes.size());
    std::vector<CUdeviceptr> d_vertices(model->meshes.size());
    std::vector<CUdeviceptr> d_indices(model->meshes.size());
    std::vector<uint32_t> triangleInputFlags(model->meshes.size());
    OptixTraversableHandle asHandle{};

    for (int meshID = 0; meshID < model->meshes.size(); meshID++) {
        // upload the model->to the device: the builder
        TriangleMesh& mesh = *(model->meshes[meshID]);
        vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
        indexBuffer[meshID].alloc_and_upload(mesh.index);

        triangleInput[meshID] = {};
        triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
        d_indices[meshID] = indexBuffer[meshID].d_pointer();

        triangleInput[meshID].triangleArray.vertexFormat =
            OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
        triangleInput[meshID].triangleArray.numVertices =
            (int)mesh.vertex.size();
        triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

        triangleInput[meshID].triangleArray.indexFormat =
            OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
        triangleInput[meshID].triangleArray.numIndexTriplets =
            (int)mesh.index.size();
        triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

        triangleInputFlags[meshID] = 0;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    optixCheck(
        optixAccelComputeMemoryUsage(optixContext,
                                     &accelOptions,
                                     triangleInput.data(),
                                     (int)model->meshes.size(), // num_build_inputs
                                     &blasBufferSizes));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    optixCheck(optixAccelBuild(optixContext,
                                /* stream */ 0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)model->meshes.size(),
                                tempBuffer.d_pointer(),
                                tempBuffer.byteSize(),

                                outputBuffer.d_pointer(),
                                outputBuffer.byteSize(),

                                &asHandle,

                                &emitDesc,
                                1));
    cudaSyncCheck();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    optixCheck(optixAccelCompact(optixContext,
                                  /*stream:*/ 0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.byteSize(),
                                  &asHandle));
    cudaSyncCheck();

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}

void SampleRenderer::render() {
    if (!launchParams.frame.colorBuffer || launchParams.frame.size.x == 0) {
        spdlog::warn(
            "Ran with invalid launch params, launchParamsfbSize.x is 0 or "
            "launchParams.frame.colorBuffer is nullptr. Need to init first.");
        return;
    }

    launchParamsBuffer.upload(&launchParams, 1);
    const int depth = 1;
    optixCheck(optixLaunch(pipeline,
                           stream,
                           launchParamsBuffer.d_pointer(),
                           launchParamsBuffer.byteSize(),
                           &sbt,
                           launchParams.frame.size.x,
                           launchParams.frame.size.y,
                           depth));

    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    cudaSyncCheck();
}

void SampleRenderer::saveFramebuffer() {
    std::vector<uint32_t> pixels;
    downloadFramebuffer(pixels);
    const fs::path filename = g_debugImagesPath / "example2.png";
    if (!stbi_write_png(filename.string().c_str(),
                        launchParams.frame.size.x,
                        launchParams.frame.size.y,
                        4,
                        reinterpret_cast<const void*>(pixels.data()),
                        launchParams.frame.size.x * sizeof(uint32_t))) {
        spdlog::error("Failed to save framebuffer to {}.", filename.string());
    }

    spdlog::info("Framebuffer saved to {} ... done.", filename.string());

    printSuccess();
}

void SampleRenderer::resizeFramebuffer(const glm::ivec2& newSize) {
    if (newSize.x <= 0 || newSize.y <= 0) {
        return;
    }

    colorBuffer.resize(newSize.x * newSize.y * sizeof(int));
    launchParams.frame.size = newSize;
    // colorBuffer.resize free's and reallocs so we need to set the pointer again
    launchParams.frame.colorBuffer = colorBuffer.dataAsU32Pointer();

    setCamera(lastSetCamera);

    printSuccess();
}

void SampleRenderer::setCamera(const Camera& camera) {
    lastSetCamera = camera;
    launchParams.camera.position = camera.from;
    launchParams.camera.direction = glm::normalize(camera.at - camera.from);
    const float cosFovy = 0.66f;
    const float aspect =
        launchParams.frame.size.x / float(launchParams.frame.size.y);
    launchParams.camera.horizontal = cosFovy * aspect *
        glm::normalize(glm::cross(launchParams.camera.direction, camera.up));
    launchParams.camera.vertical = cosFovy *
        glm::normalize(glm::cross(launchParams.camera.horizontal,
                                  launchParams.camera.direction));
}

void SampleRenderer::downloadFramebuffer(std::vector<uint32_t>& outPayload) {
    outPayload.resize(launchParams.frame.size.x * launchParams.frame.size.y);
    colorBuffer.download(&outPayload[0], outPayload.size());
}
