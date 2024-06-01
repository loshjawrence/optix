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

static void context_log_cb(unsigned int level,
                           const char* tag,
                           const char* message,
                           [[maybe_unused]] void*) {
    spdlog::warn("\n[{}][{}]: {}", level, tag, message);
}

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RayGetRecord {
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
    int objectID{};
};

static void printSuccess(
    std::source_location sl = std::source_location::current()) {
    spdlog::info("{} successfully ran", sl.function_name());
}

void SampleRenderer::init() {
    initOptix();

    createContext();
    createModule();
    createRaygenPrograms();
    createMissPrograms();
    createHitgroupPrograms();
    createPipeline();
    //buildSBT();

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

void SampleRenderer::resizeFrameBuffer(const glm::ivec2& newSize)
{
    if (newSize.x <= 0 || newSize.y <= 0) {
        return;
    }

	colorBuffer.resize(newSize.x*newSize.y*sizeof(int));
	launchParams.fbSize = newSize;
    // colorBuffer.resize free's and reallocs so we need to set the pointer again
	launchParams.colorBuffer = colorBuffer.dataAsU32Pointer();
}

uint32_t* SampleRenderer::downloadFrameBuffer()
{
	void* result = colorBuffer.download();
    return reinterpret_cast<uint32_t*>(result);
}
