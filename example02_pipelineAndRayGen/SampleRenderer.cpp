#include "SampleRenderer.h"


#include <source_location>
#include <format>
#include <filesystem>
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

#include "CudaUtil.h"
#include "OptixUtil.h"
#include "IOUtil.h"
#include "CMakeGenerated.h"

static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             [[maybe_unused]]void*)
{
	spdlog::warn("\n[{}][{}]: {}", level, tag, message );
}

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RayGetRecord
{
    // sbt record for raygen program
    // TODO: probably want to extract these two to SBTRecord
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    void* data{};
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    // sbt record for miss program
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    void* data{};
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    // sbt record for miss program
    char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    int objectID{};
};

static void printSuccess(std::source_location sl = std::source_location::current())
{
    spdlog::info("{} successfully ran", sl.function_name());
}

void SampleRenderer::init()
{
    initOptix();

    createContext();
    createModule();
    //createRaygenPrograms();
    //createMissPrograms();
    //createHitgroupPrograms();
    //createPipeline();
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

void SampleRenderer::createModule()
{
    moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;

#if DEBUG
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
	pipelineCompileOptions.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW
        | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH
        | OPTIX_EXCEPTION_FLAG_USER;
#else
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif

    const fs::path moduleFilename = g_optixIRPath / "devicePrograms.optixir";
    const std::vector<char> ptxCode = getBinaryDataFromFile(moduleFilename);
    OptixResult result = optixModuleCreate(optixContext,
                                   &moduleCompileOptions,
                                   &pipelineCompileOptions,
                                   ptxCode.data(),
                                   ptxCode.size(),
                                   nullptr,
        nullptr,
                                   &module);
	optixCheck(result);
	printSuccess();
}
