#include "OptixUtil.h"
#include "CudaUtil.h"

#include <source_location>   

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
// NOTE: this include may only appear in a single .cpp
// TODO: Should probably wrap it better then...
#include <optix_function_table_definition.h>

#include <spdlog/spdlog.h>

void optixCheck(OptixResult result, std::source_location sl)
{
    if( OPTIX_SUCCESS != result )
    {
        spdlog::error(
            "\nOptix call {}::{}::{} failed with OptixResult {}: {}",
            sl.file_name(),
            sl.function_name(),
            sl.line(),
            optixGetErrorName(result),
            optixGetErrorString(result)
            );
        exit( 2 );
    }
}

void initOptix()
{
    // I think calling cudaFree creates an optix context implicitly
    cudaCheck(cudaFree(0));

    int numDevices;
    cudaCheck(cudaGetDeviceCount(&numDevices));

    if (numDevices == 0)
    {
        spdlog::error("\nERROR: no CUDA capable devices found!");
    }

    optixCheck( optixInit() );

    spdlog::info("\noptixInit: OPTIX_SUCCESS. Found {} CUDA devices", numDevices);
}

static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             [[maybe_unused]]void*)
{
	spdlog::error("\n[{}][{}]: {}", level, tag, message );
}

void optixSetLogCallback(const OptixDeviceContext &optixContext)
{
    optixCheck(optixDeviceContextSetLogCallback(optixContext,
                                                context_log_cb,
                                                nullptr,
                                                4));
}
