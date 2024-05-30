#include "OptixUtil.h"

#include <source_location>   

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <spdlog/spdlog.h>

static void OptixCheck(OptixResult result, std::source_location sl = std::source_location::current())
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

void InitOptix()
{
    // I think calling cudaFree creates an optix context implicitly
    cudaFree(0);

    int numDevices;
    cudaGetDeviceCount(&numDevices);

    if (numDevices == 0)
    {
        spdlog::error("\nERROR: no CUDA capable devices found!");
    }

    OptixCheck( optixInit() );

    spdlog::info("\noptixInit: OPTIX_SUCCESS. Found {} CUDA devices", numDevices);
}
