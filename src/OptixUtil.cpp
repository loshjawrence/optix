#include "OptixUtil.h"

#include <source_location>   

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <spdlog/spdlog.h>

// no reason this needs to be exposed
static void OptixCheck(int result, std::source_location sl = std::source_location::current())
{
    if( OPTIX_SUCCESS != result )
    {
        spdlog::error("\nOptix call {}::{}::{} failed with code {}", sl.file_name(), sl.function_name(), sl.line(), result );
        exit( 2 );                                                      \
    }                                                                 \
}

void InitOptix()
{
    // I think calling cudaFree creates an optix context
    cudaFree(0);

    int numDevices;
    cudaGetDeviceCount(&numDevices);

    if (numDevices == 0)
    {
        spdlog::error("\nERROR: no CUDA capable devices found!");
    }

    spdlog::info("\nFound {} CUDA devices", numDevices);

    OptixCheck( optixInit() );
}
