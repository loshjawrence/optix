#include "CudaUtil.h"

#include <spdlog/spdlog.h>

#include <cuda_runtime.h>

void cudaCheck(int res, std::source_location sl)
{
    const cudaError_t result = static_cast<cudaError_t>(res);
    if( cudaSuccess != result )
    {
        spdlog::error(
            "\nCUDA call {}::{}::{} failed with cudaError_t {}: {}",
            sl.file_name(),
            sl.function_name(),
            sl.line(),
            cudaGetErrorName(result),
            cudaGetErrorString(result)
        );
        exit( 2 );
    }
}

void cudaSyncCheck(std::source_location sl)
{
    cudaCheck(cudaDeviceSynchronize(), sl);
}
