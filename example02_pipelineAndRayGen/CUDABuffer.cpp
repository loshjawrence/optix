#include "CUDABuffer.h"
#include "CudaUtil.h"

CUdeviceptr CUDABuffer::d_pointer() const
{
    return (CUdeviceptr)d_ptr;
}

void CUDABuffer::resize(size_t size)
{
    if (d_ptr)
    {
        free();
    }

    alloc(size);
}

void CUDABuffer::alloc(size_t size)
{
    assert(!d_ptr);
    sizeInBytes = size;
    cudaCheck(cudaMalloc(&d_ptr, sizeInBytes));
}

void CUDABuffer::free()
{
    cudaCheck(cudaFree(d_ptr));
    d_ptr = nullptr;
    sizeInBytes = 0;
}

