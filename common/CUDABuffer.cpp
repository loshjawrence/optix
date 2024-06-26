#include "CUDABuffer.h"
#include "CudaUtil.h"

#include <cuda_runtime.h>

CUdeviceptr CUDABuffer::d_pointer() const {
    return (CUdeviceptr)d_ptr;
}

uint32_t* CUDABuffer::dataAsU32Pointer() const {
    return reinterpret_cast<uint32_t*>(d_ptr);
}

void CUDABuffer::resize(size_t size) {
    if (d_ptr) {
        free();
    }

    alloc(size);
}

void CUDABuffer::alloc(size_t size) {
    assert(!d_ptr);
    sizeInBytes = size;
    cudaCheck(cudaMalloc(&d_ptr, sizeInBytes));
    assert(d_ptr);
}

void CUDABuffer::free() {
    cudaCheck(cudaFree(d_ptr));
    d_ptr = nullptr;
    sizeInBytes = 0;
}

size_t CUDABuffer::byteSize() const {
    return sizeInBytes;
}
