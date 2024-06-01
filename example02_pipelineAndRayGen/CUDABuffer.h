#pragma once

#include "CudaUtil.h"

#include <vector>
#include <assert.h>

#include <cuda_runtime.h>
#include <optix_types.h>

struct CUDABuffer {
    CUdeviceptr d_pointer() const;
    uint32_t* dataAsU32Pointer() const;
    void resize(size_t size);
    void alloc(size_t size);
    void free();

    template <class T>
    void alloc_and_upload(const std::vector<T>& vt) {
        alloc(vt.size() * sizeof(T));
        upload((const T*)vt.data(), vt.size());
    }

    template <class T>
    void upload(const void* t, size_t byteCount) {
        assert(!d_ptr);
        assert(byteCount == sizeInBytes);
        cudaCheck(cudaMemcpy(d_ptr, t, sizeInBytes, cudaMemcpyDeviceToHost));
    }

    void* download();

private:
    size_t sizeInBytes{};
    void* d_ptr{};
};
