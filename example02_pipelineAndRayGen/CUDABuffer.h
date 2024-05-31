#pragma once

#include "CudaUtil.h"

#include <assert.h>
#include <vector>

#include <optix_types.h>
#include <cuda_runtime.h>
#include <optix.h>

struct CUDABuffer {
    CUdeviceptr d_pointer() const;
    void resize(size_t size);
    void alloc(size_t size);
    void free();

    template<class T>
    void alloc_and_upload(const std::vector<T>& vt)
    {
        alloc(vt.size() * sizeof(T));
        upload((const T*)vt.data(), vt.size());
    }

    template<class T>
    void upload(const T* t, size_t count)
    {
        assert(!d_ptr);
        const size_t byteCount = count*sizeof(T);
        assert(byteCount == sizeInBytes);
        cudaCheck(cudaMemcpy(d_ptr, t, sizeInBytes, cudaMemcpyDeviceToHost));
    }

    template<class T>
    void download(T* t, size_t count)
    {
        assert(!d_ptr);
        const size_t byteCount = count*sizeof(T);
        assert(byteCount == sizeInBytes);
        cudaCheck(cudaMemcpy((void*)t, d_ptr, sizeInBytes, cudaMemcpyHostToDevice));
    }

private:
    size_t sizeInBytes{};
    void* d_ptr{};
};
