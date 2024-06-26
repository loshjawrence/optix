#pragma once

#include "CudaUtil.h"

#include <vector>
#include <assert.h>

#include <cuda_runtime.h>
#include <optix_types.h>

struct CUDABuffer {
    CUdeviceptr d_pointer() const;
    size_t byteSize() const;
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
    void upload(const T* t, size_t size) {
        assert(d_ptr);
        const size_t byteCount = size * sizeof(T);
        assert(byteCount == sizeInBytes);
        cudaCheck(cudaMemcpy(d_ptr,
                             reinterpret_cast<const void*>(t),
                             sizeInBytes,
                             cudaMemcpyHostToDevice));
    }

    template<typename T>
    void download(T *t, size_t count)
    {
      assert(d_ptr != nullptr);
        const size_t byteCount = count * sizeof(T);
      assert(byteCount == byteCount);
      cudaCheck(cudaMemcpy((void *)t, d_ptr,
                        byteCount, cudaMemcpyDeviceToHost));
    }

private:
    size_t sizeInBytes{};
    void* d_ptr{};
};
