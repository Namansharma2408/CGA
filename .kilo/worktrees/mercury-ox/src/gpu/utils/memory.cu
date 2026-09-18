
#include "gpu/utils/memory.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") \
            + cudaGetErrorString(err) + " in " + __FILE__ + ":" + std::to_string(__LINE__)); \
} while(0)

void* gpu_alloc(size_t bytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
}
void  gpu_free(void* ptr)                          { cudaFree(ptr); }
void  gpu_memcpy_h2d(void* d, const void* s, size_t b) { CUDA_CHECK(cudaMemcpy(d, s, b, cudaMemcpyHostToDevice)); }
void  gpu_memcpy_d2h(void* d, const void* s, size_t b) { CUDA_CHECK(cudaMemcpy(d, s, b, cudaMemcpyDeviceToHost)); }
void  gpu_memset_zero(void* p, size_t b)           { CUDA_CHECK(cudaMemset(p, 0, b)); }


template<typename T>
DevPtr<T>::DevPtr(int n) : n(n) {
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
}
template<typename T>
DevPtr<T>::~DevPtr() { if (ptr) cudaFree(ptr); }

template<typename T>
void DevPtr<T>::upload(const T* host_data) {
    CUDA_CHECK(cudaMemcpy(ptr, host_data, n * sizeof(T), cudaMemcpyHostToDevice));
}
template<typename T>
void DevPtr<T>::download(T* host_data) const {
    CUDA_CHECK(cudaMemcpy(host_data, ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
}

template struct DevPtr<int>;
template struct DevPtr<float>;
