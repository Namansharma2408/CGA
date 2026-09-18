#pragma once
#include <vector>

void* gpu_alloc(size_t bytes);
void  gpu_free(void* ptr);
void  gpu_memcpy_h2d(void* dst, const void* src, size_t bytes);
void  gpu_memcpy_d2h(void* dst, const void* src, size_t bytes);
void  gpu_memset_zero(void* ptr, size_t bytes);
float gpu_timer_ms();

template<typename T>
struct DevPtr {
    T* ptr{nullptr};
    int n{0};
    explicit DevPtr(int n);
    ~DevPtr();
    T* get() { return ptr; }
    const T* get() const { return ptr; }
    void upload(const T* host_data);
    void download(T* host_data) const;
    DevPtr(const DevPtr&) = delete;
    DevPtr& operator=(const DevPtr&) = delete;
};

extern template struct DevPtr<int>;
extern template struct DevPtr<float>;
