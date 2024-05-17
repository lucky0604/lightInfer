#include "device.h"
#include "tensor.h"

#ifndef __APPLE__
#include <malloc.h>
#endif

#define ALIGN_SIZE (32)

using namespace lightInfer;

void* Device::aligned_alloc(size_t size) {
#ifdef WIN32
    return _aligned_malloc(size, ALIGN_SIZE);
#elif defined(__ANDROID__ || defined (ANDROID))
    return memalign(ALIGN_SIZE, size);
#else
    void* ptr = nullptr;
    auto err = posix_memalign(&ptr, ALIGN_SIZE, size);
    INFER_ASSERT(!err, "failed to malloc");
    return ptr;
#endif

}

void Device::aligned_free(void* ptr) {
#ifdef WIN32
    _aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

void* CPUDevice::allocate(size_t len) {
#ifdef ENABLE_ASAN
    return aligned_alloc(len);
#else
    auto it = m_free_memory.lower_bound(len);
    void* ptr = nullptr;
    if (it != m_free_memory.end() && it->second.size() > 0) {
        ptr = it->second.back();
        it->second.pop_back();
        if (it->second.size() < 1) {
            m_free_memory.erase(it);
        }
    } else {
        ptr = aligned_alloc(len);
        m_alloc_memory[ptr] = len;
    }
    return ptr;
#endif
}

void* CPUDevice::allocate_host(size_t len) {
    return aligned_alloc(len);
}

void CPUDevice::free_host(void* ptr) {
    aligned_free(ptr);
}

void CPUDevice::free_device(void* ptr) {
#ifdef ENABLE_ASAN
    aligned_free(ptr);
#else
    INFER_ASSERT(m_alloc_memory.count(ptr) == 1, "memory is not allocated by the DeviceCPU");
    size_t len = m_alloc_memory[ptr];
    m_free_memory[len].push_back(ptr);
#endif
}

CPUDevice::~CPUDevice() {

#ifndef ENABLE_ASAN
    for (auto it: m_free_memory) {
        for (auto ptr: it.second) {
            INFER_ASSERT(m_alloc_memory.count(ptr) == 1, "memory is not allocated by the DeviceCPU");
            aligned_free(ptr);
        }
    }
#endif
}

#if ENABLE_GPU
void* GPUDevice::allocate(size_t len) {
    auto it = m_free_memory.lower_bound(len);
    void* ptr = nullptr;

    if (it != m_free_memory.end() && it->second.size() > 0) {
        ptr = it->second.back();
        it->second.pop_back();
        if (it->second.size() < 1) {
            m_free_memory.erase(it);
        }
    } else {
        CUDA_CHECK(cudaMalloc(&ptr, len));
        m_alloc_memory[ptr] = len;
    }
    return ptr;
}

void GPUDevice::free_device(void* ptr) {
    INFER_ASSERT(m_alloc_memory.count(ptr) == 1, "memory is not allocated by the DeviceGPU");
    size_t len = m_alloc_memory[ptr];
    m_free_memory[len].push_back(ptr);
}

void* GPUDevice::allocate_host(size_t len) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, len));
    return ptr;
}

void GPUDevice::free_host(void* ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

GPUDevice::~GPUDevice() {
    for (auto it: m_free_memory) {
        for (auto ptr: it.second) {
            INFER_ASSERT(m_alloc_memory.count(ptr) == 1, "memory is not allocated by the DeivceGPU");
            CUDA_CHECK(cudaFree(ptr));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(m_handle.stream));
    CUBLAS_CHECK(cublasDestroy(m_handle.cublas_handle));
}

void GPUDevice::host2device_copy(void* device, const void* host, size_t size, bool async) {
    if (async) {
        CUDA_CHECK(cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice, m_handle.stream));
    } else {
        CUDA_CHECK(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    }
}

void GPUDevice::device2host_copy(void* host, const void* device, size_t size, bool async) {
    if (async) {
        CUDA_CHECK(cudaMemcpyAsync(host, device, size, cudaMemcpyDeviceToHost, m_handle.stream));
    } else {
        CUDA_CHECK(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }
}

void GPUDevice::device2device_copy(void* dst, const void* src, size_t size, bool async) {
    if (async) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, m_handle.stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    }
}

#endif