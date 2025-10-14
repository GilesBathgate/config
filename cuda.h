#ifndef CUDA_H
#define CUDA_H

#ifdef __NVCC__

// NVCC path: just include the real CUDA headers
#include <cuda_runtime.h>

#else

// G++ path: mock everything
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <coroutine>
#include <cstring> // For memcpy

// Mock CUDA keywords
#define __device__
#define __host__
#define __global__
#define __shared__ static

// Mock CUDA error handling
using cudaError_t = int;
const int cudaSuccess = 0;

// Mock memory management
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    *devPtr = new char[size];
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int) {
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    delete[] static_cast<char*>(devPtr);
    return cudaSuccess;
}

// Mock dim3 struct
struct dim3 {
    unsigned int x, y, z;
};

// Global thread/block variables for simulation
dim3 threadIdx, blockIdx, blockDim;

// --- Coroutine Machinery for __syncthreads() ---

template <typename... Args>
struct Task
{
    struct promise_type;
private:
    using handle = std::coroutine_handle<promise_type>;
    handle handle_;
    void (*coroutine_)(Args...);
public:
    Task(void (*coroutine)(Args...))
    : coroutine_(coroutine)
    {
    }

    void start(Args... args)
    {
        coroutine_(args...);
        handle_ = handle::from_address(promise_type::current_address);
        promise_type::current_address = nullptr;
        handle_.resume(); // Start the coroutine until the first suspension
    }

    ~Task() {
        if (handle_) {
            handle_.destroy();
        }
    }

    void resume() const { if (handle_) handle_.resume(); }
    bool done() const { return handle_ && handle_.done(); }

    struct promise_type
    {
        static auto initial_suspend() { return std::suspend_always(); }
        static auto final_suspend() noexcept { return std::suspend_always(); }
        static auto yield_value(int) noexcept { return std::suspend_always(); }
        static void return_void() {}
        static void unhandled_exception() { std::terminate(); }
        static void* current_address;

        void get_return_object()
        {
            current_address = handle::from_promise(*this).address();
        }
    };
};

template <typename... Args>
void* Task<Args ...>::promise_type::current_address = nullptr;

// Specialize coroutine_traits for void returning functions to use our Task's promise_type
template <typename... Args>
struct std::coroutine_traits<void, Args...> {
   using promise_type = Task<Args...>::promise_type;
};


// The all-important mock for __syncthreads
#define __syncthreads() co_yield 0

#endif // __NVCC__

#endif // CUDA_H
