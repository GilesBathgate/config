#ifndef CUDA_H
#define CUDA_H

#include <stdio.h> // For fprintf
#include <stdlib.h> // For exit

#ifdef __NVCC__
// --- NVCC Path ---
// If compiling with nvcc, just include the real CUDA headers.
#include <cuda_runtime.h>

// Real implementation of gpuAssert
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#else
// --- g++ Path ---
#include <cstddef> // For size_t
#include <cstring> // For memcpy

// Mock for __global__ and __device__ specifiers
#define __global__
#define __device__
#define __shared__
#define __syncthreads() (void)0

// Mock enums for CUDA types
typedef enum { cudaSuccess = 0 } cudaError_t;
typedef enum { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost } cudaMemcpyKind;

// Mock for the dim3 type
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};

// Mock function implementations
inline cudaError_t cudaMalloc(void** devPtr, size_t size) { *devPtr = new char[size]; return cudaSuccess; }
inline cudaError_t cudaFree(void* devPtr) { delete[] (char*)devPtr; return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) { memcpy(dst, src, count); return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t error) { return "mock cudaSuccess"; }

// Mock for built-in variables
inline dim3 threadIdx;
inline dim3 blockIdx;
inline dim3 blockDim;
inline dim3 gridDim;

// Mock implementation of gpuAssert
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: mock error %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}
#endif // __NVCC__

// Error checking macro - available to both compilers
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


#endif // CUDA_H
