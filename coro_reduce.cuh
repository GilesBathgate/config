#ifndef CORO_REDUCE_CUH
#define CORO_REDUCE_CUH

#include "cuda.h"

// Define a block size to be used in the kernel and driver
#define BLOCK_SIZE 256

__global__ void coroReduceKernel(int* g_idata, int n) {
    // Shared memory for the reduction
    __shared__ int sdata[BLOCK_SIZE];

    // Get thread and global indices
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element from global to shared memory
    // Perform a bounds check
    sdata[tid] = (i < n) ? g_idata[i] : 0;

    // Synchronize to make sure all data is loaded into shared memory
    __syncthreads();

    // The reduction loop
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        g_idata[blockIdx.x] = sdata[0];
    }
}

#endif // CORO_REDUCE_CUH
