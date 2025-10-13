#include "reduce.cuh"
#include "mock_cub.h"

template<int BLOCK_SIZE>
__global__ void reduceKernel(const int* g_idata, int* g_odata, unsigned int n) {
    // Get thread index
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data for this thread
    int thread_data = (i < n) ? g_idata[i] : 0;

    // Allocate a CUB block reduction object
    cub::BlockReduce<int, BLOCK_SIZE> block_reduce;

    // Perform the reduction
    int block_sum = block_reduce.Sum(thread_data);

    // The first thread writes the result for this block
    if (threadIdx.x == 0) {
        g_odata[blockIdx.x] = block_sum;
    }
}

// Explicitly instantiate the template for the block size we are using.
// This is what allows the implementation to be in a .cu file.
template __global__ void reduceKernel<256>(const int* g_idata, int* g_odata, unsigned int n);
