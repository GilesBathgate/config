#include <iostream>
#include <vector>
#include <numeric> // For std::accumulate

#include "reduce.cuh"


int main() {
    // Setup data
    unsigned int n = 1024;
    std::vector<int> h_idata(n);
    for (unsigned int i = 0; i < n; ++i) {
        h_idata[i] = i;
    }

    // --- Unified CUDA/CPU Code ---
    int* d_idata = nullptr;
    int* d_odata = nullptr;

    // Allocate memory
    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    std::vector<int> h_odata(gridSize);

    gpuErrchk(cudaMalloc((void**)&d_idata, n * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_odata, gridSize * sizeof(int)));

    // Copy data to device
    gpuErrchk(cudaMemcpy(d_idata, h_idata.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
#ifdef __NVCC__
    // The third launch parameter specifies the size of dynamic shared memory
    reduceKernel<blockSize><<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_idata, d_odata, n);
    gpuErrchk(cudaDeviceSynchronize());
#else
    blockDim = dim3(blockSize);
    gridDim = dim3(gridSize);
    // Loop over blocks
    for (int j = 0; j < gridDim.x; ++j) {
        blockIdx.x = j;
        // Inner loop to simulate threads within a block in REVERSE ORDER
        for (int i = blockDim.x - 1; i >= 0; --i) {
            threadIdx.x = i;
            reduceKernel<blockSize>(d_idata, d_odata, n);
        }
    }
#endif

    // Copy result back
    gpuErrchk(cudaMemcpy(h_odata.data(), d_odata, gridSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Final reduction on CPU
    int final_sum = std::accumulate(h_odata.begin(), h_odata.end(), 0);

    // Free memory
    gpuErrchk(cudaFree(d_idata));
    gpuErrchk(cudaFree(d_odata));

    // Verification
    int expected_sum = n * (n - 1) / 2;
    std::cout << "Final Sum: " << final_sum << std::endl;
    std::cout << "Expected Sum: " << expected_sum << std::endl;
    if (final_sum == expected_sum) {
        std::cout << "Test PASSED" << std::endl;
    } else {
        std::cout << "Test FAILED" << std::endl;
    }

    return 0;
}
