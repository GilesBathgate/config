#include <iostream>
#include <vector>

#include "kernel.cuh"


int main() {
    // Driver code
    int n = 5;
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {10, 20, 30, 40, 50};
    std::vector<int> c(n);

    // --- Unified CUDA/CPU Code ---

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    // Allocate "device" memory (real GPU or mock CPU)
    gpuErrchk(cudaMalloc((void**)&dev_a, n * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_b, n * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_c, n * sizeof(int)));

    // Copy data from "host" to "device"
    gpuErrchk(cudaMemcpy(dev_a, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_b, b.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // --- Kernel Launch (the only part that differs) ---
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

#ifdef __NVCC__
    vectorAddKernel<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, n);
    gpuErrchk(cudaDeviceSynchronize());
#else
    blockDim = dim3(blockSize);
    gridDim = dim3(gridSize);

    for (int j = 0; j < gridDim.x; ++j) {
        blockIdx.x = j;
        for (int i = 0; i < blockDim.x; ++i) {
            threadIdx.x = i;
            vectorAddKernel(dev_a, dev_b, dev_c, n);
        }
    }
#endif

    // Copy result back from "device" to "host"
    gpuErrchk(cudaMemcpy(c.data(), dev_c, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Free "device" memory
    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(dev_c));

    std::cout << "Result:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
