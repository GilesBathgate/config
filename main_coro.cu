#include "coro_reduce.cuh"
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <cmath>

// Host function to verify the result
void verify_result(int* data, int n, int num_blocks) {
    long long computed_sum = 0;
    for(int i = 0; i < num_blocks; ++i) {
        computed_sum += data[i];
    }

    long long expected_sum = 0;
    for(int i = 0; i < n; ++i) {
        expected_sum += i;
    }

    std::cout << "Expected sum: " << expected_sum << std::endl;
    std::cout << "Computed sum: " << computed_sum << std::endl;
    assert(computed_sum == expected_sum);
    std::cout << "Success!" << std::endl;
}


#ifdef __NVCC__
// =================================================================
// NVCC (Real GPU) Path
// =================================================================
int main() {
    const int n = BLOCK_SIZE; // Use a single block for simplicity
    const int num_blocks = 1;

    std::vector<int> h_data(n);
    std::iota(h_data.begin(), h_data.end(), 0);

    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);

    std::cout << "--- NVCC Path ---" << std::endl;
    coroReduceKernel<<<grid, block>>>(d_data, n);

    // Check for kernel errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    std::vector<int> h_result(num_blocks);
    cudaMemcpy(h_result.data(), d_data, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    verify_result(h_result.data(), n, num_blocks);

    return 0;
}

#else
// =================================================================
// G++ (CPU Coroutine Simulation) Path
// =================================================================
int main() {
    const int n = BLOCK_SIZE; // Use a single block for simplicity
    const int num_blocks = 1;

    std::vector<int> h_data(n);
    std::iota(h_data.begin(), h_data.end(), 0);

    int* g_idata = h_data.data();

    std::cout << "--- G++ Coroutine Path ---" << std::endl;

    // --- Simulate a single block ---
    std::vector<Task<int*, int>*> tasks;
    for(int i = 0; i < BLOCK_SIZE; ++i) {
        tasks.push_back(new Task(&coroReduceKernel));
    }

    // Set up mock CUDA variables for the single block
    blockIdx.x = 0;
    blockDim.x = BLOCK_SIZE;

    // Start the tasks
    for(int i = 0; i < BLOCK_SIZE; ++i) {
        threadIdx.x = i;
        tasks[i]->start(g_idata, n);
    }

    // Drive tasks to completion
    bool all_done = false;
    while(!all_done) {
        all_done = true;
        for(auto t: tasks) {
            if (!t->done()) {
                t->resume();
                all_done = false;
            }
        }
    }

    // Cleanup
    for (auto t : tasks) {
        delete t;
    }

    verify_result(g_idata, n, num_blocks);

    return 0;
}
#endif
