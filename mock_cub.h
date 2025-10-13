#ifndef MOCK_CUB_H
#define MOCK_CUB_H

#include "cuda.h"

#ifdef __NVCC__
// If compiling with nvcc, include the real CUB library.
#include <cub/cub.cuh>

#else
// If compiling with g++, provide a mock implementation.
namespace cub {

template<typename T, int BLOCK_SIZE>
class BlockReduce {
    // This static storage is the key to the reverse iteration solution.
    // It will persist across the sequential calls to Sum() within a single block simulation.
    static T temp_storage[BLOCK_SIZE];

public:
    T Sum(T input) {
        // Each thread places its data into the shared storage.
        temp_storage[threadIdx.x] = input;

        // The LAST thread to run in the reverse loop (i.e., threadIdx.x == 0)
        // will perform the reduction and store the final result.
        if (threadIdx.x == 0) {
            T total = 0;
            for (unsigned int i = 0; i < blockDim.x; i++) {
                total += temp_storage[i];
            }
            temp_storage[0] = total;
        }

        // All threads return the value from the first element.
        // In the reverse loop, by the time thread 0 runs and returns,
        // the correct sum will have been placed there.
        return temp_storage[0];
    }
};

// Definition for the static member of the template class
template<typename T, int BLOCK_SIZE>
T BlockReduce<T, BLOCK_SIZE>::temp_storage[BLOCK_SIZE];

} // namespace cub
#endif // __NVCC__

#endif // MOCK_CUB_H
