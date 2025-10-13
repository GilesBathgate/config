#ifndef REDUCE_CUH
#define REDUCE_CUH

#include "cuda.h"

template<int BLOCK_SIZE>
__global__ void reduceKernel(const int* g_idata, int* g_odata, unsigned int n);

#endif // REDUCE_CUH
