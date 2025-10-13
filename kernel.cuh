#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "cuda.h"

__global__ void vectorAddKernel(const int* a, const int* b, int* c, int n);

#endif // KERNEL_CUH
