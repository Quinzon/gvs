#ifndef __KERNEL_H_INCLUDED__
#define __KERNEL_H_INCLUDED__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREADS_PER_BLOCK 256
void cudaLinear_helper(float* X, float* W, float* B, float* Y, int M, int N, int K);

#endif 