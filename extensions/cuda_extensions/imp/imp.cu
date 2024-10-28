#include <torch/extension.h>

__global__ void GPUimplementation(float *a, float *b, float *result, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i < n) {
        sdata[tid] = a[i] * b[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// Функция для вызова из Python
void gpu_sum(torch::Tensor a, torch::Tensor b, torch::Tensor result) {
    int n = a.size(0);

    // Параметры блока и сетки
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Вызов CUDA-ядра
    GPUimplementation<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// Регистрация функции в модуле
PYBIND11_MODULE(imp, m) {
    m.def("gpu_sum", &gpu_sum, "Sum of two tensors");
}
