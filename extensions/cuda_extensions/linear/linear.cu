#include <torch/extension.h>

__global__ void cudaLinear(float* X, float* W, float* B, float* Y, int M, int N, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M)
    {
        for (int j = 0; j < K; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < N; k++)
            {
                sum += X[row * N + k] * W[k * K + j];
            }

            Y[row * K + j] = sum + B[j];
        }
    }
}

// Функция для вызова из Python
void gpu_linear(torch::Tensor input, torch::Tensor weights, torch::Tensor bias, torch::Tensor output) {

    int M = input.size(0); //строки X и Y
    int N = input.size(1); //столбики X и строки W
    int K = weights.size(1); //столбики W и Y

    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    cudaLinear<<<blocksPerGrid, threadsPerBlock>>>(input.data_ptr<float>(), weights.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
	{
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(linear, m)
{
    m.def("gpu_linear", &gpu_linear, "Linear Layer Calculation");
}