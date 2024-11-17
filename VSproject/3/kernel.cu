#include "kernel.cuh"

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
                //sum += X[row * N + k] * W[k * K + j]; //Условимся, что матрица W уже транспонирована
                sum += X[row * N + k] * W[j * K + k];
            }

            Y[row * K + j] = sum + B[j];
        }
    }
}

void cudaLinear_helper(float* X, float* W, float* B, float* Y, int M, int N, int K)
{
    float* d_X, * d_W, * d_B, * d_Y;

    cudaMalloc((void**)&d_X, M * N * sizeof(float));
    cudaMalloc((void**)&d_W, N * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * sizeof(float));
    cudaMalloc((void**)&d_Y, M * K * sizeof(float));


    cudaMemcpy(d_X, X, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * sizeof(float), cudaMemcpyHostToDevice);


    //int threadsPerBlock = 256;
    int blocksPerGrid = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaLinear <<< blocksPerGrid, THREADS_PER_BLOCK >>> (d_X, d_W, d_B, d_Y, M, N, K);

    cudaMemcpy(Y, d_Y, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    //delete[] d_W_transposed;
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_B);
    cudaFree(d_Y);
}
