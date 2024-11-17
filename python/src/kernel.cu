#include <torch/extension.h>
#include <stdexcept>
#include <sstream>


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
				sum += X[row * N + k] * W[j * N + k];
            }

            Y[row * K + j] = sum + B[j];
        }
    }
}

// Функция для вызова из Python
void gpu_linear(torch::Tensor input, torch::Tensor weights, torch::Tensor bias, torch::Tensor output)
{

	int Mx = input.size(0);   //строки X    
    int Nx = input.size(1);   //столбики X
    
    //кол-во столбцов должно быть равно кол-ву строк
    //Nx и Nw должны быть равны, так как матрица W пока не транспонирована
    //Mw и Mb должны быть равны, так как матрица W пока не транспонирована

    int Mw = weights.size(0);   //строки W
    int Nw = weights.size(1);   //столбики W

	int Mb = bias.size(0);   //размер вектора 

    int threadsPerBlock = 256;
    int blocksPerGrid = (Mx + threadsPerBlock - 1) / threadsPerBlock;	
	
	//Проверка размеров тензоров
    if (Nx != Nw)
	{
        std::stringstream ss;
        ss << "First matrix columns (" << Nx << ") must match number of columns in second matrix (" << Nw << ")";
        throw std::runtime_error(ss.str());
    }
	
    if (Mw != Mb)
	{
        std::stringstream ss;
        ss << "Second matrix rows (" << Mw << ") must match size of a vector (" << Mb << ")";
        throw std::runtime_error(ss.str());
    }
	
    if (Mw != output.size(1))
	{
        std::stringstream ss;
        ss << "Output matrix columns (" << output.size(1) << ") must match number of rows in second matrix (" << Mw << ")";
        throw std::runtime_error(ss.str());
    }

	cudaLinear<<<blocksPerGrid, threadsPerBlock>>>(input.data_ptr<float>(), weights.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), Mx, Nx, Mw);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::stringstream ss;
		ss << "CUDA error: " << cudaGetErrorString(err);
		throw std::runtime_error(ss.str());
	}
	
}

PYBIND11_MODULE(kernel_cuda_extension, m)
{
    m.def("gpu_linear", &gpu_linear, "Linear Layer Calculation");
}