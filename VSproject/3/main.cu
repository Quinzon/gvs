
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <curand.h>
#include "kernel.cuh"

#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

void generateRandomValues(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

//void matrixVectorAddCPU(const float* X, const float* W, const float* B, float* Y, int M, int N, int K)
//{
//    for (int i = 0; i < M; i++) 
//    {
//        for (int j = 0; j < K; j++) 
//        {
//            float sum = 0.0f;
//
//            for (int k = 0; k < N; k++) 
//            {
//                //sum += X[i * N + k] * W[k * K + j];   //условимся, что матрица W уже транспонирована
//                sum += X[i * N + k] * W[j * K + k];
//            }
//
//            Y[i * K + j] = sum + B[j];
//        }
//    }
//}

//X, W - матрицы размеров Mx на Nx и Mw на Nw соответственно
//B - вектор размером Mb
//Y - результирующая матрица размером Mx на Nw
//Y = X * Wt + B - матрица W не транспонирована
//void matrixVectorAddCPU(const float* X, const float* W, const float* B, float* Y, int Mx, int Nx, int Mw, int Nw, int Mb)
//{
//    for (int i = 0; i < Mx; i++)
//    {
//        for (int j = 0; j < Mx; j++)
//        {
//            float sum = 0.0f;
//
//            for (int k = 0; k < Nw; k++)
//            {
//                //sum += X[i * N + k] * W[k * K + j];   //условимся, что матрица W уже транспонирована
//                sum += X[i * Nw + k] * W[i * Mx + k];
//            }
//
//            Y[i * Mx + j] = sum + B[j];
//        }
//    }
//}

void matrixVectorAddCPU(const float* X, const float* W, const float* B, float* Y, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < N; k++)
            {
                sum += X[i * N + k] * W[j * K + k];
            }

            Y[i * K + j] = sum + B[j];
        }
    }
}

void printMatrix(const float* matrix, int rows, int cols)
{

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            cout << matrix[i * cols + j] << " ";
        }
        cout << "\n";
    }
}

void transposeMatrixDevice(float* input, float* output, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

int main()
{

    setlocale(LC_ALL, "Russian");

    const int Mx = 3;   //строки X    
    const int Nx = 4;   //столбики X
    
    //кол-во столбцов должно быть равно кол-ву строк
    //Nx и Nw должны быть равны, так как матрица W пока не транспонирована
    //Mw и Mb должны быть равны, так как матрица W пока не транспонирована

    const int Mw = 4;   //строки W
    const int Nw = 4;   //столбики W

    const int Mb = 4;   //размер вектора 

    float h_X[Mx * Nx];
    float h_W[Mw * Nw];
    float h_B[Mb];

    //srand(static_cast<unsigned>(time(0)));
    srand(time(nullptr));

    generateRandomValues(h_X, Mx, Nx);

    generateRandomValues(h_W, Mw, Nw);

    generateRandomValues(h_B, 1, Mb);

    float h_Y_GPU[Mx * Mw];
    float h_Y_CPU[Mx * Mw];

    cout << "\nМатрица X:\n";
    printMatrix(h_X, Mx, Nx);

    cout << "\nМатрица W:\n";
    printMatrix(h_W, Mw, Nw);

    cout << "\nВектор B:\n";
    printMatrix(h_B, 1, Mb);

    if ((Nx == Nw) && (Mw == Mb))
    {
        cudaLinear_helper(h_X, h_W, h_B, h_Y_GPU, Mx, Nx, Mw);

        matrixVectorAddCPU(h_X, h_W, h_B, h_Y_CPU, Mx, Nx, Mw);

        cout << "\nРезультат GPU:\n";
        printMatrix(h_Y_GPU, Mx, Mw);

        cout << "\nРезультат CPU:\n";
        printMatrix(h_Y_CPU, Mx, Mw);
    }
    else
    {
        cout << "\nРазмеры не сходятся!\n";
    }

    return 0;
}