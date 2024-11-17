import torch
import kernel_cuda_extension as kce

def main():

    Mx = 2   #строки X    
    Nx = 4   #столбики X
    
    #кол-во столбцов должно быть равно кол-ву строк
    #Nx и Nw должны быть равны, так как матрица W пока не транспонирована
    #Mw и Mb должны быть равны, так как матрица W пока не транспонирована
    #Обработчик ошибок есть

    Mw = 2   #строки W
    Nw = 4   #столбики W

    Mb = 2   #размер вектора 


    X = torch.rand(Mx, Nx).cuda()
    W = torch.rand(Mw, Nw).cuda()
    B = torch.rand(Mb).cuda()
    Y = torch.empty(Mx, Mw).cuda()

    print("Матрица X:")
    print(X)
    
    print("Матрица W:")
    print(W)
    
    print("Вектор B:")
    print(B)

    try:
        kce.gpu_linear(X, W, B, Y)
        print("Матрица Y:")
        print(Y)
    except RuntimeError as err:
        print(f"Произошла ошибка: {err}")
    
    print("Эталон Y")
    print(torch.matmul(X, W.T) + B)

if __name__ == "__main__":
    main()