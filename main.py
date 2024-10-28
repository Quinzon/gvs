import torch
from cuda_extensions import linear

def main():
    M = 4   #строки X и Y
    N = 3   #столбики X и строки W
    K = 2   #столбики W и Y

    X = torch.rand(M, N).cuda()
    W = torch.rand(N, K).cuda()
    B = torch.rand(K).cuda()
    Y = torch.empty(M, K).cuda()

    linear.gpu_linear(X, W, B, Y)

    print("Output Y:")
    print(Y)

if __name__ == "__main__":
    main()
