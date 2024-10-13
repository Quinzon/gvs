import torch
import imp_cuda_extension

n = 1024
a = torch.randn(n).cuda()
b = torch.randn(n).cuda()
result = torch.zeros(1).cuda()

imp_cuda_extension.gpu_sum(a.data_ptr(), b.data_ptr(), result.data_ptr(), n)

print("Результат:", result.item())
