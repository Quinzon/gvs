import torch
import imp_cuda_extension


def test_gpu_sum(size: int) -> None:
    a = torch.randn(size).cuda()
    b = torch.randn(size).cuda()

    result_cuda = torch.zeros(1).cuda()
    imp_cuda_extension.gpu_sum(a, b, result_cuda)
    torch.cuda.synchronize()

    result_torch = torch.sum(a + b)

    abs_error = abs(result_cuda.item() - result_torch.item())
    rel_error = abs_error / abs(result_torch.item()) if result_torch.item() != 0 else 0

    print(f"Тест для размера {size}:")
    print(f"  Результат CUDA: {result_cuda.item()}")
    print(f"  Результат Torch: {result_torch.item()}")
    print(f"  Абсолютная погрешность: {abs_error}")
    print(f"  Относительная погрешность: {rel_error}")
    print()


if __name__ == '__main__':
    test_sizes = [1, 10, 100, 1024, 10240, 102400, 1024000, 10240000]

    for test_size in test_sizes:
        test_gpu_sum(test_size)
