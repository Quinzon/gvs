import pytest
import torch
from cuda_extensions.linear import gpu_linear
from tests.functional.testdata.common import matrix_dimensions, tolerance


@pytest.mark.parametrize("m, n, k", matrix_dimensions)
def test_gpu_linear_correctness(m, n, k):
    x = torch.rand(m, n).cuda()
    w = torch.rand(k, n).cuda()
    b = torch.rand(k).cuda()
    y = torch.empty(m, k).cuda()

    gpu_linear(x, w, b, y)

    w_transposed = w.transpose(0, 1)
    output_torch = torch.mm(x, w_transposed) + b

    abs_error = torch.abs(y - output_torch).max().item()
    rel_error = abs_error / torch.abs(output_torch).max().item()

    assert rel_error < tolerance, f"Relative error {rel_error} exceeded tolerance {tolerance}"


@pytest.mark.parametrize("invalid_input", [0, "invalid", None])
def test_gpu_linear_invalid_input(invalid_input):
    w = torch.rand(16, 64).cuda()
    b = torch.rand(16).cuda()
    y = torch.empty(32, 16).cuda()

    with pytest.raises(TypeError):
        gpu_linear(invalid_input, w, b, y)

    x = torch.rand(32, 64).cuda()

    with pytest.raises(TypeError):
        gpu_linear(x, invalid_input, b, y)

    with pytest.raises(TypeError):
        gpu_linear(x, w, invalid_input, y)

    with pytest.raises(TypeError):
        gpu_linear(x, w, b, invalid_input)


@pytest.mark.parametrize(
    "x_shape, w_shape, b_shape, y_shape",
    [
        # Несоответствие: число столбцов X и строк W
        ((32, 64), (32, 64), (16,), (32, 16)),
        # Несоответствие: число строк W и размер вектора B
        ((32, 64), (16, 64), (8,), (32, 16)),
        # Несоответствие: число столбцов W и выходного числа столбцов Y
        ((32, 64), (16, 64), (16,), (32, 8)),
    ]
)
def test_gpu_linear_dimension_mismatch(x_shape, w_shape, b_shape, y_shape):
    """
    Тест на несовпадение размеров входных данных с параметризацией.
    """
    x = torch.rand(*x_shape).cuda()
    w = torch.rand(*w_shape).cuda()
    b = torch.rand(*b_shape).cuda()
    y = torch.empty(*y_shape).cuda()

    with pytest.raises(RuntimeError):
        gpu_linear(x, w, b, y)
