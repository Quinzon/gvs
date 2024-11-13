import pytest
import torch
from cuda_extensions.linear import gpu_linear
from tests.functional.testdata.common import matrix_dimensions, tolerance


@pytest.mark.parametrize("m, n, k", matrix_dimensions)
def test_gpu_linear(m, n, k):
    x = torch.rand(m, n).cuda()
    w = torch.rand(n, k).cuda()
    b = torch.rand(k).cuda()
    y = torch.empty(m, k).cuda()

    gpu_linear(x, w, b, y)

    output_torch = torch.mm(x, w) + b

    abs_error = torch.abs(y - output_torch).max().item()
    rel_error = abs_error / torch.abs(output_torch).max().item()

    assert rel_error < tolerance


@pytest.mark.parametrize("invalid_input", [0, "invalid", None])
def test_gpu_linear_invalid_input(invalid_input):
    w = torch.rand(64, 16).cuda()
    b = torch.rand(16).cuda()
    y = torch.empty(32, 16).cuda()

    with pytest.raises(Exception):
        gpu_linear(invalid_input, w, b, y)

    x = torch.rand(32, 64).cuda()
    with pytest.raises(Exception):
        gpu_linear(x, invalid_input, b, y)

    with pytest.raises(Exception):
        gpu_linear(x, w, invalid_input, y)

    with pytest.raises(Exception):
        gpu_linear(x, w, b, invalid_input)
