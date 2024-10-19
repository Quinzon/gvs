import pytest
import torch
import imp_cuda_extension

from tests.functional.testdata.common import test_sizes, tolerance


@pytest.mark.parametrize("pair_random_cuda_tensors", test_sizes, indirect=True)
def test_gpu_sum(pair_random_cuda_tensors):
    a, b = pair_random_cuda_tensors

    result_cuda = torch.zeros(1).cuda()
    imp_cuda_extension.gpu_sum(a, b, result_cuda)

    result_torch = torch.dot(a, b)

    abs_error = abs(result_cuda.item() - result_torch.item())
    rel_error = abs_error / abs(result_torch.item())

    assert rel_error < tolerance
