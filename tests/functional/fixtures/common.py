import pytest
import torch


@pytest.fixture
def pair_random_cuda_tensors(request):
    size = request.param
    a = torch.randn(size).cuda()
    b = torch.randn(size).cuda()
    return a, b
