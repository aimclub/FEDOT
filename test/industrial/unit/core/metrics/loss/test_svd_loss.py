import torch
import torch.nn as nn
from torch import Tensor

from fedot_ind.core.metrics.loss.svd_loss import HoyerLoss, OrthogonalLoss, SVDLoss


def test_svdloss_initialization():
    loss = SVDLoss()
    assert loss.factor == 1.0


def test_orthogonalloss_forward():
    loss = OrthogonalLoss()
    model = torch.nn.Module()
    U = nn.Parameter(torch.randn(5, 5))
    setattr(model, 'U', U)
    result = loss.forward(model)
    assert isinstance(result, Tensor)


def test_hoyerloss_forward():
    loss = HoyerLoss()
    model = torch.nn.Module()
    U = nn.Parameter(torch.randn(5, 5))
    setattr(model, 'S', U)
    result = loss.forward(model)
    assert isinstance(result, Tensor)
