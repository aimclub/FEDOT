import torch
import torch.nn as nn
from fedot_ind.core.models.nn.network_modules.losses import lambda_prepare, ExpWeightedLoss, HuberLoss, \
    LogCoshLoss, MaskedLossWrapper, CenterLoss, FocalLoss, TweedieLoss, SMAPELoss, RMSELoss


def test_lambda_prepare_int():
    val = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lambda_ = 5
    result = lambda_prepare(val, lambda_)
    assert torch.allclose(result, torch.tensor([[5.0, 5.0]]))


def test_lambda_prepare_list():
    val = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lambda_ = [1, 2]
    result = lambda_prepare(val, lambda_)
    assert torch.allclose(result, torch.tensor([[1.0, 2.0]]))


def test_lambda_prepare_tensor():
    val = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lambda_ = torch.tensor([3.0, 4.0])
    result = lambda_prepare(val, lambda_)
    assert torch.allclose(result, torch.tensor([3.0, 4.0]))


def test_exp_weighted_loss():
    time_steps = 5
    tolerance = 0.1
    loss_fn = ExpWeightedLoss(time_steps, tolerance)

    input_ = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    target = torch.tensor(
        [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5], [9.5, 10.5]])

    loss = loss_fn(input_, target)
    assert torch.isclose(loss, torch.tensor(1.357), atol=1e-3)


def test_huber_loss_mean():
    loss_fn = HuberLoss(reduction='mean', delta=1.0)

    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(input_, target)
    assert torch.isclose(loss, torch.tensor(0.125))


def test_huber_loss_sum():
    loss_fn = HuberLoss(reduction='sum', delta=1.0)

    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(input_, target)
    assert torch.isclose(loss, torch.tensor(0.375))


def test_huber_loss_none():
    loss_fn = HuberLoss(reduction='none', delta=1.0)

    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(input_, target)
    assert torch.allclose(loss, torch.tensor([0.125]))


def test_log_cosh_loss_mean():
    loss_fn = LogCoshLoss(reduction='mean', delta=1.0)

    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(input_, target)
    assert torch.isclose(loss, torch.tensor(0.12), atol=1e-3)


def test_log_cosh_loss_sum():
    loss_fn = LogCoshLoss(reduction='sum', delta=1.0)

    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(input_, target)
    assert torch.isclose(loss, torch.tensor(0.36), atol=1e-3)


def test_log_cosh_loss_none():
    loss_fn = LogCoshLoss(reduction='none', delta=1.0)

    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(input_, target)
    assert torch.allclose(loss, torch.tensor([0.12]), atol=1e-3)


def test_masked_loss_wrapper():
    loss_fn = nn.MSELoss()
    masked_loss_fn = MaskedLossWrapper(loss_fn)

    input_ = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.5, float('nan')], [3.5, 4.5]])

    loss = masked_loss_fn(input_, target)
    assert torch.isclose(loss, torch.tensor(0.25))


def test_center_loss():
    c_out = 3
    logits_dim = 2
    loss_fn = CenterLoss(c_out, logits_dim)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels = torch.tensor([0, 1, 2])

    loss = loss_fn(x, labels)
    assert torch.isclose(loss, torch.tensor(39.24), atol=100)


# def test_center_plus_loss():
#     c_out = 3
#     logits_dim = 2
#     loss_fn = nn.CrossEntropyLoss()
#     center_plus_loss_fn = CenterPlusLoss(
#         loss_fn, c_out, λ=0.1, logits_dim=logits_dim)
#
#     x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#     labels = torch.tensor([0, 1, 2])
#
#     loss = center_plus_loss_fn(x, labels)
#     assert torch.isclose(loss, torch.tensor(3.3133), atol=1e-4)


def test_focal_loss_mean():
    alpha = torch.tensor([0.25, 0.75])
    gamma = 2.0
    reduction = 'mean'
    loss_fn = FocalLoss(alpha, gamma, reduction)

    x = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    y = torch.tensor([0, 1])

    loss = loss_fn(x, y)
    assert torch.isclose(loss, torch.tensor(0.037), atol=1e-3)


def test_focal_loss_sum():
    alpha = None
    gamma = 1.5
    reduction = 'sum'
    loss_fn = FocalLoss(alpha, gamma, reduction)

    x = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    y = torch.tensor([0, 1])

    loss = loss_fn(x, y)
    assert torch.isclose(loss, torch.tensor(0.222), atol=1e-3)


def test_focal_loss_none():
    alpha = torch.tensor([0.5, 0.5])
    gamma = 1.0
    reduction = 'none'
    loss_fn = FocalLoss(alpha, gamma, reduction)

    x = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    y = torch.tensor([0, 1])

    loss = loss_fn(x, y)
    assert torch.allclose(loss, torch.tensor([0.077, 0.102]), atol=1e-2)


def test_tweedie_loss_p1_5():
    p = 1.5
    eps = 1e-8
    loss_fn = TweedieLoss(p, eps)

    inp = torch.tensor([1.0, 2.0, 3.0])
    targ = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(inp, targ)
    assert torch.isclose(loss, torch.tensor(6.289), atol=1e-3)


def test_tweedie_loss_p1_8():
    p = 1.8
    eps = 1e-8
    loss_fn = TweedieLoss(p, eps)

    inp = torch.tensor([1.0, 2.0, 3.0])
    targ = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(inp, targ)
    torch.isclose(loss, torch.tensor(7.486), atol=1e-3)


def test_smape_loss():
    loss_fn = SMAPELoss()

    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(input_, target)
    assert torch.isclose(loss, torch.tensor(25.868), atol=1e-3)


def test_rmse_loss():
    loss_fn = RMSELoss()

    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(input_, target)
    assert torch.isclose(loss, torch.tensor(0.5))
