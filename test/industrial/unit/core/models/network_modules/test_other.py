import numpy as np
from torch import Tensor, nn

from fedot_ind.core.models.nn.network_modules.other import correct_sizes, pass_through, if_module_to_torchscript


def test_correct_sizes():
    corrected = correct_sizes(sizes=(4, 1, 8))
    assert corrected == [3, 1, 7]


def test_pass_through():
    tensor = Tensor([1, 2, 3])
    assert pass_through(tensor) is tensor


def test_if_module_to_torchscript():
    module = nn.Linear(4, 11)
    tensor = Tensor(np.random.rand(11, 11, 4))
    assert if_module_to_torchscript(m=module,
                                    inputs=tensor,
                                    script=True,
                                    verbose=True,
                                    serialize=True)
