import torch
import pytest

from fedot.core.operations.evaluation.operation_implementations.rules import (
    SpectrumNComponentsMethod,
)
from fedot.core.operations.evaluation.operation_implementations.tools import (
    broken_stick_expectations,
    n_components_from_broken_stick,
    n_components_from_elbow,
    resolve_spectrum_n_components,
)
from fedot.validation.errors import FedotValidationError


@pytest.mark.unit
def test_broken_stick_expectations_sum_to_one():
    n = 5
    b = broken_stick_expectations(n)
    assert b.shape == (n,)
    assert torch.allclose(b.sum(), torch.tensor(1.0), atol=1e-6)
    # First piece is the largest share
    assert torch.all(b[:-1] >= b[1:])


@pytest.mark.unit
def test_broken_stick_expectations_rejects_non_positive_n():
    with pytest.raises(FedotValidationError, match='n >= 1'):
        broken_stick_expectations(0)


@pytest.mark.unit
def test_n_components_from_broken_stick_keeps_dominant_leading():
    # Strong first two components, then noise-level shares
    props = torch.tensor([0.55, 0.30, 0.05, 0.05, 0.05], dtype=torch.float32)
    k = n_components_from_broken_stick(props)
    assert k == 2


@pytest.mark.unit
def test_n_components_from_elbow_on_clear_knee():
    # Sharp drop after the 3rd singular value; max chord-distance is at index 3 → k=4
    spectrum = torch.tensor([10.0, 9.0, 8.0, 1.0, 0.5, 0.2], dtype=torch.float32)
    k = n_components_from_elbow(spectrum)
    assert k == 4


@pytest.mark.unit
def test_spectrum_selectors_handle_degenerate_inputs():
    assert n_components_from_elbow(torch.tensor([1.0])) == 1
    assert n_components_from_broken_stick(torch.tensor([1.0])) == 1
    assert n_components_from_elbow(torch.zeros(4)) == 1
    assert n_components_from_broken_stick(torch.zeros(4)) == 1


@pytest.mark.unit
def test_resolve_spectrum_uses_enum_mapping():
    singular_values = torch.tensor([10.0, 9.0, 8.0, 1.0, 0.5, 0.2], dtype=torch.float32)
    k = resolve_spectrum_n_components(
        SpectrumNComponentsMethod.ELBOW,
        singular_values=singular_values,
        max_components=6,
    )
    assert k == 4


@pytest.mark.unit
def test_resolve_spectrum_rejects_unknown_method_and_missing_input():
    with pytest.raises(FedotValidationError, match='Unsupported spectrum'):
        resolve_spectrum_n_components('nope', singular_values=torch.ones(3), max_components=3)

    with pytest.raises(FedotValidationError, match='requires singular_values or proportions'):
        resolve_spectrum_n_components('elbow', max_components=3)
