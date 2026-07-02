import pytest
import torch

from fedot.core.backend.backend import Backend
from fedot.validation.errors import FedotValidationError


@pytest.mark.unit
def test_backend_normalize_name_accepts_cuda_device():
    assert Backend.normalize_name('cuda:0') == 'cuda:0'
    assert Backend.normalize_name(' CUDA:2 ') == 'cuda:2'


@pytest.mark.unit
def test_backend_normalize_name_accepts_mps():
    assert Backend.normalize_name('mps') == 'mps'
    assert Backend.normalize_name(' MPS ') == 'mps'


@pytest.mark.unit
def test_backend_normalize_name_prefers_cpu_over_gpu_aliases():
    assert Backend.normalize_name('cpu') == 'cpu'
    assert Backend.is_gpu_name('cpu') is False


@pytest.mark.unit
def test_backend_normalize_name_keeps_explicit_cuda_separate_from_generic_gpu():
    assert Backend.normalize_name('gpu') == 'gpu'
    assert Backend.normalize_name('cuda') == 'cuda'


@pytest.mark.unit
def test_backend_torch_device_for_name_maps_mps():
    assert str(Backend.torch_device_for_name('mps')) == 'mps'


@pytest.mark.unit
def test_backend_torch_device_for_name_maps_cuda_device(monkeypatch):
    monkeypatch.setattr(Backend, 'is_cuda_stack_compatible', lambda: True)
    assert str(Backend.torch_device_for_name('cuda:1')) == 'cuda:1'


@pytest.mark.unit
def test_backend_torch_device_for_name_resolves_generic_gpu_to_cuda_when_available(monkeypatch):
    monkeypatch.setattr(Backend, 'is_cuda_stack_compatible', lambda: True)
    monkeypatch.setattr(Backend, 'is_mps_compatible', lambda: True)
    assert str(Backend.torch_device_for_name('gpu')) == 'cuda'


@pytest.mark.unit
def test_backend_torch_device_for_name_resolves_generic_gpu_to_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(Backend, 'is_cuda_stack_compatible', lambda: False)
    monkeypatch.setattr(Backend, 'is_mps_compatible', lambda: True)
    assert str(Backend.torch_device_for_name('gpu')) == 'mps'


@pytest.mark.unit
def test_backend_detect_compatible_gpu_backend_prefers_cuda_stack(monkeypatch):
    monkeypatch.setattr(Backend, 'is_cuda_stack_compatible', lambda: True)
    monkeypatch.setattr(Backend, 'is_mps_compatible', lambda: True)
    assert Backend.detect_compatible_gpu_backend() == 'gpu'


@pytest.mark.unit
def test_backend_detect_compatible_gpu_backend_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr(Backend, 'is_cuda_stack_compatible', lambda: False)
    monkeypatch.setattr(Backend, 'is_mps_compatible', lambda: True)
    assert Backend.detect_compatible_gpu_backend() == 'mps'


@pytest.mark.unit
def test_backend_detect_compatible_gpu_backend_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(Backend, 'is_cuda_stack_compatible', lambda: False)
    monkeypatch.setattr(Backend, 'is_mps_compatible', lambda: False)
    with pytest.raises(RuntimeError, match='No FEDOT-compatible GPU backend'):
        Backend.detect_compatible_gpu_backend()


@pytest.mark.unit
def test_backend_explicit_cuda_requires_cuda_stack(monkeypatch):
    monkeypatch.setattr(Backend, 'is_cuda_stack_compatible', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA stack is not available'):
        Backend.torch_device_for_name('cuda')


@pytest.mark.unit
def test_backend_normalize_name_rejects_unknown_value():
    with pytest.raises(FedotValidationError, match='Unsupported backend name'):
        Backend.normalize_name('tpu')


@pytest.mark.unit
def test_backend_normalize_name_rejects_empty_string():
    with pytest.raises(FedotValidationError, match='must be a non-empty string'):
        Backend.normalize_name('   ')


@pytest.mark.unit
def test_backend_normalize_name_rejects_non_string():
    with pytest.raises(FedotValidationError, match='must be a non-empty string'):
        Backend.normalize_name(123)


@pytest.mark.unit
def test_backend_set_mps_backend(monkeypatch):
    backend = Backend()
    backend.reset()
    monkeypatch.setattr(Backend, 'is_mps_compatible', lambda: True)

    backend.set('mps')

    assert backend.name == 'mps'
    assert backend.xp.__name__ == 'numpy'
    assert backend.pd.__name__ == 'pandas'
    assert str(backend.device) == 'mps'

    backend.reset()
    backend.set('cpu')


@pytest.mark.unit
def test_backend_set_generic_gpu_uses_mps_when_cuda_stack_unavailable(monkeypatch):
    backend = Backend()
    backend.reset()
    monkeypatch.setattr(Backend, 'is_cuda_stack_compatible', lambda: False)
    monkeypatch.setattr(Backend, 'is_mps_compatible', lambda: True)

    backend.set('gpu')

    assert backend.name == 'mps'
    assert str(backend.device) == 'mps'

    backend.reset()
    backend.set('cpu')
