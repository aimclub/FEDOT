import pytest

from fedot.core.backend.backend import Backend


@pytest.mark.unit
def test_backend_normalize_name_accepts_cuda_device():
    assert Backend.normalize_name('cuda:0') == 'cuda:0'
    assert Backend.normalize_name(' CUDA:2 ') == 'cuda:2'


@pytest.mark.unit
def test_backend_normalize_name_prefers_cpu_over_gpu_aliases():
    assert Backend.normalize_name('cpu') == 'cpu'
    assert Backend.is_gpu_name('cpu') is False


@pytest.mark.unit
def test_backend_torch_device_for_name_maps_gpu_aliases_to_default_cuda():
    assert str(Backend.torch_device_for_name('gpu')) == 'cuda'
    assert str(Backend.torch_device_for_name('cuda:1')) == 'cuda:1'


@pytest.mark.unit
def test_backend_normalize_name_rejects_unknown_value():
    with pytest.raises(ValueError, match='Unsupported backend name'):
        Backend.normalize_name('tpu')
