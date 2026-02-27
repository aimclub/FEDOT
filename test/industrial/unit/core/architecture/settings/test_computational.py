from fedot_ind.core.architecture.settings.computational import BackendMethods, global_imports, default_device
import pytest


@pytest.mark.parametrize('device_type', ['CUDA', 'cpu'])
def test_backend_methods(device_type):
    backend_methods, backend_scipy = BackendMethods(device_type).backend
    assert backend_methods is not None
    assert backend_scipy is not None


def test_global_imports():
    global_imports('scipy')


@pytest.mark.parametrize('device_type', ['CUDA', 'cpu', None])
def test_default_device(device_type):
    default_device(device_type)
