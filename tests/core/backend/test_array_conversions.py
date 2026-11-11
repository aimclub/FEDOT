"""Storage sharing, explicit transfers and ownership at model/data boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fedot.core.backend.array_conversions import as_cupy, as_numpy, as_torch, torch_to_xp
from fedot.core.backend.backend import Backend
from fedot.core.data.tensor_data.tools import to_tensor


pytestmark = pytest.mark.unit


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64, np.bool_])
def test_numpy_and_cpu_torch_share_storage_and_preserve_dtype(dtype):
    array = np.array([0, 1, 0, 1], dtype=dtype)
    tensor = as_torch(array)
    restored = as_numpy(tensor)
    assert tensor.data_ptr() == array.__array_interface__['data'][0]
    assert np.shares_memory(array, restored)
    assert restored.dtype == array.dtype
    np.testing.assert_array_equal(array, restored)


def test_copy_and_dtype_requests_do_not_modify_the_source():
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    tensor = as_torch(array, dtype=torch.float32, copy=True)
    assert tensor.dtype == torch.float32
    tensor[0, 0] = -1
    assert array[0, 0] == 0
    independent = as_numpy(as_torch(array), copy=True)
    assert not np.shares_memory(array, independent)
    independent[0, 0] = -2
    assert array[0, 0] == 0
    narrowed = as_numpy(as_torch(array), dtype=np.float32)
    assert narrowed.dtype == np.float32


def test_explicit_tensor_copy_and_contiguous_preserve_values():
    source = torch.arange(12.).reshape(3, 4).t()
    assert not source.is_contiguous()
    shared = as_torch(source)
    assert shared.data_ptr() == source.data_ptr()
    assert shared.stride() == source.stride()
    contiguous = as_torch(source, contiguous=True)
    independent = as_torch(source, copy=True)
    assert contiguous.is_contiguous()
    assert independent.data_ptr() != source.data_ptr()
    torch.testing.assert_close(contiguous, source)


def test_torch_detaches_for_libraries_and_can_preserve_extension_autograd():
    source = torch.arange(4., requires_grad=True)
    detached = as_torch(source)
    assert not detached.requires_grad
    assert detached.data_ptr() == source.data_ptr()
    private = as_torch(source, copy=True, detach=False)
    assert private.requires_grad and private.data_ptr() != source.data_ptr()
    private.sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))
    np.testing.assert_array_equal(as_numpy(source), np.arange(4.))


def test_none_values_and_explicit_preprocessing_module():
    assert as_numpy(None) is as_torch(None) is as_cupy(None) is None
    source = torch.arange(4.)
    assert np.shares_memory(torch_to_xp(source, np), as_numpy(source))
    with pytest.raises(TypeError, match='Expected torch.Tensor'):
        torch_to_xp(np.arange(4), np)
    with pytest.raises(ValueError, match='Unsupported xp backend'):
        torch_to_xp(source, SimpleNamespace(__name__='other'))


def test_shared_helpers_ignore_global_backend_and_legacy_creator_keeps_ownership():
    backend = Backend()
    with backend.override('cpu'):
        source = np.arange(4, dtype=np.float32)
        old_style = to_tensor(source)
        explicit = to_tensor(source, dtype=torch.float32)
        assert old_style.dtype == torch.float64
        assert explicit.dtype == torch.float32
        explicit[0] = -1
        assert source[0] == 0
        assert as_torch(source).dtype == torch.float32
        prepared_objects = np.array([[1., 0.], [0., 1.]], dtype=object)
        normalized = to_tensor(prepared_objects, dtype=torch.float32)
        torch.testing.assert_close(normalized, torch.eye(2))


@pytest.mark.parametrize('source_kind', ['numpy', 'torch_cpu', 'torch_cuda', 'cupy'])
def test_cuda_array_exchange_and_explicit_cpu_output(source_kind):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    cupy = pytest.importorskip('cupy')
    numpy_source = np.arange(12, dtype=np.float32).reshape(3, 4)
    sources = {'numpy': numpy_source, 'torch_cpu': as_torch(numpy_source),
               'torch_cuda': as_torch(numpy_source, device='cuda:0'), 'cupy': cupy.asarray(numpy_source)}
    array = as_cupy(sources[source_kind], device='cuda:0')
    tensor = as_torch(array)
    assert tensor.device == torch.device('cuda:0')
    assert tensor.data_ptr() == array.data.ptr
    assert tensor.requires_grad is False
    assert as_cupy(tensor).data.ptr == tensor.data_ptr()
    cpu = as_torch(array, device='cpu')
    assert cpu.device.type == 'cpu'
    np.testing.assert_array_equal(as_numpy(cpu), numpy_source)
    np.testing.assert_array_equal(as_numpy(array), numpy_source)
    private = as_cupy(array, copy=True)
    assert private.data.ptr != array.data.ptr
    private_tensor = as_torch(array, copy=True)
    assert private_tensor.data_ptr() != array.data.ptr


def test_cuda_dtype_contiguity_and_torch_gradients_at_dlpack_boundary():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    cupy = pytest.importorskip('cupy')
    source = torch.arange(12., device='cuda:0', requires_grad=True).reshape(3, 4).t()
    shared = as_cupy(source)
    assert shared.data.ptr == source.data_ptr()
    contiguous = as_cupy(source, dtype=cupy.float64, contiguous=True)
    assert contiguous.flags.c_contiguous and contiguous.dtype == cupy.float64
    tensor = as_torch(contiguous, dtype=torch.float32, contiguous=True)
    assert tensor.is_contiguous() and tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, source.detach())
    private = as_cupy(source, copy=True)
    assert private.data.ptr != source.data_ptr()
    with pytest.raises(ValueError, match='CUDA device'):
        as_cupy(source, device='cpu')
