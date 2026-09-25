import pytest

from fedot.core.operations.evaluation.operation_implementations.data_operations.topological. \
    topological_backend_rules import TopologicalBackend, resolve_topological_backend


@pytest.mark.parametrize(('gph_available', 'ripser_available', 'expected_backend'),
                         [(True, True, TopologicalBackend.GPH),
                          (True, False, TopologicalBackend.GPH),
                          (False, True, TopologicalBackend.RIPSER),
                          (False, False, None)])
def test_resolve_topological_backend(gph_available, ripser_available, expected_backend):
    resolution = resolve_topological_backend(gph_available=gph_available,
                                             ripser_available=ripser_available)

    assert resolution.backend is expected_backend
    assert resolution.is_available is (expected_backend is not None)
