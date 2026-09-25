import logging

from fedot.core.operations.evaluation.operation_implementations.data_operations.topological. \
    topological_backend_rules import TopologicalBackend, resolve_topological_backend

try:
    from gph import ripser_parallel as _gph_ripser
except ModuleNotFoundError:
    _gph_ripser = None

try:
    from ripser import ripser as _ripser
except ModuleNotFoundError:
    _ripser = None


TOPOLOGICAL_BACKEND = resolve_topological_backend(gph_available=_gph_ripser is not None,
                                                  ripser_available=_ripser is not None)
TOPOLOGICAL_BACKEND_AVAILABLE = TOPOLOGICAL_BACKEND.is_available

if not TOPOLOGICAL_BACKEND_AVAILABLE:
    logging.log(100,
                "Topological features operation requires extra dependencies for time series forecasting, which are"
                " not installed. It can influence the performance. Please install them with 'pip install"
                " fedot[extra]'")


def ripser(data, maxdim, coeff, metric):
    if TOPOLOGICAL_BACKEND.backend is TopologicalBackend.GPH:
        return _gph_ripser(data,
                           maxdim=maxdim,
                           coeff=coeff,
                           metric=metric,
                           n_threads=1,
                           collapse_edges=False)
    if TOPOLOGICAL_BACKEND.backend is TopologicalBackend.RIPSER:
        return _ripser(data, maxdim=maxdim, coeff=coeff, metric=metric)
    raise ModuleNotFoundError("Install topological dependencies with 'pip install fedot[extra]'")
