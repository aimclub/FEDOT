"""
Utilities for measuring execution time on CPU and CUDA.

Provides wall-clock timers, CUDA event timers, and helpers built on
:mod:`torch.utils.benchmark` and :mod:`timeit` for profiling scripts and tests.
"""

from __future__ import annotations

import time
import timeit
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, TypeVar

import torch
import torch.utils.benchmark as benchmark

T = TypeVar("T")


@dataclass(frozen=True)
class TimingResult(Generic[T]):
    """Result of a single timed call."""

    value: T
    elapsed_s: float


@dataclass
class Stopwatch:
    """Mutable container filled by :func:`stopwatch` after the context exits."""

    elapsed_s: float = 0.0


def _require_cuda(device: str = "cuda") -> str:
    if not device.startswith("cuda"):
        raise ValueError(f"expected a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return device


def _cuda_device(device: str = "cuda") -> torch.device:
    _require_cuda(device)
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _sync_cuda(device: str = "cuda") -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(_cuda_device(device))


def warmup_cuda(repeat: int = 10, device: str = "cuda") -> None:
    """
    Run lightweight CUDA ops to warm up the GPU.

    Args:
        repeat: Number of warmup matrix multiplications.
        device: CUDA device identifier. Non-CUDA values are ignored.

    Examples:
        >>> warmup_cuda(device="cuda:0")
    """
    if not device.startswith("cuda"):
        return

    dev = _cuda_device(device)
    x = torch.randn(1024, 1024, device=dev)
    for _ in range(repeat):
        _ = x @ x
    torch.cuda.synchronize(dev)


def _measure_cuda(fn: Callable[..., T], *args: Any, **kwargs: Any) -> TimingResult[T]:
    """Measure pure GPU kernel time between CUDA events (excludes CPU overhead)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    value = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    return TimingResult(value=value, elapsed_s=start.elapsed_time(end) / 1000.0)


def measure_wall_time(
    fn: Callable[..., T],
    *args: Any,
    sync_cuda: bool = False,
    device: str = "cuda",
    **kwargs: Any,
) -> TimingResult[T]:
    """
    Measure wall-clock time for one callable invocation.

    Args:
        fn: Callable to time.
        *args: Positional arguments forwarded to ``fn``.
        sync_cuda: Whether to synchronize CUDA before and after ``fn``.
        device: CUDA device used when ``sync_cuda`` is enabled.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Callable result and elapsed wall time in seconds.

    Examples:
        >>> result = measure_wall_time(my_fn, data, sync_cuda=True, device="cuda:0")
        >>> print(result.elapsed_s)
    """
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize(_cuda_device(device))

    started = time.perf_counter()
    value = fn(*args, **kwargs)

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize(_cuda_device(device))

    return TimingResult(value=value, elapsed_s=time.perf_counter() - started)


def measure_gpu_time(
    fn: Callable[..., T],
    *args: Any,
    device: str = "cuda",
    warmup: bool = True,
    **kwargs: Any,
) -> TimingResult[T]:
    """
    Measure pure GPU kernel time via CUDA events.

    Args:
        fn: GPU callable to time.
        *args: Positional arguments forwarded to ``fn``.
        device: CUDA device on which ``fn`` is executed.
        warmup: Whether to run :func:`warmup_cuda` before timing.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Callable result and elapsed GPU kernel time in seconds.

    Raises:
        ValueError: If *device* is not a CUDA device string.
        RuntimeError: If CUDA is unavailable.

    Examples:
        >>> result = measure_gpu_time(gpu_fn, tensor, device="cuda:0")
        >>> print(result.elapsed_s)
    """
    _require_cuda(device)
    if warmup:
        warmup_cuda(device=device)

    dev = _cuda_device(device)
    torch.cuda.set_device(dev)
    _sync_cuda(device)
    return _measure_cuda(fn, *args, **kwargs)


def measure_time(
    fn: Callable[..., T],
    *args: Any,
    device: str = "cpu",
    warmup: bool = True,
    **kwargs: Any,
) -> TimingResult[T]:
    """
    Measure execution time on CPU or CUDA.

    On CUDA backends this delegates to :func:`measure_wall_time` with
    ``sync_cuda=True`` so the timer waits for GPU completion.

    Args:
        fn: Callable to time.
        *args: Positional arguments forwarded to ``fn``.
        device: ``"cpu"`` or a CUDA device string.
        warmup: Whether to warm up CUDA before timing GPU work.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Callable result and elapsed time in seconds.

    Examples:
        >>> result = measure_time(my_fn, data, device="cpu")
        >>> result = measure_time(gpu_fn, tensor, device="cuda:0")
    """
    if device.startswith("cuda"):
        _require_cuda(device)
        if warmup:
            warmup_cuda(device=device)
        return measure_wall_time(fn, *args, sync_cuda=True, device=device, **kwargs)

    return measure_wall_time(fn, *args, sync_cuda=False, **kwargs)


@contextmanager
def stopwatch(
    device: str = "cpu",
    warmup: bool = True,
    sync_cuda: bool | None = None,
    use_gpu_events: bool = False,
) -> Iterator[Stopwatch]:
    """Context manager that records elapsed time into :attr:`Stopwatch.elapsed_s`.

    By default uses wall-clock timing. Set ``use_gpu_events=True`` on CUDA to
    measure only GPU kernel time between CUDA events.

    Examples:
        >>> with stopwatch(device="cpu") as sw:
        ...     my_fn(data)
        >>> print(sw.elapsed_s)
    """
    sw = Stopwatch()

    if device.startswith("cuda"):
        _require_cuda(device)
        if warmup:
            warmup_cuda(device=device)

        if use_gpu_events:
            torch.cuda.set_device(_cuda_device(device))
            _sync_cuda(device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield sw
            finally:
                end.record()
                _sync_cuda(device)
                sw.elapsed_s = start.elapsed_time(end) / 1000.0
            return

        should_sync = sync_cuda if sync_cuda is not None else True
        if should_sync:
            _sync_cuda(device)

        started = time.perf_counter()
        try:
            yield sw
        finally:
            if should_sync:
                _sync_cuda(device)
            sw.elapsed_s = time.perf_counter() - started
        return

    started = time.perf_counter()
    try:
        yield sw
    finally:
        sw.elapsed_s = time.perf_counter() - started


def torch_benchmark(
    fn: Callable[..., Any],
    *args: Any,
    device: str = "cpu",
    min_run_time: float = 1.0,
    label: str | None = None,
    warmup: bool = True,
    **kwargs: Any,
) -> benchmark.Measurement:
    """
    Benchmark *fn* with :mod:`torch.utils.benchmark`.

    Args:
        fn: Callable to benchmark.
        *args: Positional arguments forwarded to ``fn``.
        device: ``"cpu"`` or a CUDA device string.
        min_run_time: Minimum total runtime passed to ``blocked_autorange``.
        label: Optional benchmark label. Defaults to ``fn.__name__``.
        warmup: Whether to warm up CUDA before GPU benchmarking.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Torch benchmark ``Measurement`` object.

    Examples:
        >>> measurement = torch_benchmark(my_fn, data, device="cpu")
        >>> print(measurement)
    """
    if device.startswith("cuda"):
        _require_cuda(device)
        if warmup:
            warmup_cuda(device=device)

        dev = _cuda_device(device)
        setup = f"torch.cuda.synchronize({dev!r})"
        stmt = f"fn(*args, **kwargs); torch.cuda.synchronize({dev!r})"
    else:
        setup = "pass"
        stmt = "fn(*args, **kwargs)"

    timer = benchmark.Timer(
        stmt=stmt,
        setup=setup,
        globals={"fn": fn, "args": args, "kwargs": kwargs, "torch": torch},
        label=label or getattr(fn, "__name__", "fn"),
        sub_label=device,
        description="torch.utils.benchmark",
    )
    return timer.blocked_autorange(min_run_time=min_run_time)


def cpu_benchmark_timer(
    fn: Callable[..., Any],
    *args: Any,
    min_run_time: float = 1.0,
    warmup: bool = True,
    **kwargs: Any,
) -> benchmark.Measurement:
    """
    Benchmark on CPU via :func:`torch_benchmark`.

    Examples:
        >>> measurement = cpu_benchmark_timer(my_fn, data)
        >>> print(measurement.median)
    """
    return torch_benchmark(
        fn, *args, device="cpu", min_run_time=min_run_time, warmup=warmup, **kwargs
    )


def cuda_benchmark_timer(
    fn: Callable[..., Any],
    *args: Any,
    min_run_time: float = 1.0,
    warmup: bool = True,
    **kwargs: Any,
) -> benchmark.Measurement:
    """
    Benchmark on CUDA via :func:`torch_benchmark`.

    Examples:
        >>> measurement = cuda_benchmark_timer(gpu_fn, tensor)
        >>> print(measurement.median)
    """
    return torch_benchmark(
        fn, *args, device="cuda", min_run_time=min_run_time, warmup=warmup, **kwargs
    )


def timeit_benchmark(
    fn: Callable[..., Any],
    *args: Any,
    number: int = 100,
    repeat: int = 5,
    device: str = "cpu",
    warmup: bool = True,
    **kwargs: Any,
) -> list[float]:
    """
    Repeated timing with :mod:`timeit`.

    Args:
        fn: Callable to benchmark.
        *args: Positional arguments forwarded to ``fn``.
        number: Number of inner-loop executions per repeat.
        repeat: Number of repeat measurements.
        device: ``"cpu"`` or a CUDA device string.
        warmup: Whether to warm up CUDA before GPU benchmarking.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        List of total elapsed seconds for each repeat.

    Examples:
        >>> runs = timeit_benchmark(my_fn, data, number=100, repeat=5)
        >>> print(min(runs), max(runs))
    """
    globals_dict = {"fn": fn, "args": args, "kwargs": kwargs, "torch": torch}

    if device.startswith("cuda"):
        _require_cuda(device)
        if warmup:
            warmup_cuda(device=device)

        dev = _cuda_device(device)
        setup = f"torch.cuda.synchronize({dev!r})"
        stmt = f"fn(*args, **kwargs); torch.cuda.synchronize({dev!r})"
    else:
        setup = "pass"
        stmt = "fn(*args, **kwargs)"

    return timeit.repeat(
        stmt,
        setup=setup,
        globals=globals_dict,
        number=number,
        repeat=repeat,
    )


def _runs_summary(runs: list[float]) -> dict[str, float]:
    ordered = sorted(runs)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2

    return {
        "mean_s": sum(runs) / len(runs),
        "median_s": median,
        "min_s": min(runs),
        "max_s": max(runs),
        "runs": float(len(runs)),
    }


def pytest_benchmark_timer(
    fn: Callable[..., Any],
    *args: Any,
    rounds: int = 5,
    device: str = "cpu",
    warmup: bool = True,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Repeated benchmark via :mod:`timeit` with pytest-benchmark-style summary.

    In pytest tests prefer the ``benchmark`` fixture from ``pytest-benchmark``.

    Args:
        fn: Callable to benchmark.
        *args: Positional arguments forwarded to ``fn``.
        rounds: Number of repeat measurements.
        device: ``"cpu"`` or a CUDA device string.
        warmup: Whether to warm up CUDA before GPU benchmarking.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Mapping with ``mean_s``, ``median_s``, ``min_s``, ``max_s``, and ``runs``.

    Examples:
        >>> stats = pytest_benchmark_timer(my_fn, data, rounds=5)
        >>> print(stats["median_s"])
    """
    runs = timeit_benchmark(
        fn, *args, number=1, repeat=rounds, device=device, warmup=warmup, **kwargs
    )
    return _runs_summary(runs)
