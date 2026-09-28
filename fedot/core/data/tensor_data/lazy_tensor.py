from typing import Callable, Optional

from fedot.core.data.tensor_data.tensor_data import TensorData


class LazyTensor:
    """
    Lazy wrapper around `TensorData` creation.

    The wrapper stores a zero-argument factory and delays data reading,
    preprocessing, and tensor conversion until :meth:`get` or :meth:`to` is called.
    The materialized `TensorData` is cached in `_data`, so the factory is executed
    only once per `LazyTensor` instance.

    Examples:
        >>> lazy_td = TensorDataCreator.create_lazy(X, backend_name='cpu')
        >>> lazy_td
        LazyTensor(initialized=False)
        >>> td = lazy_td.get()
    """

    def __init__(self, create_fn: Callable[[], TensorData]):
        """
        Args:
            create_fn: Factory function used to materialize `TensorData`.
        """
        self._create_fn = create_fn
        self._data: Optional[TensorData] = None

    def get(self) -> "TensorData":
        """
        Materialize and return the cached `TensorData`.

        On the first call, `_create_fn` is executed and its result is stored in
        `_data`. Later calls return the same `TensorData` object.

        Returns:
            TensorData: Materialized tensor data.
        """
        if self._data is None:
            self._data = self._create_fn()
        return self._data

    def to(self, device: str):
        """
        Materialize data if needed and move it to the given device.

        Args:
            device: Target device, for example `"cpu"` or `"cuda"`.

        Returns:
            TensorData: Materialized tensor data moved to `device`.
        """
        data = self.get()
        return data.to(device)

    def __repr__(self):
        """
        Build a short debug representation.

        Returns:
            str: Representation showing whether `TensorData` has been materialized.
        """
        return f"LazyTensor(initialized={self._data is not None})"
