Run on GPU
----------

FEDOT supports native RAPIDS cuML model evaluation for both ``InputData``
and ``TensorData`` pipelines. CUDA ``TensorData`` uses a DLPack boundary
between Torch and CuPy, so model fit and prediction do not require a host
copy. Supported operations include linear models, random forests, SVC,
k-nearest neighbours, Naive Bayes, mini-batch SGD and KMeans; consult
``gpu_models_repository.json`` for the current list.

The cuML model engine is currently enabled only on native Linux with an
available CUDA device. WSL and other operating systems are not selected.
Install the matching CUDA 12 optional dependency group with the project
package manager; FEDOT currently pins CuPy, cuDF, and cuML to compatible
versions in ``pyproject.toml``. Consult the `RAPIDS platform support`_
page before installing the GPU dependencies.

Select the GPU repository through the public API:

.. code-block:: python

   from fedot import Fedot

   model = Fedot(problem='classification', preset='gpu')
   model.fit(features, target)
   prediction = model.predict(test_features)

The complete example is available in ``examples/advanced/gpu_example.py``.
The legacy ``InputData`` path remains supported, while new integrations
should prefer ``TensorData`` so consecutive GPU operations can keep data
on the device.


.. _RAPIDS platform support: https://docs.rapids.ai/platform-support/
