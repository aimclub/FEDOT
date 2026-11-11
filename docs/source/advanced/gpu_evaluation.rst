Run on GPU
----------

FEDOT selects CPU/CUDA implementations from one model catalogue according
to the device of the input ``TensorData``. For cuML models, CUDA ``TensorData`` uses
a DLPack boundary between Torch and CuPy, so the adapter does not require a host
copy. Supported cuML operations include linear models, random forests, SVC,
k-nearest neighbours, Naive Bayes, mini-batch SGD and KMeans; consult
the ``implementations`` entries in ``model_repository.json`` for the current list.

The cuML model engine is currently enabled only on native Linux with an
available CUDA device. WSL and other operating systems are not selected.
Install the matching CUDA 12 optional dependency group with the project
package manager; FEDOT currently pins CuPy, cuDF, and cuML to compatible
versions in ``pyproject.toml``. Consult the `RAPIDS platform support`_
page before installing the GPU dependencies.

Use a regular preset and prepare data on CUDA:

.. code-block:: python

   from fedot import Fedot

   model = Fedot(problem='classification', preset='fast_train',
                 tensor_data_config={'backend_name': 'cuda:0'})
   train = model.create_data(features, target=target)
   model.fit(train, predefined_model='rf')
   test = model.create_data(test_features, from_data=train)
   prediction = model.predict(test)

The complete example is available in ``examples/advanced/gpu_example.py``.
The ``gpu`` preset and separate GPU model IDs have been removed. Neural models
and boosting libraries use the selected device through their own parameters;
CPU-only models remain available in mixed pipelines. The chosen implementation
and device are retained with the fitted estimator for prediction and saving.
Native Torch data operations already compute on their input tensors.

LightGBM uses its OpenCL ``gpu`` learner by default on CUDA inputs; the installed
LightGBM build must support it. GPU training does not guarantee that every internal
inference routine runs on GPU. Device transfers at external-library boundaries
remain necessary for libraries that accept only host arrays.

This first version binds model execution to the data device. General tuning spaces
still require implementation-specific constraints: for example, CatBoost's ordinary
GPU objectives reject ``colsample_bylevel`` from the shared CPU search space.
Complete GPU pipeline-search and tuning support is a separate step.


.. _RAPIDS platform support: https://docs.rapids.ai/platform-support/
