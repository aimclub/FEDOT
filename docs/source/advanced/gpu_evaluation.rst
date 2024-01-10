Run on GPU
----------

FEDOT now supports the evaluation of some models within pipelines with
help of RAPIDS library for GPU evaluation. Currently, FEDOT allows you
to use Ridge, Lasso, LogisticRegression, RandomForestClassifier,
RandomForestRegressor, KMeans, SVC. This list will be extended further.

Due to the multiple hardware ans software limits set by RAPIDS cuml and
cudf libraries it is simpler to make such evaluations within containers.

Using the official RAPIDS docker image we extended it with FEDOT project
requirements. For simple usage create an image from the
`Dockerfile`_ provided in repository and run the gpu_examples.py
from /home/FEDOT/examples/ directory. Read the hardware prerequisites on
the `RAPIDS official page`_.

But this way of using is quite solid because you can’t change anything.
In case of willing to explore more you will need to rebuild the image
every time.

That is why we want to share the way we use RAPIDS:

-  Clone the project
   ``git clone https://github.com/aimclub/FEDOT.git``
-  Use your lovely FTP client to copy the project to the host where the
   Docker is preinstalled or make the deployment via IDE you use
-  Pull the RAPIDS image via
   ``docker pull nvcr.io/nvidia/rapidsai/rapidsai:21.06-cuda11.2-base-ubuntu18.04``
-  and
   ``docker run -it --rm -e NVIDIA_VISIBLE_DEVICES=0 -v /host/path/project:/home/FEDOT rapids``
-  Inside the container run ``pip3 install .[extra]’``
-  Run ``python3 /home/FEDOT/examples/gpu_example.py’``

This approach doesn’t has an entry point for the container so it allows
you make contributions and check the changes in place.


.. _Dockerfile: https://github.com/aimclub/FEDOT/blob/master/gpu/Dockerfile
.. _RAPIDS official page: https://rapids.ai/start.html
