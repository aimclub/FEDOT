Tabular data
------------

Here are overall classification problem results across state-of-the-art AutoML frameworks
using ``paper <https://github.com/openml/automlbenchmark>`__ <https://arxiv.org/abs/1907.00909>`__
test suite:

.. raw:: html
   :file: amlb_res.html


The results are obtained using sever based on Xeon Cascadelake (2900MHz)
with 12 cores and 24GB memory for experiments with the local infrastructure.

Despite the obtained metrics are a bit different from ALMB `paper <https://arxiv.org/abs/1907.00909>`__
the results confirms that FEDOT is competitive with SOTA solutions.

The statistical analysis was conducted using Friednman t-test.
The results of confirms that FEDOT results are statistically indistinguishable
from SOTA competitors H2O, AutoGluon and LAMA (see below)

  .. image:: img_benchmarks/stats.png
      :width: 100%