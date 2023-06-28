Tabular data
------------

Here are overall classification problem results across state-of-the-art AutoML frameworks
using `AutoMlBenchmark <https://github.com/openml/automlbenchmark>`__ test suite:

.. raw:: html
   :file: amlb_res.html


The results are obtained using sever based on Xeon Cascadelake (2900MHz)
with 12 cores and 24GB memory for experiments with the local infrastructure. 1h8c configuration was used for AMLB.

Despite the obtained metrics being a bit different from AMLB's `paper <https://arxiv.org/abs/2207.12560>`__
the results confirm that FEDOT is competitive with SOTA solutions.

The statistical analysis was conducted using the Friedman t-test.
The results of experiments and analysis confirm that FEDOT results are statistically indistinguishable
from SOTA competitors H2O, AutoGluon and LAMA (see below).

.. image:: img_benchmarks/stats.png
