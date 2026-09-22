AMLB Sampling Benchmark
=======================

FEDOT provides a benchmark entrypoint for AMLB-style tabular datasets with pre-fit sampling stage.

Script path:

- ``examples/benchmark/run_amlb.py``

Default behavior
----------------

- Uses AMLB category profile ``amlb_top20_mix``.
- Runs two modes for each dataset:

  - ``fedot_full_dataset`` (baseline, no sampling).
  - ``fedot_sampling_stage`` (sampling enabled through ``sampling_config``).

- Time budget per dataset is **15 minutes** by default.
- Saves optimization artifacts for each run:

  - ``opt_history.json``
  - history visualizations (fitness, KDE, animated bars/diversity where available)
  - pipeline visualization
  - predictions, metrics and timing reports

Quick start
-----------

.. code-block:: bash

    python examples/benchmark/run_amlb.py

Custom run examples
-------------------

Run only sampling mode:

.. code-block:: bash

    python examples/benchmark/run_amlb.py --disable-baseline

Run specific AMLB datasets with fixed 15-minute budget:

.. code-block:: bash

    python examples/benchmark/run_amlb.py \
      --datasets amlb_adult amlb_credit_g \
      --timeout-minutes 15

Tune sampling protocol options:

.. code-block:: bash

    python examples/benchmark/run_amlb.py \
      --candidate-ratios 0.15,0.2,0.3,0.5 \
      --delta-threshold 0.03 \
      --sampling-strategy random

Output artifacts
----------------

By default results are stored under:

- ``examples/benchmark/results/run_amlb_fedot_sampling_<timestamp>/``

Run-level files:

- ``run_meta.json``
- ``benchmark_runs.csv``
- ``benchmark_runs.json``
- ``report.md``
- ``run_summary.json``

Per-dataset and per-mode files include all optimization and visualization artifacts produced during run.
