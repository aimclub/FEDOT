Sampling Stage Before AutoML Search
==================================

FEDOT supports an optional ``sampling_stage`` in ``Fedot.fit()``.
This stage runs before evolutionary pipeline composition and can reduce
training set size for classification/regression tabular tasks.

The stage is controlled by a single API parameter: ``sampling_config``.
If ``sampling_config`` is ``None`` (default), FEDOT behavior is unchanged.

Quick Start
-----------

For AMLB-style benchmark execution with sampling mode and saved optimization artifacts, see :doc:`/examples/amlb_sampling_benchmark`.

.. code-block:: python

    from fedot import Fedot

    model = Fedot(
        problem='classification',
        timeout=10,
        sampling_config={
            'provider': 'sampling_zoo',
            'strategy': 'random',
            'candidate_ratios': [0.15, 0.2, 0.3, 0.5],
            'delta_metric_threshold': 0.03,
        }
    )

    model.fit(features=x_train, target=y_train)

    # Metadata is available after fit
    print(model.sampling_stage_metadata)

V1 Scope and Behavior
---------------------

- Supported tasks: ``classification`` and ``regression`` only.
- Supported data container: ``InputData`` only.
- Supported data type: tabular only.
- Stage is applied only in ``fit`` (not in ``predict`` and not in ``tune``).
- Error mode is ``fail_fast`` only in V1.
- Budget policy is ``dynamic_cap`` only in V1.
- Artifact mode is ``minimal`` only in V1.

If the stage cannot be executed in these constraints, FEDOT raises an exception.

Effective Size Protocol
-----------------------

The stage uses an internal protocol to choose the effective ratio:

1. Split train data into internal train/validation parts.
2. Train a light baseline model (Random Forest) on full internal train split.
3. For each candidate ratio:

   - Run sampling and obtain candidate indices.
   - Train the same light model on sampled internal train split.
   - Measure metric delta on internal validation split.

4. Select the smallest ratio that satisfies ``delta_metric_threshold``.

If none of candidate ratios satisfies the threshold, FEDOT raises an exception.

Main ``sampling_config`` Fields
-------------------------------

- ``provider``: sampling provider name (V1 supports ``sampling_zoo``).
- ``strategy``: strategy identifier passed to provider.
- ``strategy_params``: provider-specific strategy kwargs.
- ``candidate_ratios``: ordered ratios in ``(0, 1]``.
- ``delta_metric_threshold``: allowed quality drop.
- ``delta_type``: ``relative`` or ``absolute``.
- ``validation_size``: internal validation size in ``(0, 1)``.
- ``cap_max_timeout_share``: maximal timeout share for stage.
- ``min_automl_time_minutes``: guaranteed minimal time left for AutoML after stage.
- ``infinite_timeout_cap_minutes``: absolute stage cap when timeout is infinite.
- ``random_state``: random seed.

Performance Guards
------------------

For heavy strategies, config validation checks guard limits:

- ``guard_max_rank``
- ``guard_max_modes``
- ``guard_max_partitions``
- ``guard_max_sample_size``

These limits prevent unexpectedly expensive strategy parameters.

Optional Dependency
-------------------

Sampling Zoo integration is optional.

.. code-block:: bash

    pip install "fedot[sampling_zoo]"

If dependency is unavailable and sampling stage is enabled, FEDOT raises
``ModuleNotFoundError`` in ``fail_fast`` mode.


