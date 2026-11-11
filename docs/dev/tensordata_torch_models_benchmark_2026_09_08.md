# TensorData native Torch model migration and benchmark (2026-09-08)

## Scope and decisions

- Added a conservative native-Torch pool: `torch_linear`, `torch_mlp`,
  `torch_linear_reg`, and `torch_mlp_reg`.
- Existing sklearn operations (`logit`, `linear`, `mlp`, `mlpreg`) remain
  unchanged and available.
- `sk2torch` was not adopted: it wraps an already fitted sklearn estimator for
  Torch inference and does not move estimator training to GPU.
- `skorch` was not added: it is a sklearn-compatible trainer for user-provided
  PyTorch modules, while FEDOT already owns the strategy, TensorData, parameter,
  and pipeline lifecycle needed by these small modules.
- Official references inspected:
  - https://github.com/unixpickle/sk2torch
  - https://github.com/skorch-dev/skorch
  - https://skorch.readthedocs.io/en/latest/classifier.html

## Runtime contract fixed by this slice

- Fit consumes 2D floating `TensorData.features`, a single-column
  `TensorData.target`, `task`, and `dataloader_kwargs`.
- The model records `n_features_in_`, device, immutable fit plan, feature
  normalization statistics, class labels or regression target statistics.
- `device=auto` preserves the TensorData device. An explicit `device=cuda`
  trains on CUDA and returns predictions to the input TensorData device.
- `TensorData.idx` is row-aligned (`n_samples` values), rather than being built
  from feature width.
- A TensorData model parent's `predict` is now the next node's feature input;
  a data-operation parent's transformed `features` remains the next input.
- The experimental `tabm`, `ft_transformer`, `tab_resnet`, and `realmlp`
  strategies now attach Torch predictions to TensorData instead of returning a
  NumPy-backed OutputData object.

## Reproduction

GPU: NVIDIA GeForce RTX 4080 SUPER, 16 GB. PyTorch 2.7.1+cu126.

```bash
.venv/bin/python examples/benchmark/run_experimental_tabular_models.py \
  --tasks classification:bank-marketing \
  --operations logit,mlp,torch_linear,torch_mlp \
  --fast-epochs 50 --device cpu --result-path /tmp/tensordata_torch_models_bank_cpu_after_fix.csv

.venv/bin/python examples/benchmark/run_experimental_tabular_models.py \
  --tasks classification:bank-marketing,regression:diamonds \
  --operations torch_linear,torch_mlp,torch_linear_reg,torch_mlp_reg \
  --fast-epochs 50 --device cuda --result-path /tmp/tensordata_torch_models_gpu_after_fix.csv

.venv/bin/python examples/benchmark/run_experimental_tabular_models.py \
  --tasks regression:diamonds \
  --operations linear,torch_linear_reg,torch_mlp_reg \
  --fast-epochs 50 --device cpu --result-path /tmp/tensordata_torch_models_cpu_after_fix.csv

.venv/bin/python examples/benchmark/run_experimental_tabular_models.py \
  --tasks classification:MiniBooNE --operations torch_mlp \
  --fast-epochs 30 --device DEVICE \
  --result-path /tmp/tensordata_torch_models_miniboone_DEVICE.csv
```

Machine-readable results are in
`docs/dev/tensordata_torch_models_benchmark_2026_09_08.csv`.

## Findings

- `bank-marketing`: `torch_mlp` reached ROC AUC 0.92746 on CPU and 0.92617 on
  GPU, versus 0.89273 for sklearn MLP. GPU reduced end-to-end time from 6.064 s
  to 2.160 s (2.81x).
- `diamonds`: `torch_mlp_reg` reached RMSE 518.83 on CPU and 517.68 on GPU,
  versus 655.72 for the recorded sklearn MLP run. GPU reduced time from 19.241 s
  to 6.549 s (2.94x).
- `diamonds`: exact `torch_linear_reg` matches sklearn linear within 0.08 RMSE
  on CPU (1133.17 versus 1133.09). A closed-form Torch least-squares solver was
  used after short Adam training was found to underfit.
- `MiniBooNE`: CPU/GPU ROC AUC was 0.97905/0.97912. End-to-end time was
  19.287/15.491 s (1.25x); CPU preprocessing and host/device transfer are
  included in this timing.
- The small linear classifier is not a useful GPU speed target on these shapes:
  kernel launch and transfer overhead outweigh its matrix size. Its CPU path is
  retained as a lightweight differentiable baseline.

## Benchmark correctness note

The runner previously factorized concatenated train and test features for all
TensorData models. New Torch operations now use the public `create_data` fit
boundary, and test data is created with `from_data=train`. This reuses the fitted
`PreparationState` without fitting preprocessing on the test split. The raw path
also passes the computed classification labels into the training data boundary
instead of silently ignoring them.

After rebasing this implementation onto the hardened TensorData creation and
evaluation contracts, the CPU `bank-marketing` control run reproduced the
recorded ROC AUC exactly: 0.9046677622 for `torch_linear` and 0.9274595381 for
`torch_mlp` (50 epochs, random seed 42). This checks that the new public
`create_data(..., from_data=train)` path did not change model quality.
