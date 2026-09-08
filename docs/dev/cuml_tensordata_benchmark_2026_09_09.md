# Native cuML model engine for TensorData (2026-09-09)

## Scope

- Revived the RAPIDS model repository and pinned `cuml-cu12==25.8.0` alongside the existing
  CuPy/cuDF 25.8 stack.
- Added a typed, immutable `ModelEnginePlan`. It resolves cuML, Torch, or sklearn from the
  operation's supported engines, the installed libraries, CUDA availability, and the runtime
  platform. cuML is deliberately enabled only on native Linux; WSL is rejected for now.
- Kept the legacy `InputData -> NumPy -> cuML -> NumPy -> OutputData` path for compatibility.
- Added a native TensorData boundary: CUDA Torch tensors are exposed to CuPy/cuML with DLPack,
  and CuPy predictions are returned as Torch tensors on the source TensorData device. There is
  no host copy on the CUDA TensorData path.

The supported pool is:

- classification: `logit`, `rf`, `svc`, `knn`, `multinb`, `bernb`, `minibatchsgd`;
- regression: `linear`, `ridge`, `lasso`, `elasticnet`, `rfr`, `knnreg`, `mbsgdcregr`, `cd`;
- clustering: `kmeans`.

The historical `sgd` name maps to `cuml.solvers.SGD`, a low-level solver rather than a complete
classifier. It remains available to the legacy repository but is tagged `non-default` and is
explicitly rejected on the TensorData model path. `cd` was corrected from classification to
regression. Bernoulli Naive Bayes was added, and SVC now fits probabilities when a pipeline asks
for them.

No separate cuML data-operation layer was added. TensorData `pca` and `truncated_svd` already use
Torch on the TensorData device, while obligatory/optional preprocessing already preserves the
TensorData runtime. A second CuPy implementation would add conversion and maintenance cost without
a demonstrated speed or quality benefit.

## Correctness fixes found during benchmarking

Direct Torch input is advertised by current cuML, but cuML 25.8 fails for Torch targets because
its target conversion still expects a NumPy-like dtype. The implementation therefore uses the
stable Torch/CuPy DLPack boundary for both features and targets.

cuML logistic regression in float32 lost about 0.03 ROC AUC on `bank-marketing` and reported
non-convergence. The model-specific precision plan now uses float64 plus conservative convergence
defaults; the final ROC AUC is 0.90761 versus 0.90756 for sklearn.

cuML forests use quantile split candidates rather than sklearn's exact split search. Raising the
default to `n_bins=256` improved the measured forest metrics while preserving a large speedup.

## Benchmark protocol

Hardware: NVIDIA GeForce RTX 4080 SUPER 16 GB, driver 580.173.02. Software: Python 3.10,
PyTorch 2.7.1+cu126, cuML/cuDF 25.8.0, CuPy 13.5.1.

Both engines receive the same TensorDataCreator and optional-scaling result and the same train/test
split. The CPU baseline is then bridged to `InputData`; the cuML input is moved to CUDA TensorData.
`model_seconds` includes fit and prediction. `transfer_seconds` is recorded separately. Reported
values below are medians of three independent fits on AMLB/OpenML tasks unless noted otherwise.

| Task | Model | CPU metric | cuML metric | cuML - CPU | Model speedup |
|---|---|---:|---:|---:|---:|
| bank-marketing | logit | AUC 0.907565 | AUC 0.907611 | +0.000046 | 5.09x |
| bank-marketing | rf (`n_bins=256`) | AUC 0.933130 | AUC 0.931303 | -0.001826 | 5.63x |
| bank-marketing | svc | AUC 0.688424 | AUC 0.687311 | -0.001114 | 117.06x |
| bank-marketing | knn | AUC 0.728839 | AUC 0.728432 | -0.000407 | 18.99x |
| bank-marketing | bernb | AUC 0.733074 | AUC 0.733074 | 0.000000 | 2.71x |
| diamonds | linear | R2 0.917130 | R2 0.917136 | +0.000007 | 0.89x |
| diamonds | ridge | R2 0.917134 | R2 0.917136 | +0.000002 | 0.40x |
| diamonds | lasso | R2 0.917115 | R2 0.917174 | +0.000059 | 64.61x |
| diamonds | rfr (`n_bins=256`) | R2 0.983280 | R2 0.981668 | -0.001612 | 8.06x |
| diamonds | knnreg | R2 0.909597 | R2 0.909567 | -0.000030 | 23.66x |

The slight AUC loss of GPU RF is accompanied by better accuracy (0.90447 versus 0.90093) and
macro-F1 (0.69764 versus 0.68644). SVC likewise improves accuracy and macro-F1. Linear and ridge
are too small on this dataset to amortize GPU startup and transfer; the engine plan makes it
possible to keep such operations on CPU in a later mixed-engine composer. The forest differences
are algorithmic rather than a TensorData conversion error and remain below 0.002 in the primary
metric.

Machine-readable raw results:

- `docs/dev/cuml_tensordata_benchmark_2026_09_08.csv` — complete 60-row CPU/cuML run;
- `docs/dev/cuml_tensordata_benchmark_forest_256_bins_2026_09_09.csv` — final forest refinement.

## Post-rebase control

After rebasing onto the hardened TensorData creation and evaluation contracts,
a one-repeat `bank-marketing` control run produced matching CPU/cuML logistic
ROC AUC (0.907322 for both engines). Random forest reached 0.933151 on CPU and
0.931303 on cuML; the cuML value is unchanged from the recorded three-repeat
result. The measured model times were 7.179 s on CPU and 1.190 s on cuML for
random forest (6.03x). In addition, all 21 CUDA integration tests passed on the
RTX 4080 SUPER, including a TensorData data-operation-to-cuML pipeline.

## Reproduction

```bash
.venv/bin/python examples/benchmark/run_cuml_tensordata_models.py --repeats 3
```

If pip-installed NVIDIA runtime libraries are not globally discoverable, add the `cuda_nvrtc`,
`nvjitlink`, and `cuda_runtime` package directories from `.venv/.../site-packages/nvidia/` to
`LD_LIBRARY_PATH`. This is an environment packaging requirement, not a TensorData host transfer.

Official references inspected:

- https://docs.rapids.ai/api/cuml/nightly/cuml_intro/
- https://docs.rapids.ai/platform-support/
- https://docs.rapids.ai/api/cuml/legacy/supported_versions/
