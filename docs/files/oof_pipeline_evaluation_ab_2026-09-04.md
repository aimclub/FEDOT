# OOF pipeline evaluation A/B, 2026-09-04

## Scope

This run checks the second local commit, `feat: add out of fold cross validation`, on branch
`fedot_1.0.0_new`. It deliberately exercises the legacy `InputData` operation runtime. The multi-node
TensorData runtime is not complete yet: using it here would mix the OOF change with unrelated TensorData
merging/model limitations.

The raw result table is in `oof_pipeline_evaluation_ab_2026-09-04.csv` next to this report.

## Defects found before the comparison

1. The public API stored `evaluation_mode='oof'` as a string while `GPComposer` compared it to the enum by
   identity. Consequently, the API never selected the OOF evaluator. The requirement now normalizes the public
   value to `PipelineEvaluationMode`.
2. `GPComposer` was abstract after the TensorData entrypoint rename because it no longer implemented the
   historical `compose_pipeline` contract. A delegating compatibility method now preserves that call pattern.
3. OOF returns legacy `OutputData`, but quality metrics now require TensorData/torch values. A boundary adapter
   now converts only the metric inputs; OOF execution itself stays on `InputData`.
4. The original implementation replaced a non-row-aligned TensorData index with generated positions. After the
   TensorData contract hardening, this fallback is intentionally removed: invalid `idx` now fails at the bridge
   boundary instead of hiding an upstream contract violation.
5. The first OOF algorithm cross-fitted every data operation independently. For `scaling -> logit`, rows used to
   train `logit` were therefore expressed in coordinate systems produced by different scalers, while its validation
   rows used another scaler. On Australian this produced OOF ROC AUC `0.844528`, versus legacy CV `0.915375` and
   external test `0.950764`. Data operations are now fitted once per current outer fold and transform both sides of
   that fold. Model parents use nested OOF predictions for the downstream training side and a model fitted on the
   complete outer-train side for prediction. After the fix, the same OOF score is `0.913422`.
6. The old test module collected zero tests because its helper import reached a module-level TensorData skip. The
   tests are now self-contained under `tests/core/optimisers/objective/`.

## Method

- Official AMLB/OpenML train/test splits: `Australian`, `car`, and `boston`.
- Three CV folds with seed 42.
- Same bounded candidate pool for both selectors: 6 classification or 7 regression pipelines, including single
  models, `scaling -> model`, model stacking, and two-parent stacking.
- `legacy_cv`: pipeline-level CV behavior where a downstream node trains on in-sample predictions from an upstream
  model fitted on the same outer-train rows.
- `oof_cv`: corrected fold-consistent OOF behavior from `PipelineOOFExecutor`.
- After each selector chooses its best candidate, that candidate is fitted on the complete AMLB train split using
  the legacy `InputData` operation path and measured on the untouched AMLB test split.
- Classification uses ROC AUC (higher is better); regression uses RMSE (lower is better).

This is a controlled pipeline search rather than a long stochastic genetic run. It evaluates multiple competing
pipeline structures while isolating the evaluator change and completing in seconds once OpenML metadata is cached.

In addition, a direct `ComposerBuilder` smoke run on Iris used a population of four, two requested generations,
operations `scaling`, `logit`, and `rf`, depth two, and three OOF folds. The optimizer completed four recorded
generations / 13 evaluated individuals and returned `logit`. This confirms that OOF is usable by the genetic search
itself.

## Selection outcome

| Dataset | Legacy choice | Legacy choice test | OOF choice | OOF choice test | OOF effect |
|---|---|---:|---|---:|---:|
| Australian | `rf` | 0.947368 | `rf -> logit` | 0.947368 | same |
| car | `rf` | 0.997154 | `rf -> logit` | 0.997681 | +0.000526 ROC AUC |
| boston | `rfr -> linear` | 2.262502 | `rfr -> linear` | 2.262502 | same |

Observed selection result: one improvement, two ties, no degradation. Metrics with different directions are not
averaged together. This small run is a smoke/regression check, not enough evidence for a general quality claim.

An interesting ranking detail is that `rf + dt -> logit` had the best external score on `car` (`0.997960`) but
neither three-fold selector ranked it first. More folds or repeated seeds would be needed to resolve that very small
difference reliably.

## TensorData boundary observed during testing

An attempted end-to-end `Fedot.fit` with the same multi-node `scaling -> logit` initial assumption failed before
composition while fitting the assumption: the current scaling implementation expects DataFrame `.iloc`, but received
a torch `Tensor`. This is an existing incomplete multi-node TensorData execution path, not an OOF executor failure.
For that reason the direct genetic smoke stops after composition, and the A/B quality check uses the legacy
`InputData` operation runtime for both alternatives. A single-node public API OOF run is covered and passes.

The legacy `test/unit/optimizer/test_pipeline_objective_eval.py` suite was also attempted, but collection currently
fails in an unrelated time-series example import because `fedot.core.composer.metrics.root_mean_squared_error` is
missing. The newer focused objective suite does collect and pass; the legacy collection failure was not changed as
part of the OOF commit.

## Adaptation after the TensorData evaluation-contract hardening

After rebasing onto `d1875e7a`, OOF selection is performed by the standard typed evaluator factory. Its request now
contains the evaluation mode, source TensorData and fold count. The OOF evaluator exposes the same
`EvaluationComplete` / `EvaluationIncomplete` / `EvaluationReused` lifecycle as the default evaluator, including
validation, explicit retry policy, deterministic cleanup and bounded result reuse. The legacy `InputData` conversion
is isolated inside the recursive OOF executor.

A short Australian `scaling -> logit` check produced OOF ROC AUC `0.931086`; the previous recorded run was
`0.913422`. This is not treated as an exact A/B comparison because the hardened source/preparation boundary changed
how the temporary raw OpenML check was constructed, but it shows no quality collapse. A strict default-vs-OOF search
must be repeated after the later native multi-node TensorData commit is applied: at this intermediate commit, the
default multi-node path already enters `TensorDataMerger`, while its model-output merge fix has not yet been rebased.

## Verification

The focused suite covers:

- OOF split coverage;
- real one- and multi-parent OOF evaluation;
- multiple metrics;
- no prediction of a row by the same fitted node instance that trained on that row;
- fold-consistent data-operation output supplied to the downstream node;
- public API activation with the string `evaluation_mode='oof'`;
- typed evaluator-factory selection and complete/reused outcome contracts;
- deterministic requirement and metric-boundary rules;
- strict TensorData-to-InputData row-index validation.

Command:

```bash
pytest -q \
  tests/core/pipelines/test_pipeline_composer_requirements_rules.py \
  tests/core/data/test_tensor_data_bridge_rules.py \
  tests/core/optimisers/objective/test_oof_objective_rules.py \
  tests/core/optimisers/objective/test_oof_objective_eval.py
```
