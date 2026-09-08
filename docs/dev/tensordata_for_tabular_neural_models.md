# TensorData and tabular neural models

Updated: 2026-09-26, after the TensorData contract hardening in `d1875e7a`.

## Runtime contract

`TensorData` is now the native pipeline container. It owns prepared Torch
tensors, row labels, feature metadata, task/type information and an optional
`PreparationState`. Creation validates row alignment immediately and keeps
`features`, `target`, `predict` and tensor `idx` on one device.

Use the public creation boundary so prediction reuses fitted preprocessing:

```python
from fedot import Fedot, create_data

train = create_data(X_train, target=y_train, backend='cpu')
test = create_data(X_test, from_data=train, backend='cpu')

model = Fedot(problem='classification')
model.fit(train, predefined_model='torch_mlp')
prediction = model.predict(test)
```

`TensorDataCreator` remains the implementation shell, but new application and
benchmark code should prefer `create_data(..., from_data=train)`. Recreating
test data independently can fit a second preprocessing state and invalidate a
quality comparison.

The `InputData` bridges are compatibility boundaries for legacy execution such
as the recursive OOF evaluator. They are not the main storage path and a bridge
must reject an invalid row index instead of generating a replacement.

## Native Torch models

The native pool contains:

- `torch_linear` and `torch_mlp` for classification;
- `torch_linear_reg` and `torch_mlp_reg` for regression;
- the experimental `tabm`, `ft_transformer`, `tab_resnet` and `realmlp`
  classification/regression variants.

The lightweight Torch models consume `TensorData` directly. Their immutable
fit plan resolves the exact device, batching, hidden sizes, validation split,
early stopping and random state before training. `device='auto'` follows the
actual input tensor device, including a CUDA ordinal such as `cuda:1`.
Predictions are returned on the input device and attached by creating a new
`TensorData`; model strategies do not mutate the caller's container.

For a secondary pipeline node, parent output selection is semantic:

- a model parent contributes `predict`;
- a data-operation parent contributes transformed `features`;
- merged values become the child node's `features`, and `predict` is cleared.

The selection is planned from the parent operation role, not inferred only from
whether a value happens to be present in `predict`.

## Legacy sklearn models

Classic sklearn implementations still require host-compatible arrays and do
not become GPU models merely because their input originated as TensorData.
`MLPRegressor`, tree ensembles and similar operations therefore remain useful
compatibility baselines. Native Torch or cuML implementations should be chosen
when device-resident training is the objective.

## Remaining limitations

- Some legacy operations still convert tensors to NumPy internally.
- The recursive OOF executor deliberately uses an isolated InputData execution
  path until every pipeline operation is TensorData-native.
- GPU acceleration is workload-dependent: small linear models can be slower on
  GPU because transfer and kernel-launch overhead dominate.
- External tabular neural packages may perform their own CPU preprocessing even
  when their training module uses Torch.

The next migration steps are to keep expanding the native model/operation pool,
remove compatibility bridges only after their callers are gone, and evaluate
quality and end-to-end speed on datasets appropriate for each model class.
