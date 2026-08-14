# TensorData categorical detection after the origin rebase (2026-09-26)

## Final rule

Mixed tabular sources become a common object array before obligatory
preprocessing. To retain support for NumPy and other non-DataFrame inputs,
categorical detection therefore still inspects column values rather than relying
only on pandas dtypes.

A numeric-looking object column is treated as categorical when it has between 3
and 12 non-missing unique values **and at least one value is repeated**.
Non-numeric string columns remain categorical. Homogeneous numeric arrays remain
numeric.

The repetition condition resolves the origin contract regression: a four-row
numeric column `[7, 2, 9, 4]` is no longer encoded merely because it has four
unique values, while repeated category codes in Australian and Boston retain the
pre-rebase behavior. The rule also remains applicable to NumPy object arrays,
where source dtype metadata is unavailable.

## Alternatives checked

Treating every pandas `category` dtype as an unconditional encoding instruction
changed the historical feature plan. On Australian it reduced ROC AUC from
0.953735 to 0.948642 for ExtraTrees, from 0.933786 to 0.926995 for HistGB, and
from 0.962649 to 0.955857 for EBM. It also changed Boston EBM RMSE from 2.247305
to 2.336932. That approach was discarded.

## Post-rebase control run

Command:

```bash
python examples/benchmark/run_experimental_tabular_models.py \
  --tasks classification:Australian,regression:boston \
  --operations extra_trees,hist_gb,ebm,hist_gbreg,ebmreg,mlpreg
```

| Dataset | Operation | Metric | Pre-rebase | Final post-rebase |
|---|---|---:|---:|---:|
| Australian | extra_trees | ROC AUC | 0.9537351443 | 0.9537351443 |
| Australian | hist_gb | ROC AUC | 0.9337860781 | 0.9337860781 |
| Australian | ebm | ROC AUC | 0.9626485569 | 0.9626485569 |
| Boston | hist_gbreg | RMSE | 2.4241542652 | 2.4241542652 |
| Boston | ebmreg | RMSE | 2.2473045199 | 2.2473045199 |
| Boston | mlpreg | RMSE | 5.7168199928 | 5.7168199928 |

All six metrics reproduce the saved pre-rebase results to floating-point
precision. Focused TensorData/bridge/planner tests also pass, including the
origin round-trip invariant and direct repeated/all-unique object-column cases.
