# PR #1467 multiclass-quality experiments

Branch: `codex/pr1467-multiclass-quality`, parent
`2667e5df1081efcf342c44567ff9dddeb974cbf5`.

## Motivation and scope

The largest losses on matching OpenML tasks/folds against the AMLB paper's
AutoGluon 0.8.0 results include Dionis, fabert and Covertype. The original FEDOT
scores precede the parent's full-training-data refit fix; they are not a clean
baseline for additional changes. Paper hardware differs from our servers.

On original Dionis fold 0, LightGBM repeatedly selected iteration 1 (inner
logloss around 3.32–3.38). Evolution returned scaling/logistic regression;
the tuner reported infinite initial and final fitness. Its cause is still
under investigation and is **not** claimed to be fixed here.

## Candidate: multiclass leaf-Hessian regularization

LightGBM defaults to a minimum leaf Hessian of 0.001. Small Hessians can allow
extreme Newton updates for rare classes. This candidate uses 1.0 for
single-output multiclass classification when the user has not specified this
parameter or any of its LightGBM aliases. Binary, regression and multioutput
defaults are unchanged. No task IDs, outer test labels, additional estimators,
or changes to the search time allocation enter this rule.

This is an experimental hypothesis, **not a demonstrated AMLB improvement**.
The isolated train-only probe compares 0.001 and 1.0 on the same inner split
of the OpenML outer training fold. It is a mechanism check, not a benchmark.

## Controlled full-run protocol

- Baseline: the parent above; candidate: this branch's recorded commit.
- One sequential baseline/candidate pair per server, same task/fold/seed.
- Initial pairs: Dionis fold 0, fabert fold 3, Covertype fold 3.
- AMLB `1h8c`: 3600 seconds, 8 threads; `_portfolio=false`, `best_quality`,
  tuning enabled, sequential pipeline evaluation.
- Same original harness and pinned GOLEM vendor for both variants.
- Source archives are immutable per experiment; commits and import paths are
  recorded. The original PR checkout, original scores and PR branch are untouched.
- `examples/advanced/run_multiclass_quality_pair.sh` refuses any run already
  marked `STARTED`, including failures. It records finite scores and checks
  both training and job duration against 3600 seconds. Over-budget results
  must not be counted as successful improvements.
- Compare both logloss and time; inspect finite/normalized prediction files,
  selected graphs, tuning fitness and errors before drawing conclusions.
- One fold per task is screening evidence only. Require further folds and
  regression checks before proposing this default for the original PR.
