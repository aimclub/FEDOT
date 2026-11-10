# PR1467 follow-up: measured CV budget and prediction headroom

Experimental branch: `codex/pr1467-multiclass-quality`. Original PR and all
previous deployed code/results remain unchanged. No quality gain is claimed.

## Predeclared series (2026-11-10)

Stage on each host: `experiments/pr1467-diagnostics-20261109/budget-hypotheses-20261110`.
Runs are exclusive, immutable, and refuse to restart a STARTED directory.

1. Copper / Dionis0: regularization candidate `0bbb1fe5` versus measured-budget
   revision `6f05a1ae`. Same 1h8c harness, seed104729, tuning enabled, portfolio
   disabled. Test whether selected-graph CV timing prevents invalid initial
   and final tuner fitness. Skipping infeasible tuning is a valid timing fix,
   but not evidence of better predictions.
2. Ruthenium / Dionis0: measured-budget revision versus the additional native
   prediction-headroom change (full immutable commits in each run's
   `source.commit`). For single-output classification with >=100 classes,
   automatically injected native boosting limits use 65% rather than 90% of
   the unchanged pipeline fit deadline. Explicit node limits are preserved.
   Hypothesis: model conversion/train prediction can finish after native
   training stops. Risk: fewer trees reduce quality. Compare actual iterations,
   failed CV folds, graphs, probabilities, logloss and total time.
3. Dysprosium / Covertype3: train-only fixed five-node graph factorial ablation.
   Cross both previously selected parameter sets with explicit LightGBM leaf
   Hessian 0.001 / 1.0, at seeds42 and104732 (eight fits). Same inner stratified
   80/20 split within the OpenML outer training data. This separates parameter
   selection from the Hessian change without accessing outer-test labels.
   It is NOT a full FEDOT run or an AutoMLBenchmark score. Each model gets an
   explicit seed; stochastic execution and hardware can still affect timings.

## Timing implementation

GOLEM `individual.metadata['computation_time_in_seconds']` measures the whole
CV, including preprocessing and predictions. Match valid history individuals
by exact graph descriptive ID and use the largest observed per-fold time.
Do not use `Pipeline.computation_time`, which stores minutes and excludes parts
of evaluation. Reserve estimated full-data refit separately; budget initial
CV, an in-flight trial and final CV with 30% headroom. Try three then two folds;
skip tuning if neither fits. Unmeasured/small graphs retain the old policy.

This estimate is not a hard overall deadline: slower tuned parameters and
prediction overhead must still be checked against the 3600-second job limit.

## Acceptance / next steps

Validate source commits, matching data/seed/resource settings, finite and
normalized probabilities, independently recomputed logloss and <=3600s job
and training time. Record failures, not only completed scores. New trials may
follow only after checking the current runs and specifying a new hypothesis;
do not duplicate active work. Promote changes only after paired evidence and
additional folds, not a single favorable result. Stop exhausted hypotheses.
