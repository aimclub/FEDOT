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

The first Covertype preparation stopped before any fit because OpenML marks
its binary indicator columns as categorical. Preserve that FAILED run. A new
stage `budget-hypotheses-probe-v2-20261110` explicitly accepts these columns
only after checking that their outer-training values are numeric 0/1. No
encoder is fitted; all factorial cells share this representation. The full
AMLB runs on copper/ruthenium are unaffected and must not be restarted.

Implementation verification before launch: 44 local tests passed (budget,
LightGBM, multiclass headroom and native-limit restoration); 16 budget tests
passed separately in each of the three server environments. The immutable
full-run headroom candidate is `caddb242d09f64151ecc52f369240da7b9406923`.

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

## Interim inspection: 2026-11-10, 16:07 UTC

All three queues were checked against live processes, not only marker files.
No jobs were restarted and deployed code was not edited.

### Dionis on ruthenium: measured-budget control completed

Revision `6f05a1ae`, fold0, seed104729: independently recomputed outer-test
logloss **0.8051071883477428**, job **1097.1 s**, training **917.9 s**.
All 41,619 rows / 355 probability columns were finite, in [0,1], with maximum
normalization error 8.44e-15. The recomputed loss matches the printed score.
Prediction SHA-256:
`c00140cbedd1ce23935c0be53b1cb419e0997ef71cc5ee30f26066753e06b59f`.
Harness and vendored tuner hashes still match the preceding report.

This is **not evidence that the measured-CV budgeting change improved quality**:
the initial LightGBM fit took 584.1 s, estimated five-fold cost was 2920.6 s,
and composition was skipped. Tuning was also skipped with zero estimated
remaining allocation. Final full-data fitting exceeded its 324-second cap;
FEDOT restored the fitted initial `scaling -> lgbm` assumption. There was no
selected-graph evolutionary CV timing for the new policy to consume.
The headroom candidate started at 16:04:06 UTC and is still running.

On copper the regularized control is still in composition. Its initial fit
took 367.6 s; its LightGBM CV fit subsequently exceeded 324 s. The difference
in initial fit duration across hosts changes whether composition runs at all;
cross-host score differences cannot isolate a code change.

### Covertype fixed-graph probe: partial results only

Seven of eight fits had printed results by this inspection; the last fit
was still running and no COMPLETE marker existed. Completed matched cells:

| Seed | Frozen parameter source | Hessian 0.001 inner loss | Hessian 1.0 inner loss |
| --- | --- | ---: | ---: |
| 42 | Earlier baseline | 0.3026504446866336 | 0.3021077789025798 |
| 42 | Earlier candidate | 0.16129716838365343 | 0.16287027659432696 |
| 104732 | Earlier baseline | 0.30240434047870385 | 0.30280942387032667 |

For seed42 the parameter-set effect is much larger than the Hessian effect;
changing only Hessian slightly helps the baseline set but hurts the candidate
set. Do not attribute the earlier 51% full-run difference to regularization.
These losses are the diagnostic's inner holdout scores, not outer-test AMLB
scores. The runner checked finite, bounded, normalized probabilities before
printing each result. Unlike full AMLB runs, this probe does not save its
probability arrays, so these partial losses have not been independently
recomputed from prediction artifacts. Wait for all eight cells before
summarizing the factorial result or deciding the next experiment.
