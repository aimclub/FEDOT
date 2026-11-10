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
the tuner reported infinite initial and final fitness. The paired baseline
reproduced this as a CV evaluation timeout (see the final findings below);
it is **not** fixed by this candidate.

## Candidate: multiclass leaf-Hessian regularization

LightGBM defaults to a minimum leaf Hessian of 0.001. Small Hessians can allow
extreme Newton updates for rare classes. This candidate uses 1.0 for
single-output multiclass classification when the user has not specified this
parameter or any of its LightGBM aliases. Binary, regression and multioutput
defaults are unchanged. No task IDs, outer test labels, additional estimators,
or changes to the search time allocation enter this rule.

This is an experimental hypothesis, **not an established general AMLB improvement**.
The isolated train-only probe compares 0.001 and 1.0 on the same inner split
of the OpenML outer training fold. It is a mechanism check, not a benchmark.

Train-only Dionis fold 0 check (2026-11-10, copper): 374,569 outer-training
rows, 60 features, 355 classes; stratified 80/20 inner split, seed 42.
The 0.001 default produced inner logloss **3.192454**, selected iteration **1**,
and took **62.74 s**. Changing only the leaf-Hessian minimum to 1.0 produced
inner logloss **0.829552**, selected iteration **100**, and took **288.50 s**.
This supports the numerical-stability hypothesis, but also increases fit
cost. It does not establish an improvement over FEDOT's selected logistic
regression, nor does it resolve the infinite CV fitness symptom.

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

The initial Covertype invocation mistakenly requested the `medium` suite.
Both commands stopped with `Incorrect task name: Covertype` before training;
their failure logs are retained. The corrected invocation uses `large` and
separate suite-qualified run directories. The runner now validates membership
before creating `STARTED`. The actual model-source comparison remains parent
`2667e5df` versus candidate `0bbb1fe5`; later runner-only changes do not alter it.

## Final paired results (verified 2026-11-10)

All six intended runs have `COMPLETE` markers. No corresponding benchmark or
queue processes remained at the final inspection. No jobs were restarted and
no new expensive experiments were launched. The original PR and deployed
model code were not modified during monitoring.

Logloss below was independently recomputed from the saved prediction files,
not merely copied from the rounded AMLB scoreboard. Lower is better.

| Task / fold | Variant | Recomputed logloss | Job duration (s) | Training duration (s) | Selected graph |
| --- | --- | ---: | ---: | ---: | --- |
| Dionis / 0 | Baseline | 0.8345753647977301 | 2938.3 | 2870.0 | scaling -> logit |
| Dionis / 0 | Candidate | 0.8345753647977301 | 2889.7 | 2826.4 | scaling -> logit |
| fabert / 3 | Baseline | 1.1048747623578592 | 2602.5 | 2567.3 | scaling -> catboost |
| fabert / 3 | Candidate | 1.1027186183792097 | 2743.6 | 2714.5 | scaling -> catboost |
| Covertype / 3 | Baseline | 0.3035519522582302 | 1937.9 | 1902.9 | scaling -> (catboost, xgboost, lgbm) -> logit |
| Covertype / 3 | Candidate | 0.14845232889769483 | 1889.7 | 1861.1 | scaling -> (catboost, xgboost, lgbm) -> logit |

### Validation and provenance

Every probability was finite and in [0, 1]; every truth/predicted label had
a corresponding probability column. The probability columns were checked
against lexical class ordering before evaluating `sklearn.metrics.log_loss`
in bounded-memory batches with explicit labels. Recomputed scores agree with
the reported scores to their printed precision (absolute tolerance 1e-5).
Both job and training duration were below 3600 seconds for every run.

- Dionis: 41,619 prediction rows, 355 classes; maximum row-sum error
  8.89e-16 for both variants.
- fabert: 824 rows, 7 classes; maximum row-sum error 4.45e-16 for both.
- Covertype: 58,101 rows, 7 classes; maximum row-sum errors 6.67e-16
  (baseline) and 5.56e-16 (candidate).

The experiment stage on each host is
`/var/essdata/<user>/fedot-amlb-official-1h8c/experiments/pr1467-diagnostics-20261109/multiclass-quality-20261110`.
The valid run paths are `runs/{baseline,candidate}/Dionis-fold0` on copper,
`runs/{baseline,candidate}/fabert-fold3` on ruthenium, and
`runs/{baseline,candidate}/Covertype-fold3-large` on dysprosium.
Scores and predictions are under each run's
`results/fedot.multiclass-quality-<variant>.<task>.fold<fold>/` directory.
The earlier Covertype paths without `-large` are setup failures and excluded.

Prediction-file SHA-256 checksums:

- Dionis, both variants: `fe9a30a51d06f27d13bdedec8a9a26d8e1eb9fae5cf16e43abb0bc442668bace`.
- fabert baseline: `52b0d18c6b1f6d7a50ae9d95bb1695c17a29dae18f64ff3e8edac4a808033e75`.
- fabert candidate: `578ec157f3d319a46fe326cb48dc65d587da516e8b499cd2b74f5f4640a81cfe`.
- Covertype baseline: `77078473ea84fa639ed385a4b6652b5f0e652a4daac78e44371131ae0fd932e3`.
- Covertype candidate: `c248242705c913da4a5c879e829d8e6feeb107a8eb7285224d4c1277a3a9e726`.

The original harness directory resolves to a `ram-v2` installation on
copper/ruthenium and `ram-v1` on dysprosium. These are harness paths, not the
FEDOT model-source baseline: the recorded FEDOT import paths use the isolated
baseline/candidate archives. All three hosts have the same FEDOT harness
`exec.py` SHA-256 `3b8958e469c4558267ac751129616e2a4f629ece17d60a1963fdd904d25cc649`
and vendored GOLEM `tuner_interface.py` SHA-256
`29db83fe92a643b4a1a7747008c5865f91c3f8a6496497273acae1f3ab491097`.

### Interpretation and next priorities

**Dionis: no end-to-end improvement.** The predictions are byte-identical.
In the baseline, three-fold logistic-regression CV took about 444 seconds
during composition, but tuning allocated only 310 seconds per evaluation,
using the initial, faster LightGBM fit to estimate cost. The third fold was
terminated with only 30 seconds left at 07:27:23 UTC. The initial, trial and
final-check fitness became invalid/`inf`; this is not infinite test logloss.
In the candidate, initial LightGBM fitting took 363.8 seconds. Its subsequent
CV fit exceeded the 324-second per-fold limit and was rejected. Composition
returned logistic regression; tuning was skipped with zero allocated time
remaining. Stabilizing the tree updates alone therefore does not address the
dominant scheduling bottleneck.

**fabert: inconclusive small difference.** The observed relative logloss
reduction is 0.195%, with job duration increasing by 141.1 seconds. Neither
selected graph contains LightGBM. CatBoost `colsample_bylevel` differs
(0.23738628 versus 0.56648208), and the tuner completed two versus three
trials. This does not establish a direct benefit from the new leaf-Hessian
default; search-path and timing variability are plausible explanations.

**Covertype: promising single-fold result, not isolated causality.** The
observed relative logloss reduction is 51.095%, with job duration decreasing
by 48.2 seconds. The graph structure is the same and both tuners completed
two trials, but their sampled/chosen hyperparameters differ substantially:
LightGBM leaves 166 versus 252, learning rate 0.02981574 versus 0.11400404;
CatBoost feature fraction 0.40348418 versus 0.93386455; and meta-logistic C
8.50796568 versus 2.738182. Tuner CV loss changed from 0.313976 to 0.185377.
The shared benchmark seed did not produce identical hyperparameter proposals.
Do not interpret this pair as a fixed-parameter ablation or as proof of a
51% general improvement from the Hessian change.

Before promoting the candidate to PR #1467, prioritize (1) estimating tuning
cost from the selected graph rather than a cheaper initial model, (2) making
useful multiclass booster fits feasible within the per-fit allowance, and
(3) isolating the Covertype effect with matched hyperparameter proposals and
additional folds. These are follow-up proposals, not changes or new runs
performed as part of this monitoring task. The candidate remains experimental
on the separate branch; no improvement is being promoted to the original PR.

## Follow-up status: 2026-11-10 21:28 UTC

The isolated fold5 controls declared in
[the budget hypothesis report](pr1467_budget_hypotheses.md) are still active;
their candidates have not STARTED. These later experiments do not change the
initial paired results above or promote changes to PR1467.

Covertype5 baseline has finished finite tuning evaluations (final CV
0.2033320365023361) and is fitting the final model; outer predictions and
budget validation are pending. Fabert5 baseline selected CatBoost but its
86s tuner cap has already produced invalid initial fitness and two timed-out
proposals. Saved history is not available yet: two evolutionary CatBoost
evaluations have identical fitness but different log durations/parameters,
so selected-duration provenance must not be inferred from fitness alone.
Both frozen runtime manifests verify. No new jobs or code changes were made
in this monitoring pass; continue the existing queues before further series.
