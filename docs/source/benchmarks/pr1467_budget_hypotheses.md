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

## Completed follow-ups and next hypotheses: 16:25 UTC

### Headroom pair: no quality gain

The ruthenium candidate completed with independently recomputed logloss
**0.8051071883489818**, 1034.9 s job / 892.6 s training. All 41,619 x 355
probabilities pass finite/range/normalization checks (maximum error 8.44e-15).
Prediction SHA-256:
`ba685f7df7cc0521205a10b768ecdf4a20104ca6625892560b5a221b6c4d3353`.
Loss differs from its paired control by only 1.24e-12. Both skip composition
and tuning, both hit the 324 s final-refit cap and return their initial LGBM.
The shorter duration alone does not establish a speed improvement.

### Completed fixed-graph factorial

All eight fits completed; parameter dictionaries were independently checked
to differ only in leaf Hessian within each seed/source pair. The last cell
(candidate parameters, seed104732, Hessian1) has inner loss
**0.16285091880539448**, versus **0.16392230968462296** at Hessian0.001.
Mean loss over the two seeds: baseline/candidate parameter sets
**0.3025273926 / 0.1626097390** at Hessian0.001 and
**0.3024586014 / 0.1628605977** at Hessian1. Parameter selection dominates in
this train-only diagnostic; the Hessian effect is small and inconsistent.
JSON SHA-256: `51b046f9c2baffad337ee777f3fc85fd114d31e93cd93a16096ac486b4e14324`.
Raw probabilities were not saved by this completed version; no claim of
independent probability-level recomputation is made for this probe.

### Next bounded series (predeclared)

- Ruthenium, stage `remaining-refit-20261110`, Dionis0: one new candidate
  using `final_fit_use_remaining_time=true`, compared with the just-completed
  caddb242 headroom control. Do not rerun that control. The new flag is opt-in,
  default false; preserves the CV cap, uses elapsed wall time since API.fit
  began, and reserves 5% of the API budget (minimum 5 s) for post-fit work.
  The full AMLB 3600 s limit is unchanged. Hypothesis: the full-data fit can
  finish when allowed to use otherwise idle budget. Record actual refit
  success, graph, quality and total runtime; require another fold if promising.
- Dysprosium, stage `covertype-components-20261110`: six fixed-graph train-only
  fits, seed42, Hessian0.001 throughout. Two complete frozen parameter sets
  plus four one-operation swaps (catboost/xgboost/lgbm/logit) into baseline.
  No new parameter search. Hypothesis: isolate which parameter block drives
  the earlier effect before proposing a general search-space change. Save
  each probability array and labels for independent score recomputation.
- Copper: leave the existing baseline/candidate budget queue running.

Original PR, old archives and finished artifacts remain untouched.

Both new series use immutable revision
`95d90fdc64f8827f759471f5e70a8651daa48251`, archive SHA-256
`dc0006103566b267f238adbe24176a2422e6e85e65e0848ab97e02fd4302e880`.
Local checks: 15 final-refit/fallback tests, 28 budget/preparation tests and
one component-probe test passed (the groups overlap). On each new host,
10 focused tests passed before launch. The minimal archive omits package
initializers for examples, so the probe test was explicitly loaded from its
frozen file path to avoid the installed examples package shadowing it.
No deployed model source was changed for this test correction.
Ruthenium launched candidate-only at about 16:33 UTC; dysprosium launched
the component queue after its preceding factorial had completed.

## Interim validation: 16:47 UTC

Copper's regularized control completed: independently recomputed logloss
**0.8345753647977304**, job **2897.3 s**, training **2840.5 s**. All
41,619 x 355 probabilities were finite, in [0,1], maximum normalization error
1.12e-15. Its prediction SHA-256 is the same as the earlier logit runs:
`fe9a30a51d06f27d13bdedec8a9a26d8e1eb9fae5cf16e43abb0bc442668bace`.
The chosen graph is scaling/logit; tuning was skipped. Copper's measured-CV
candidate started at 16:34:22 UTC and is still composing; no paired conclusion
is available yet.

Ruthenium's remaining-refit run is still active. The harness recorded
`final_fit_use_remaining_time='true'` and the API log explicitly allocated
**2512.1 seconds** to final refit at 16:43:21 UTC. The 324 s CV cap was not
changed. This verifies activation of the intended policy, not refit success
or a quality improvement; those remain pending.

Four completed Covertype component cells were independently recomputed from
their saved NPZ probabilities and identical validation labels (104,583 rows,
7 classes). All passed finite/range/normalization checks; maximum row-sum
error was 6.67e-16. The remaining two cells are still pending.

| Parameter source | Independently recomputed inner logloss |
| --- | ---: |
| Baseline | 0.3026504446866336 |
| All candidate blocks | 0.16129716838364516 |
| Only CatBoost block replaced | 0.30011858740622166 |
| Only XGBoost block replaced | 0.31586872331841176 |

Probability NPZ SHA-256, in the same order:

- `f00e8167124aab0432ebfce3c28d4630d52732fe066bc500bbb05512f4c5eb75`
- `167af842c0dee10d2bf69f4ba533432da419bc87d3ad20f0787dbdf4993aa2d0`
- `c90838dbd7fb753c7223fd9f6e2273f62f645cf76aca9d0439bc86274e14ff96`
- `839bdf148b8d48ad24c8300c8f5adcafd14202d8be6eb87d648e55a6bb7320e2`

Neither the CatBoost-only nor XGBoost-only replacement explains the large
all-block effect. Wait for LightGBM/logit cells before identifying a driver;
these are still train-only diagnostics, not full AutoML benchmark gains.
No new experiment was launched during this inspection: all three servers
have active work, and the outstanding results determine the next hypothesis.

## Completed refit/component checks: 17:08 UTC

Ruthenium's opt-in remaining-budget refit completed without fallback. The log
allocates 2512.1 s at 16:43:21 and reports `Final pipeline was fitted` at
16:55:41, with the same scaling/LightGBM graph. Independently recomputed
Dionis0 outer loss is **0.8021429238192925**, versus the same-host completed
control **0.8051071883489818** (0.368% lower). Job/training time increased
from 1034.9/892.6 s to **1427.1/1306.8 s**, both within 3600 s. The harness
recorded the true flag; all 41,619 x 355 probabilities are finite, bounded,
normalized (maximum error 8.44e-15). Prediction SHA-256:
`3e2f203ceb3706a34acadbd5a511769899de997c8db0b0cce3c2e57fa68253b6`.
This supports successful full refit on one fold, not general quality gain.

All six Covertype component cells are COMPLETE. Independent recomputation
of the last two saved arrays gives **0.16127422159206897** for LightGBM-only
replacement (148.99 s) and **0.3017539731968004** for logit-only replacement
(152.26 s). All cells have identical validation labels and pass probability
checks, maximum row-sum error 6.67e-16. Parameter dictionaries were checked:
each swap changes exactly the named operation and uses its candidate block.
Thus the LightGBM block accounts for the large effect in this fixed-graph
train-only probe. Neither a specific LightGBM parameter nor a general search
policy is yet established. JSON SHA-256:
`eba53b40f0c5f47fc41a5be0039436725c7f1983ffaacc3921f0cc1d4ce3da80`.
Last two probability SHA-256 values:

- `7d8ae36ed2e34916da1b9f62bf8a577840c90631c005b9b1a29b1bbfc82c2d5c`
- `eb3b4469077c9d3ffc90c8905c29139f584db12ac1eef19ed178c16d46a9e4c1`

Copper's measured-budget candidate remains active; do not duplicate it.

### Next bounded tests (declared before launch)

- Ruthenium, new stage `remaining-refit-fold1-20261110`: Dionis fold1,
  sequential baseline/candidate with **identical model source 95d90fdc**;
  only candidate enables `final_fit_use_remaining_time=true`. Same 1h8c
  settings, seed104730. Baseline retains the default false flag. Require
  independent score checks, successful refit, no fallback and <=3600 s;
  record any runtime-dependent search differences rather than attributing
  every score change to refit. No fold0 control is rerun.
- Dysprosium, new stage `covertype-lgbm-parameters-20261110`: nine train-only
  fixed-graph fits, unchanged seed42/Hessian0.001/inner split. Controls are
  the frozen baseline and baseline with the full candidate LightGBM block.
  Seven one-parameter replacements are learning_rate, num_leaves,
  min_data_in_leaf, colsample_bytree, reg_alpha, reg_lambda and subsample.
  All other blocks stay baseline; save each probability array. Hypothesis:
  the larger learning rate helps the fixed 100-tree horizon, but test all
  changed fields before selecting a cause or range. This is not a search
  space change and never evaluates outer-test labels.

No general defaults or original PR code are changed by these new tests.

Both new queues launched at **17:12:32 UTC**, after confirming the preceding
queues and benchmark processes had exited. Ruthenium's immutable model is
`95d90fdc64f8827f759471f5e70a8651daa48251` for both variants (archive SHA
`dc0006103566b267f238adbe24176a2422e6e85e65e0848ab97e02fd4302e880`).
Dysprosium's diagnostic revision is `113a8dc0c7ae17554fcf6f00f83e71cfbdada27a`
(archive SHA `b4bdccaa60169ea4df14dae96dd2857b42329035a0aca4d206f4dd618a0cf51d`);
its FEDOT model source is unchanged from 95d90fdc. Archive hashes and LF
shell syntax were checked before launch. Fifteen focused local tests passed;
remote preflight passed 10 tests on ruthenium and 12 on dysprosium, explicitly
loading the diagnostic module by path to avoid installed-package shadowing.
No model defaults were changed in the new diagnostic revision.

## Interim inspection: 17:26 UTC

### Copper measured-budget pair completed: no prediction gain

Both variants are COMPLETE and no benchmark process remains. Candidate
`6f05a1ae` has independently recomputed outer-test loss
**0.8345753647977304**, job **2903.3 s**, training **2849.3 s**. Its
41,619 x 355 probabilities pass finite/range/normalization checks (maximum
error 1.12e-15), and the prediction file is byte-identical to the control:
`fe9a30a51d06f27d13bdedec8a9a26d8e1eb9fae5cf16e43abb0bc442668bace`.
Control job/training time was 2897.3/2840.5 s. No quality or speed improvement
is established by this pair.

The candidate did exercise the measured-budget branch. At 17:19:04 it logged
**193.4 s per fold**, **135.5 s available for tuning**, **485.3 s refit reserve**,
then skipped infeasible tuning rather than evaluating invalid tuner fitness.
Initial LightGBM again hit the 324 s CV cap; scaling/logit had finite CV loss
0.85957458, was selected, and completed final refit. Composition consumed
37.398 minutes, leaving insufficient time for useful tuning. This pair does
not show recovery of the earlier infinite-fitness tuning case: both variants
skipped tuning, and neither tested a completed measured-budget tuning trial.

Audit limitation: no serialized optimization history was saved by the frozen
runner. The candidate run and stage contain no history artifact, so exact
history-individual/graph-ID matching and its metadata duration cannot be
independently reconstructed from artifacts. The log proves activation, and
source/tests check matching logic, but these are not a substitute for the
missing history. The **future-run** shell runner now requests the existing
harness `_save_artifacts=info` facility to save history. No active/deployed
runner is edited, and no completed run is repeated just to add artifacts.

### Covertype parameter ablation: five of nine cells checked

All five completed cells were independently recomputed from saved NPZ files;
validation labels are identical (104,583 rows, seven classes), finite/range/
normalization checks pass, and each named single-field variant changes only
that field. All other model blocks, seeds and Hessian are identical.

| Variant | Inner logloss | Fit + prediction (s) |
| --- | ---: | ---: |
| Baseline | 0.3026504446866336 | 154.62 |
| Full LightGBM block | 0.16127422159202612 | 152.33 |
| Only learning_rate: 0.02981574 -> 0.11400404 | 0.21095839154652385 | 149.80 |
| Only num_leaves: 166 -> 252 | 0.2729749987622811 | 161.04 |
| Only min_data_in_leaf: 10 -> 93 | 0.3035219087028718 | 153.83 |

Learning rate is a substantial contributor, but alone does not reproduce the
full block's effect. Wait for the remaining four fields before selecting a
range or testing interactions. These remain train-only diagnostics, not full
AutoML scores. NPZ SHA-256, in row order:

- `f00e8167124aab0432ebfce3c28d4630d52732fe066bc500bbb05512f4c5eb75`
- `aed65e95c86bcda85f7c8a4a21e92edf0b7a6885faebf9a4042454793ec477c5`
- `87717991022d01678543b14165748c721f63a53aed66827de23886bdec75e322`
- `38541d6eab3967b5219755397d388b66b8660cbdda48a3e6f74e9064a7ff9bf4`
- `bbb80079011a1e0bea0ba78d0bf980608ff37385fd81f382d2539c656831ef0f`

Ruthenium fold1 baseline remains active; its initial fit took 558.3 s and it
skipped composition/tuning, like fold0. No paired fold1 score is available.
Copper is free, but no new expensive run is started before the outstanding
parameter/refit evidence determines a justified next test.

## Parameter ablation completed: 17:46 UTC

All nine Covertype cells are COMPLETE and the probe process has exited.
All NPZ probabilities and labels were independently checked; all single-field
parameter controls are exact. The last four inner losses are:

- colsample_bytree only: **0.28814437112154057**, 159.38 s;
- reg_alpha only: **0.3029925662822302**, 156.01 s;
- reg_lambda only: **0.3020101001475171**, 153.19 s;
- subsample only: **0.3026504446866336**, 153.16 s.

Subsample-only predictions are byte-identical to baseline. In these frozen
parameters bagging_fraction is also explicitly set; the isolated subsample
change therefore should not be assumed to control effective row sampling.
Learning rate contributes the largest single-field improvement, with leaves
and feature fraction also helping. No single replacement reaches the full
LightGBM block's loss; do not assume independent/additive effects.
This is one inner split of outer fold3, not a general AMLB improvement.

Final JSON SHA-256:
`c7046e04fc387398b6be71ddbf081ad4c28e3875784f8da3fe9cf0f250ba0c45`.
Last four NPZ SHA-256, in the above order:

- `ec4ad56761fd6ff012b3b0a0f5ed3d6ce0ed3374de8247514469ae29beb062db`
- `559663518926b20eb60f1707141bcf492824be1bfb8f3629d23123200f936390`
- `756a967b16534c6750caac0bea62c36322162e3cfe4dc7748af57a51fcfaaab2`
- `f00e8167124aab0432ebfce3c28d4630d52732fe066bc500bbb05512f4c5eb75`

Ruthenium fold1 baseline is COMPLETE: independently recomputed loss
**0.8207629887502289**, job **1045.0 s**, training **892.5 s**. The 41,619 x355
probabilities are finite, bounded, normalized (max error 8.66e-15); SHA-256
`597358e8f5d04b6185d61b94ecad5946924587566d0b72355084d721162281d6`.
Like fold0, final fit exceeded 324 s and fell back to the initial LGBM.
The remaining-refit candidate started at 17:30:20 UTC and is still running;
there is no completed fold1 pair yet.

### Next bounded hypothesis: learning rate versus tree horizon

Dysprosium, new stage `covertype-lr-horizon-20261110`, same train-only fold3
and seed42/Hessian0.001: eight fixed-graph fits. Cross learning rates
**0.029815738804900536 (exact baseline), 0.06, 0.10, 0.15** with explicit
**100 / 400 trees**. All other parameters and the inner split remain baseline.
Save probabilities plus actual fitted-round/best-iteration metadata.
Hypothesis: some of the learning-rate effect reflects underfitting at the
fixed 100-tree horizon. This distinguishes allowing more rounds from simply
raising learning rate, and tests a small prespecified range rather than a
general default change. Check runtime as well as loss. No outer-test labels
enter this test; require another fold before proposing a search policy.
No model source or active ruthenium runner is changed. Copper remains free
for the subsequent fold/control once this bounded range check is available.

The horizon queue uses diagnostic revision
`6e0037bfc891b3309c37eef12227572781dbba72` (FEDOT model code unchanged), archive
SHA-256 `4788466a800d6fd7ecfdc6dced7e4502ad1adec05b86d7182b6577490621a4f9`.
Six current probe tests passed locally, including a real LightGBM pipeline
round-count check; the three preparation tests also passed. All nine passed
on dysprosium before launching. The initial preflight collection failed
because the installed examples package shadowed the preparation module;
both diagnostic modules were then loaded by their exact frozen paths.
Both preflight logs are retained; no deployed source was edited and no
training run was repeated. The queue was launched after confirming that the
preceding probe and all benchmark processes on dysprosium had exited.

## Second refit fold and partial horizon check: 18:06 UTC

Ruthenium's Dionis1 pair is COMPLETE, with no remaining benchmark process.
Candidate loss independently recomputed from predictions is
**0.8093934665168628**, versus **0.8207629887502289** for baseline (1.385%
lower). Job/training time is **1476.1/1342.1 s**, versus 1045.0/892.5 s.
All 41,619 x355 probabilities are finite, bounded and normalized (max error
8.66e-15). Candidate prediction SHA-256:
`7a1b3b5686380332edb211768055cbcbae82c305aaf416696a54f7c93e932c3c`.
The harness flag was true. Initial fit took 546.3 s; both variants skipped
composition and tuning. The candidate allocated 2523.5 s to final refit at
17:40:12, finished it at 17:53:19, and retained scaling/LightGBM with no
fallback. Thus the refit mechanism and modest quality benefit replicate on
two Dionis folds on this host, at increased runtime. This is not evidence of
a general multi-dataset gain or behavior after expensive composition.

Five of eight horizon cells on dysprosium have been independently checked
from NPZ: identical validation labels, finite/range/normalization pass, only
learning_rate/n_estimators differ, requested and fitted parameters agree.
All five models selected the last allowed iteration.

| Learning rate | Trees | Inner logloss | Fit + prediction (s) |
| --- | ---: | ---: | ---: |
| 0.029815738804900536 | 100 | 0.3026504446866336 | 158.14 |
| 0.06 | 100 | 0.2517127128261797 | 159.24 |
| 0.10 | 100 | 0.21875727484530932 | 156.37 |
| 0.15 | 100 | 0.19356271699303704 | 156.57 |
| 0.029815738804900536 | 400 | 0.19274855201231036 | 203.93 |

The baseline learning rate improves substantially with more trees, supporting
underfitting at the 100-tree horizon. The remaining three 400-tree cells are
still pending; do not select a policy from this partial grid.
NPZ SHA-256, in row order:

- `f00e8167124aab0432ebfce3c28d4630d52732fe066bc500bbb05512f4c5eb75`
- `bd324f379edf64ee76437ab137108d213c1dba509b32d9cecfc693f0ce197cd0`
- `65efd55ee66682ec79cbe4d99bd1b0371523e13e3f66e6aafc8e7b1db516ff4d`
- `6b3d10cc3a73bb133d65cb9fa263b7651652433251415d6ece68b39dcce25157`
- `dd79e5b5680f38d908ee2dc021eaf5454ec241710800d6adfd4f8232b24dcb5c`

### Next bounded refit check (declared before launch)

Copper, new stage `remaining-refit-fold2-20261110`: Dionis fold2,
seed104731, sequential baseline/candidate under unchanged 1h8c conditions.
Both use identical FEDOT source (model unchanged from 95d90fdc); only the
candidate enables remaining-time final refit. Both request history artifacts
using the newer diagnostic runner. Hypothesis: a refit policy that works when
composition is skipped also stays within budget and avoids harmful fallback
after search. Copper previously ran composition on fold0, but this is not
guaranteed for fold2; inspect actual execution before attributing an effect.
Compare the new same-host pair only, not copper versus ruthenium as an
ablation. No old control is repeated. Ruthenium is free for the subsequent
Covertype validation when its complete range check is available.

The new copper pair uses identical source revision
`44262616ed4b84178a8d5ff779043fb7a7362ebe`, with no FEDOT model-code diff from
95d90fdc. Archive SHA-256:
`b866d7f4b043590e18f41762920e286c2415506185c0ff0f65e82fbdfafeca4a`.
Nine final-refit tests passed locally and on copper; shell syntax, archive
hash, harness and pinned tuner hashes were verified. The queue was launched
after checking that no benchmark/probe/queue process was active on copper.
The completed ruthenium fold1 pair's truth rows and class-column order were
also explicitly compared and match for all 41,619 predictions.

## Horizon grid completed: 18:26 UTC

All eight cells are COMPLETE; no probe process remains. Independent NPZ
recomputation confirms the remaining 400-tree losses:

| Learning rate | Inner logloss | Fit + prediction (s) |
| --- | ---: | ---: |
| 0.06 | 0.15657376978545082 | 197.94 |
| 0.10 | 0.14689176235612272 | 201.88 |
| 0.15 | 0.14587823529584684 | 200.95 |

All eight share identical validation labels, have finite/bounded/normalized
probabilities, and differ only in learning_rate/n_estimators. Fitted rounds
and best iteration equal the requested 100 or 400 in every cell. The effect
is not solely a preference for a higher learning rate: allowing more trees
helps at every tested rate. This supports underfitting at the 100-tree limit
for this graph and split, but does not establish a safe general runtime or
default. In particular, increasing initial-model cost can change FEDOT's
composition/tuning decisions, as already observed on Dionis.

Full JSON SHA-256:
`eb7bfa4ea1974f2e0f8bc9ca4b8a78f700d9e1b13b76647e6ce45dd557fc6c24`.
Last three NPZ SHA-256, in row order:

- `5230a46d98cac5cb678a22a2b5d0886d41f67450ad5b198efa8857b61ba8f0cc`
- `1b8b597e72b2c456833ddcf8c4f0187a462072f11b680461b83d18840dbc60b7`
- `bc2afa6134e2b6c845126ef43b0e38e02d5865c4e1a3d21e1d27a5cdbcb04f8a`

### Prespecified additional-fold confirmation

Dysprosium, new stage `covertype-horizon-fold4-20261110`, four train-only fits
on outer-training **fold4**. Keep the original frozen baseline graph and all
other parameters from fold3's logged graph (no new tuning). Cross the exact
baseline rate 0.029815738804900536 / **0.10** with **100 / 400** trees, seed42,
Hessian0.001, same inner split policy. Rate0.10 is a standard interior value
with a loss close to 0.15 at 400 trees; do not chase the grid's smallest loss.
This tests both main changes and their interaction on another fold, while
saving probabilities and actual iteration metadata. It is still not an
outer-test score or independent multi-dataset evidence. Training folds overlap
as usual in cross-validation; treat this as replication, not a fresh dataset.
The runner now accepts an explicit preparation fold and validates task/fold/
split/cell count before marking COMPLETE. Previous deployed runners and all
old results remain unchanged. Full end-to-end budget/regression checks are
still required before any model-default or search-space proposal.

Copper's Dionis2 baseline is still active; its sequential candidate remains
queued. No active or STARTED job is restarted.

The fold4 probe uses immutable diagnostic revision
`17f6cb6aa97b8126490cd7591affa460ff32d93b`, FEDOT model unchanged from
95d90fdc. Archive SHA-256:
`233a48e5f88075a72e67c74c7d98bc01f9909abc644e8a198e1294a079dc56d4`.
Ten focused tests passed locally and on dysprosium before launch, including
the four-cell controls and real LightGBM round metadata. The archive hash
and LF shell syntax were checked. The command uses `confirm 4` explicitly.
Copper's harness confirms `_save_artifacts='info'`; initial fit took 383.2 s
and composition was entered, unlike the earlier ruthenium refit pairs.

## Additional-fold confirmation and next end-to-end test: 18:50 UTC

Dysprosium's fold4 confirmation is COMPLETE, with no remaining probe process.
All four saved NPZ files were independently recomputed: identical 104,583
validation labels / seven classes, finite values in [0,1], maximum row-sum
error 6.67e-16. Only the declared learning rate and tree count differ.
Requested rounds, fitted rounds and best iteration agree in all four cells.

| Learning rate | Trees | Inner logloss | Fit + prediction (s) |
| --- | ---: | ---: | ---: |
| 0.029815738804900536 | 100 | 0.302819962108981 | 161.46 |
| 0.10 | 100 | 0.2193775165508452 | 163.58 |
| 0.029815738804900536 | 400 | 0.19196571955887998 | 205.23 |
| 0.10 | 400 | 0.14631012443572525 | 203.37 |

JSON SHA-256: `0ca9474ca6f8ba221b2c3372e16b5210d68c868071f41e5ca541aa2c401c2776`.
NPZ SHA-256, in table order:

- `0fc9a9a644257626ed7b4048ced83c24e352b9773b9cee3a1e0ca90f469debdd`
- `b972a19de42831cfaeaacdca75cc3bbe56e9f7f6d0b31e428121b0f55acda3bd`
- `3e41510cf49ecb81283646001f9de9eb5d5039105e6c58354ad8de3c8e290dde`
- `aefebc87a7fdc6ac8da72da5f30e7840b5ca6fefd8b5ea238d7075b6aa784e8b`

This replicates the underfitting mechanism on another overlapping training
fold, not a full AutoML or multi-dataset improvement. The frozen five-node
graph and Hessian0.001 also differ from the next full-run search setting.

### Predeclared bounded end-to-end series

Stage `lgbm-horizon-e2e-20261110`: one sequential baseline/candidate pair on
each of two free hosts, **Covertype fold4 / large on dysprosium**, and
**fabert fold4 / small on ruthenium**. Copper's active Dionis2 pair is left
untouched. Identical new source revisions for baseline/candidate; the only
variant is an opt-in operation-default overlay setting classification LGBM
`n_estimators=400` instead of the library default100. Learning rate is NOT
overridden (the native default is already0.1; tuning remains free to sample
it). Early stopping30, native deadlines, multiclass Hessian policy, search
space, API budget and final-refit defaultfalse remain identical. Explicit
node parameters still override defaults. LGBM regression is unchanged.

This is an experiment profile, not a change to general defaults. The sparse
profile `examples/advanced/lgbm_400_defaults.json` is activated only by the
absolute-path environment variable `FEDOT_OPERATION_DEFAULTS_OVERLAY`,
inherited by framework/worker processes. Invalid profiles fail loudly;
cached built-in defaults are never mutated. The future runner clears any
ambient overlay for baseline, enables the frozen profile only for candidate,
and records path, hash and effective defaults in environment.txt. Existing
deployed runners/sources are not edited.

Hypothesis: a longer horizon improves end-to-end Covertype loss without
exceeding the one-hour budget; fabert screens for search/runtime regressions
on another multiclass task. Increased initial-fit cost may instead suppress
composition/tuning, or native deadlines may prevent400 rounds. Either is
a valid negative finding. Do not attribute differences solely to tree count
if graph selection or sampled tuning proposals diverge. Retain full history,
inspect graph/parameters, initial time, CV failures, final fallback and actual
round metadata where artifacts expose it. Independently check predictions,
paired labels/classes and both job/training duration <=3600s. All four runs
use 1h8c, seed104733, portfoliofalse, best_quality, tuningtrue, sequential,
n_jobs8, `_save_artifacts=info`. No prior run is repeated. This screening pair
is insufficient to promote a general default or update original PR1467.

The two new pairs use identical immutable source revision
`30142fca3293be94125505a6dbe590c380631a3c` for baseline and candidate.
Archive SHA-256:
`50a04a8d911beb8f69b22067d656b94134d9f826e0727a3d93d5f1dbd9ae6f36`.
Forty-seven focused local tests passed; 42 tests passed separately on each
new server, including worker inheritance of the overlay, explicit-parameter
precedence, unchanged regression defaults, multiclass regularization and
native time limits. Archive, harness and vendor hashes were verified.
The initial ruthenium preparation rejected the proposed medium suite before
creating the stage or any STARTED run: fabert belongs to small. The corrected
suite is recorded above; no training was attempted or repeated for medium.
Dysprosium's queue launched at **18:53:47 UTC**, ruthenium's at
**18:54:11 UTC**, after checking no preceding benchmark/probe/queue process
was active. Each has a new STARTED baseline and an exclusive queue; candidate
waits for baseline. Copper's original Dionis2 baseline is still active at
18:54 UTC and has not been restarted or changed.

## Copper fold2 control verified; history provenance issue: 19:10 UTC

Copper baseline `44262616` is COMPLETE. Independent batched recomputation
gives outer-test logloss **0.8521750855647503**, job **2965.2 s**, training
**2904.1 s**. All 41,619 x355 probabilities are finite, in [0,1], normalized
(maximum row-sum error 8.89e-16); labels and class order are valid. The score
matches the scoreboard to its printed precision. Prediction SHA-256:
`7920d079098d2720c7a06e2aec126d120e17ab51d5c7c6ae9a6db3a9960b8d7e`.
SHA-256 of UTF-8 truth labels joined with a trailing newline per row:
`76e725fac8908419462d93a20509cfb696333a57d9198c06581fd392b68c55f1`.

Composition evaluated both initial graphs with finite fitness: scaling/LGBM
CV loss1.0774162188759064, scaling/logit0.8575016046202981. No infinite tuner
fitness was produced: tuning was skipped as infeasible. Selected scaling/logit
completed final refit at18:59:27 without fallback. The log contains a
`multiprocessing.resource_tracker` cleanup KeyError for a loky semaphore;
the job nevertheless completed and saved valid predictions. Do not describe
the log as entirely error-free or interpret this cleanup warning as a fit
timeout. Candidate automatically started **19:00:02 UTC**. Its harness true
flag is confirmed; initial fit took398.7s and composition started19:07:52.
There is no completed same-fold pair yet.

### Exact selected graph matched, but shared metadata contaminates other graphs

Saved `info/Dionis/2/history.json` SHA-256:
`21f35fa8907f43ffa7ded6ede575bd370d32ef56d6cbd0e926e09a3ef40d4839`.
The selected graph's operations, parameters and edges match `_tuning_result`
and final_choices individual `a70ecd85-dc81-4a33-abaa-391c1d2945ba` exactly.
Its recorded CV duration925.1078804610297s /5 = **185.02157609220595s/fold**,
agreeing with the logged185.0. The log independently brackets logit evaluation
at18:40:12.556--18:55:37.439 (924.883s, excluding dispatcher overhead).
Measured tuning allocation74.3s and refit reserve498.2s led to skipping tuning.

However, LGBM's separate history individual has the **same duration and
evaluation timestamp**. Its own log brackets evaluation at
18:17:35.811--18:40:12.305 (1356.494s), not925.1s. A read-only in-memory
reproduction against the pinned server GOLEM confirms the cause:
`EvoGraphOptimizer` creates individuals using the same
`requirements.static_individual_metadata` dictionary; `Individual` retains
that reference and `set_evaluation_result` updates it in place. Giving two
such individuals10s and20s evaluations makes BOTH report20s and also writes
20s into the static template. Mutation/crossover constructors also reuse the
template. Individual source SHA-256:
`7b0d1de39ad586774883025bc1c690ac1539634a9f71b43a249a059c123af527`.

Thus exact graph matching alone does not establish duration provenance in
these frozen runs. Here the chosen logit happened to be evaluated last, so
its measured cost is corroborated independently; LGBM's metadata is wrong.
For all subsequent active results, compare selected-graph timing with log
boundaries and flag shared/colliding metadata. Do not retrospectively repair
history files or claim the earlier runs are all affected in the same way.
The existing unit tests for exact-ID matching did not cover shared metadata.

Potential next correction, after evaluating the current pairs: isolate
individual metadata when applying evaluation results (copy-on-write instead
of mutating the shared static template), with regression tests for initial
population, later offspring, sequential/parallel evaluation and serialized
history. Any GOLEM correction must use a NEW immutable isolated vendor copy,
record its hash and retain the existing vendor for all active runs; no global
monkeypatch or in-place deployed fix. First prove the bookkeeping correction
with cheap tests before deciding whether another full paired run is needed.
No new training or deployed-code change was made during this inspection.

### Horizon screening pairs remain active

Ruthenium fabert4 baseline initial fit76.7s, composition entered18:56:36.
Dysprosium Covertype4 baseline initial fit218.8s; composition was skipped
(five-fold estimate1094.0s), but tuning started with two folds, a513s trial
cap and683s search allowance. These are execution-path observations, not
quality results. Both candidates remain queued. The shape of the Covertype
end-to-end search is already different from the frozen five-node train-only
probe, reinforcing why its inner losses are not full AMLB improvements.

## Horizon control completed on Covertype: 19:28 UTC

Dysprosium baseline Covertype4 (`30142fca`, no defaults overlay) is COMPLETE.
Independent recomputation yields outer-test logloss **0.1587776834879601**,
job **1888.0 s**, training **1855.8 s**. All58,101 x7 probabilities are finite,
bounded and normalized (maximum error6.67e-16); class columns are ordered
1--7 and all truth/predicted labels are valid. Both durations are below3600s.
Prediction SHA-256:
`c46072f3ed6b7dda5c7e73d2ad3d94713b621351782eef41b8c220efcb6dacce`.
Truth-label SHA-256 (UTF-8, newline after each label):
`ec782fe1951ad22a62bf3c8fac5fd70e26850addd7c79be9b190add3e711c684`.

Although composition was skipped, tuning did run on the initial five-node
stack `scaling -> (catboost, xgboost, lgbm) -> logit`. Two trials completed.
History fitness values and independently bracketed log durations are:

| Evaluation | CV logloss | Log-bracketed duration (s) |
| --- | ---: | ---: |
| Initial graph | 0.3290292773165658 | 304.127 |
| Trial0 | 0.23356965076812622 | 332.043 |
| Trial1 | 0.19756742384484072 | 342.108 |
| Final check | 0.19756742452165307 | 338.434 |

All evaluations finished within the513s evaluation cap, with no invalid/inf
fitness. Final refit succeeded at19:25:42 without fallback (about211s after
tuning finished). A loky semaphore cleanup KeyError appears in the resource
tracker; as in the copper control, it did not prevent valid completed output.

Saved history SHA-256:
`eea41dfc18f141f066ec1e333b378a1449d33b4f41d6ceb7d9c9903e5c7f3316`.
The final result individual `16013ca6-16a9-41f5-8dd9-b60e77fb6b1d`
matches `_tuning_result` in graph structure/parameters; all final operation
parameters also match the final-graph log. Chosen LGBM learning_rate is
0.13925650312807544, num_leaves253, min_data_in_leaf80; logit C7.023524726465261.
These are new tuning proposals, not the frozen train-only graph parameters.
The four tuner records have distinct timestamps and **no**
`computation_time_in_seconds`; do not manufacture those durations from the
history or conflate tuner records with the shared evolutionary metadata bug.
There was no measured selected-graph composition-CV branch on this run.

Candidate automatically STARTED at **19:25:56 UTC**; the queue and benchmark
process are live. Its environment records the isolated overlay and effective
`n_estimators=400`; profile SHA-256:
`3707fa7def594cdefa9dfb27c82a941390259edfd9ad2ab774f868848e5781a0`.
This confirms configuration, not400 fitted rounds or a completed improvement;
inspect the candidate's graph/model evidence when available. Neither the
baseline's score nor earlier train-only losses establish the paired effect.

Copper's Dionis2 candidate remains in composition; ruthenium's fabert4
baseline is still active. No jobs were restarted, no deployed code/results
were edited and no additional series was launched. The monitoring log's
96.7% disk-use number was checked against the actual run/tmp filesystem:
`/var/essdata` is74% used with about249TiB available, not out of space.
