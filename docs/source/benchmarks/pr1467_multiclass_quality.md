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

## Follow-up status: 2026-11-10 21:49 UTC

Covertype fold5 baseline is COMPLETE and independently validated: logloss
**0.1761394325581716**, job1899.3s / training1869.9s,58101x7 finite normalized
probabilities, successful final refit without fallback. History's final graph,
parameters and edges match the selected tuned graph; exact checksums and
provenance are recorded in the budget hypothesis report.

Its existing candidate started21:31:09. The400-tree profile now demonstrably
reaches the actual framework exec process AND appears as n_estimators400 in
the initial CV graph. Initial CV is finite at0.2277421901954615; composition
is skipped and tuning continues with471s/evaluation,628s search. This confirms
activation, not400 fitted rounds or final quality improvement; paired outer
predictions and budget checks remain pending.

Fabert5 baseline still runs with14 completed invalid/timeout proposals and
best tuner fitnessinf; its metadata-patched candidate is queued, not started.
Do not conflate tuner fitness with outer-test loss. Both runtime manifests
remain intact. No jobs, experimental code, deployed sources or PR were changed.

## Follow-up status: 2026-11-10 22:12 UTC

The correctly transported Covertype5 horizon pair is COMPLETE. Independent
paired outer logloss is **0.1761394325581716 -> 0.20113041208684942**
(14.188% worse); job/training times are1899.3/1869.9s ->1769.5/1693.5s.
Both budgets pass, paired truth/class order matches, and all probabilities
are finite and normalized. The candidate really activates400 trees, with
explicit best-iteration400 evidence for three logged CV fits, but exact rounds
of the full-data refit are not saved. Both final refits succeeded.

Candidate performed one tuner proposal versus baseline's two and returned
the initial stack after its proposal failed to improve CV. Different search
parameters/outcomes prevent a fixed-parameter tree-count interpretation.
This is a negative end-to-end screen, not evidence for promoting400 as a
general default or a general speed claim. Checksums and timing are in the
budget hypothesis report; the old confounded fold4 pairs remain separate.

Fabert5 baseline is COMPLETE: independent logloss1.0498586672813428,
job3224.2s/training3198.1s,824x7 valid probabilities and successful untuned
CatBoost refit despite invalid initial/15trial/final tuner fitness. Saved
history selects the original CatBoost UID, whereas the later equal-fitness
offspring used cached predictions on all three folds. All14 evolutionary
records still share the last RF's overwritten timing. The metadata-patched
candidate started21:53:13 and remains in composition; its result is pending.
No new runs, deployed/shared-source changes or PR/default promotion occurred.

## Follow-up status: 2026-11-10 22:27 UTC

Fabert5 metadata candidate is still ACTIVE, now in tuning. Measured selected
CV increases from baseline22.1 to51.7s/fold, and the evaluation cap86->201s.
Initial tuner fitness is finite0.8644510262049153, unlike baseline's inf.
Its211.575s wall bracket exceeds the rounded cap, so this is not a strict
per-evaluation wall-time compliance claim. The first proposal still times
out; outer quality, refit, total budget and saved-history provenance remain
pending. Original-CV versus cached-duplicate timings are detailed in the
budget hypothesis report. Runtime hashes verify; no jobs or code were changed.

## Bounded follow-up cycle closed: 2026-11-10 22:52 UTC

Fabert5 metadata pair is COMPLETE: both independently yield
**1.0498586672813428** with byte-identical valid predictions. Candidate
job3073.6s/training3039.3s versus baseline3224.2s/3198.1s; both successful
final refits and all budgets pass. This is not an outer-quality gain.

The COW fix is verified in saved history: original CatBoost154.99696s and
cached duplicate81.56952s remain separate from last RF71.11488s. The maximum
exact-match measurement correctly drives51.66565s/fold and201.496s evaluation
allowance. Initial tuner fitness becomes finite, but all five proposals and
the final check still time out; the tuner returns the initial graph. A finite
returned-history fitness does not imply a successful final re-evaluation.

A new no-training regression protects this original-versus-cached timing
contract through serialization:8 metadata tests and25 budget/refit tests
pass locally; the new test fails as expected on the old vendor. All three
hosts are idle. Exact hashes/provenance are in the budget hypothesis report.

The current bounded hypotheses are closed: metadata/transport correctness
fixes stay isolated and experimental; the valid400-tree screen was negative;
remaining-refit retains only its limited opt-in evidence. No general default,
search-space or original-PR promotion follows. Further calibration would be
a new prospectively designed study, not another repeat of these runs. The
existing monitoring task was deleted; no new experiments were launched.

## New authorized follow-up: 2026-11-22

See [the new prospective study](pr1467_followup_20261122.md) for execution of
the subsequently requested six-direction plan. New fold6 controls use fresh
isolated copies, COW-corrected metadata and verified AMLB environment transport.
H2/H4 train-only gates were negative and their full jobs were not launched.
H3's joint learning-rate/horizon policy passed its mechanism gate; the full
Covertype comparison and H5's Dionis refit confirmation are active. Fabert H1
continues and H6 waits for its host lock. These are new experiments, not
restarts of the audited November10 pairs; no full-run quality result is claimed
yet. Source archives, old results, deployed/shared runtimes and PR1467 remain
unchanged. The previous scheduler stays deleted.

At12:48UTC H1 is closed without a full run: both fidelities completed all five
proposals and the count gate was saturated. H2/H4 remain negative. H6's task
failure was reproduced and resolved, with independent probability/source
admission and21+2 full-stage contracts passed; its isolated full comparison
is now admitted. H3 control outer loss is independently.16127634533286989
(job1825.6/train1800.0s), while its candidate and H5 remain active. These are
not final paired effects. The new study records one continuation task for the
pending work; the old November10 task remains deleted. No general defaults
or PR1467 changes follow from these interim results.

H6 full baseline subsequently STARTED12:49:05 with verified isolated runtime;
its candidate waits. H5 control independently completed at loss.8555497437494369
within both budgets (job2996.3/train2922.6s), with successful native-logit final
refit and no fallback. H5 candidate STARTED12:46:25; H3 candidate remains active.
Continue these three existing queues, audit their candidate outputs and close
the study. H1/H2/H4 will not receive full jobs; no additional experiment is
admitted simply to occupy a free server.

Read-only check12:56UTC: no new completed pair. H3's candidate joint-search
activation is demonstrated by an actual horizon_profile0 trial and its
first LightGBM fold's100 fitted rounds; the other profiles and final quality
are still pending. H5 candidate and H6 baseline are composing; H6 candidate
waits without STARTED. All three runtime manifests and actual environments
verify. H1/H2/H4 stay closed; no new jobs or model/default changes were made.

H3 outer pair is now COMPLETE/audited13:15UTC: control logloss.16127634533286989
versus candidate.17973857943837115,11.4476% worse on Covertype6. Both budgets,
probabilities, paired truth/classes and final-refit checks pass. Both sampled
candidate profiles were0=(100,.1); the150/300/450 profiles were not exercised
and other selected parameters differed. Close this policy screen without
default promotion or another run; it does not refute the train-only horizon
mechanism. Exact artifacts/history/log provenance are in the new study.
H5 candidate remains composing; H6 baseline began tuning13:14:14.948 with
3fold152s cap/1365s search, and its candidate waits. H1/H2/H3/H4 are closed;
H5/H6 still require complete paired results and final audits.

At13:26UTC the remaining original H5/H6 queues are still active. H5 candidate
is evaluating native logit after a finite LGBM CV. H6 baseline initial tuner
fitness and three completed proposals remain invalid/inf, with explicit
timeout evidence; this is not an outer-test quality result or completed fit
failure. Candidate XGBoost-schema code has not started training yet. No new
predictions/history or COMPLETE/FAILED markers on these active runs. Runtime
and actual isolated environments verify; no jobs, sources or defaults changed.

H5 closed13:47UTC: Dionis6 expanded-refit/control losses both.8555497437494369,
byte-identical predictions, same native logit graph, valid budgets and
successful refits under even the original cap. No new quality gain or basis
for a default change. H6 baseline completed with finite loss1.2840150620620987
and valid budget/probabilities despite invalid inner tuner fitness. Its
correct EVO timing included prediction-cache hits, not a fresh cost probe.
H6 candidate STARTED13:42:38 in the original queue and remains active; it is
the sole remaining full run. Final pair must use the new v3 audit helper's
source-confirmed six-significant-digit score rule (13 tests pass). Full
provenance and hashes are in the new study. Five directions closed, H6 pending.

Read-only13:56UTC: H6 candidate remains in composition, original processes
live and runtime/source hashes verified. Its four completed initial CV
evaluations match control values, but final selection and outer predictions
are pending. No COMPLETE/FAILED/history yet. Five directions stay closed;
continue only this existing run and its final paired audit.

At14:12UTC H6 candidate has completed composition and entered CatBoost tuning.
Initial inner fitness is invalid after a timeout; trial0 is active. This is
not an outer-quality result or completed framework failure. Original processes
and isolated runtime verify; final refit, predictions and paired audit remain
pending. No new jobs, restarts or source changes.

At14:27UTC H6 candidate remains active in its original queue, with six
completed tuner proposals retaining bestinf and the seventh starting. Actual
isolated environment, seven runtime hashes, run manifest and v3 auditor verify.
No predictions/history/COMPLETE/FAILED yet, so no candidate outer-quality
conclusion is available. Five directions remain closed; retain the existing
continuation task until H6 final paired and history/refit audits. No new jobs,
restarts, source changes or repeated tests.

Final closure,2026-11-22 14:43UTC: H6 COMPLETE and independently paired-audited.
Fabert6 control/candidate loss1.2840150620620987, byte-identical824x7 predictions,
paired truth/classes/columns and valid probabilities. Candidate job3232.8s,
training3209.6s; both variants under3600s. Same exact native scaling/CatBoost
graph, final tuner/pipeline parameters verified. Full candidate refit succeeds
in72.627s after generation, no fallback. All10 tuner fitness records invalid,
6 explicit expired lines; finite outer loss is independently established.
The selected graph's118.1619s metadata matches117.877s logged CV, with explicit
prediction-cache hits on all3 folds. It is not a cold timing measurement.
All1476 frozen source files and7runtime entries verify; original queue ended
14:37:03, and matching workers are absent on all three hosts at14:42:29.

All six directions are now closed. No new general quality gain is established:
H1/H2/H4 did not pass original admission gates; H3 joint-policy outer loss is
11.4476% worse (both trials sampled native profile0); H5 and H6 preserve exactly
the control predictions. H6 remains a correctness fix for the reproduced
category failure, without outer-quality improvement on this selected CatBoost.
No general defaults or PR1467 changes. Train-only results, outer comparisons,
audit hashes and limitations are summarized in pr1467_followup_20261122.md;
the companion HTML keeps the two evaluation levels separate. Six full jobs
were used, with no repeats or extra admissions.
The final comparison was previewed/rendered and the sole continuation task
was deleted with confirmed deleted=true. No pending study work remains.

2026-11-23: new user-authorized adaptive campaign begins with H7, measuring
cached/fresh CV cost for the identical CatBoost graph. This is train-only
mechanism work, not an outer quality result or a repeat of H1's fidelity gate.
See pr1467_adaptive_20261123.md for prospective controls and admission evidence.
H7 started11:22:11UTC in a new isolated stage after13 passing host contracts.
Its cost hypothesis and paired probability checks are pending; no quality gain
or default change is claimed. One new task continues the authorized campaign.

H7 audited11:40UTC: cache-cost mechanism gate PASSES with zero probability
difference on all paired7413 training rows and unchanged graph. Cached CV~40s
versus fresh~113s means30% headroom is insufficient in this controlled probe.
This is not an outer-quality gain. Its direct float32 NPZ/InputData path has
different absolute inner metrics from historical AMLB; adapter fidelity still
needs checking. H8 now tests charged calibration/feasibility under equal600s
budgets with a fixed proposal; full quality confirmation remains a later gate.
H8 v2 STARTED11:53:59UTC after31 passing host tests. An earlier admission
attempt stopped before training when the overlap guard saw its own timeout
parent; that artifact is preserved. No active job was repeated. H8 results
are pending; the same continuation task carries the adaptive campaign forward.
At11:57:40 the original H8 control is confirming the native graph after invalid
initial/proposal CVs. Partial fold metrics are not full-CV or outer scores.
Candidate policy and paired feasibility/budget audit remain pending; all680
frozen manifest entries verify. No source changes or extra launches.

H8 COMPLETE12:03:49UTC and independently audited12:12. Charged calibration
restores one valid proposal and final CV; candidate349.51583s/control240.68375s,
both under600s. Confirmed CV1.7133149501 vs common native1.7143731903 is only
0.06173% lower on the SAME splits; it is not independent quality or outer gain.
Control invalid CV is not infinite outer loss. All680 hashes verify; audit SHA
79c457ea35045e72bda4931b59ef47e1897ad1ebbf7e758394910195bb443bfd.
Next H9 is a prospective no-fit actual-adapter preprocessing audit to resolve
data-contract uncertainty before any new model-quality/full-job admission.

H9 no-fit audit COMPLETE12:19:15UTC: actual API marks85 categorical columns
where direct InputData starts with800 numerical. Rows/labels unchanged; original
frame rounding<=2.9648e-8, API32 numerically identical. This is representation
evidence, not measured quality improvement. H10 prospectively tests two new
native API-path CVs, same model/splits, bounded1400s total, no cache or outer test.
All source/runtime changes remain isolated diagnostics; no defaults promoted.

H10 STARTED12:24:39UTC after7 passing host tests and prospective commit48260d21.
Original suite1327630, API32worker1327646 active12:24:50. Both source isolation
and frozen822-entry manifest verified. No complete native API-path CV evidence
yet; do not infer quality from H9's no-fit type counts. Continue existing pair.

H10 COMPLETE12:32:33UTC, independently audited12:40: both actualAPI paths give
0.9242401336464137, byte-identical7413x7 probabilities and exact historicalH6CV.
The46.09% reduction relative to H7direct is DIAGNOSTIC FIDELITY recovery, not a
gain over existing FEDOT. Float32/rawframe predictions match exactly; combined
API preprocessing contract matters. All822hashes/probabilities/splits/graphs/
budgets verified; auditSHA3d9fa3f9afe8c748161e12fa0449e18b3efbedd8b83ce5cce354a1b97f978f21.
H11 prospectively tests one fixed feature-subsampling proposal against native
on actualAPI data with a new predeclared CV split, not an untouched dataset.
No outer data or full-job admission; protocol and limitations in adaptive report.

H11 admitted after15 host tests, prospective commit4c94e496 pushed first.
STARTED12:45:15UTC, nativeworker1329413/suite1329396 alive12:45:41 with isolated
runtime and824-entry manifest. New-split control/candidate quality gate pending.
Same existing continuation task; no full AMLB launches or default changes.
