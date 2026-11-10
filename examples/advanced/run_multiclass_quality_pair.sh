#!/usr/bin/env bash
# Isolated 1h8c baseline/candidate comparison. Never overwrite or retry a run.
set -euo pipefail
root=${1:?benchmark root}
suite=${2:?suite}
task=${3:?task}
fold=${4:?fold}
[[ "$root" == "/var/essdata/$(id -un)/fedot-amlb-official-1h8c" ]]
[[ "$suite" =~ ^(small|medium|large)$ && "$task" =~ ^[A-Za-z0-9_.-]+$ && "$fold" =~ ^[0-9]$ ]]
source_stage="$root/experiments/pr1467-5fbcaf16"
diagnostics="$root/experiments/pr1467-diagnostics-20261109"
series=${5:-multiclass-quality-20261110}
[[ "$series" =~ ^[A-Za-z0-9_-]+$ ]]
candidate_remaining_refit=${6:-false}
[[ "$candidate_remaining_refit" =~ ^(true|false)$ ]]
variants=${7:-'baseline candidate'}
[[ "$variants" == 'baseline candidate' || "$variants" == candidate ]]
candidate_profile=${8:-none}
[[ "$candidate_profile" == none || "$candidate_profile" == lgbm400 ]]
stage="$diagnostics/$series"
runtime_mode=${9:-shared}
[[ "$runtime_mode" == shared || "$runtime_mode" == isolated ]]
harness_root="$source_stage/automlbenchmark"
if [[ "$runtime_mode" == isolated ]]; then
    harness_root="$stage/harness"
    # The manifest is prepared before launch and covers both immutable vendors
    # and the isolated harness. A STARTED run never repairs runtime sources.
    test -s "$stage/runtime.sha256"
    for artifact in \
        harness/frameworks/FEDOT/__init__.py harness/frameworks/FEDOT/exec.py \
        harness/frameworks/shared/caller.py \
        vendor/baseline/golem/core/optimisers/opt_history_objects/individual.py \
        vendor/candidate/golem/core/optimisers/opt_history_objects/individual.py \
        vendor/baseline/golem/core/tuning/tuner_interface.py \
        vendor/candidate/golem/core/tuning/tuner_interface.py; do
        grep -Fq "  $artifact" "$stage/runtime.sha256"
    done
    (cd "$stage" && sha256sum --check runtime.sha256)
elif [[ "$candidate_profile" != none ]]; then
    echo 'Defaults overlays require a validated isolated AMLB transport' >&2
    exit 2
fi
benchmark_python="$root/venvs/automlbenchmark/bin/python"
framework_python="$root/automlbenchmark/frameworks/FEDOT/venv/bin/python"
# Fail before creating run markers if a task belongs to another suite.
grep -Fqx -- "- name: $task" "$harness_root/resources/benchmarks/$suite.yaml" || {
    echo "Task $task does not belong to suite $suite" >&2
    exit 2
}
exec 8>"$stage/queue.lock"
flock -n 8 || { echo 'Queue already active'; exit 7; }
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
export XDG_CACHE_HOME="${root%/fedot-amlb-official-1h8c}/.cache/official-1h8c"
export XDG_CONFIG_HOME="$XDG_CACHE_HOME/openml-config"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip" UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export JOBLIB_TEMP_FOLDER="$root/tmp"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$stage"
for variant in $variants; do
    code_root="$stage/$variant"
    vendor_root="$diagnostics/vendor"
    if [[ "$runtime_mode" == isolated ]]; then
        vendor_root="$stage/vendor/$variant"
        (cd "$stage" && sha256sum --check runtime.sha256)
    fi
    test -f "$vendor_root/golem/core/tuning/tuner_interface.py"
    test -f "$code_root/fedot/api/main.py"
    test -s "$stage/$variant.commit"
    run="$stage/runs/$variant/$task-fold$fold-$suite"
    mkdir -p "$run"
    if [[ -e "$run/STARTED" ]]; then
        echo "Refusing to repeat existing run: $run"
        continue
    fi
    date -u +%FT%TZ >"$run/STARTED"
    cp "$stage/$variant.commit" "$run/source.commit"
    export PYTHONPATH="$vendor_root:$code_root:$harness_root"
    unset FEDOT_OPERATION_DEFAULTS_OVERLAY
    if [[ "$variant" == candidate && "$candidate_profile" == lgbm400 ]]; then
        export FEDOT_OPERATION_DEFAULTS_OVERLAY="$code_root/examples/advanced/lgbm_400_defaults.json"
        test -s "$FEDOT_OPERATION_DEFAULTS_OVERLAY"
    fi
    "$framework_python" -c 'import sys, fedot, golem, lightgbm, sklearn; print(sys.version); print(fedot.__file__); print(golem.__file__); print(lightgbm.__version__, sklearn.__version__)' >"$run/environment.txt"
    if [[ "$runtime_mode" == isolated ]]; then
        cp "$stage/runtime.sha256" "$run/runtime.sha256"
        sha256sum "$vendor_root/golem/core/optimisers/opt_history_objects/individual.py" \
            "$harness_root/frameworks/FEDOT/__init__.py" \
            "$harness_root/frameworks/FEDOT/exec.py" \
            "$harness_root/frameworks/shared/caller.py" >>"$run/environment.txt"
    fi
    "$framework_python" -c 'import os, hashlib; from pathlib import Path; from fedot.core.operations.operation_parameters import get_default_params; p=os.environ.get("FEDOT_OPERATION_DEFAULTS_OVERLAY"); print("operation_defaults_overlay:", p); print("overlay_sha256:", hashlib.sha256(Path(p).read_bytes()).hexdigest() if p else None); print("lgbm_effective_defaults:", get_default_params("lgbm"))' >>"$run/environment.txt"
    session="fedot.multiclass-quality-$variant.$task.fold$fold"
    extra_args=()
    if [[ "$variant" == candidate && "$candidate_remaining_refit" == true ]]; then
        extra_args+=(--extra=f.final_fit_use_remaining_time=true)
    fi
    set +e
    timeout --signal=TERM --kill-after=60s 7500s \
        "$benchmark_python" "$harness_root/runbenchmark.py" \
        FEDOT "$suite" 1h8c -m local -p 1 -s skip -i "$root/input" \
        -o "$run/results" -t "$task" -f "$fold" --session "$session" \
        --logging=console:info,app:debug,root:info \
        --extra="seed=$((104729 + 10#$fold))" --extra=f._portfolio=false \
        --extra=f.parallelization_mode=sequential --extra=f.n_jobs=8 \
        --extra=f.preset=best_quality --extra=f.with_tuning=true \
        --extra=f._save_artifacts=info \
        --extra=f.logging_level=10 "${extra_args[@]}" >"$run/run.log" 2>&1
    status=$?
    set -e
    score="$run/results/$session/scores/results.csv"
    if (( status == 0 )) && "$benchmark_python" - "$score" >"$run/validation.log" 2>&1 <<'PY'
import csv
import math
import sys
with open(sys.argv[1], newline='') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 1, rows
row = rows[0]
assert not row.get('info'), row
assert math.isfinite(float(row['result'])), row
assert math.isfinite(float(row['logloss'])), row
assert 0 <= float(row['training_duration']) <= 3600, row
assert 0 <= float(row['duration']) <= 3600, row
print(row)
PY
    then
        sha256sum "$score" >"$run/COMPLETE"
        echo "COMPLETE $variant $task $fold"
    else
        printf 'runner_exit=%s; inspect run.log and validation.log\n' "$status" >"$run/FAILED"
        echo "FAILED $variant $task $fold"
    fi
done
date -u +%FT%TZ >"$stage/QUEUE_FINISHED"
