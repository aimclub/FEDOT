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
stage="$diagnostics/multiclass-quality-20261110"
benchmark_python="$root/venvs/automlbenchmark/bin/python"
framework_python="$root/automlbenchmark/frameworks/FEDOT/venv/bin/python"
test -f "$diagnostics/vendor/golem/core/tuning/tuner_interface.py"
# Fail before creating run markers if a task belongs to another suite.
grep -Fqx -- "- name: $task" "$source_stage/automlbenchmark/resources/benchmarks/$suite.yaml" || {
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
for variant in baseline candidate; do
    code_root="$stage/$variant"
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
    export PYTHONPATH="$diagnostics/vendor:$code_root:$source_stage/automlbenchmark"
    "$framework_python" -c 'import sys, fedot, golem, lightgbm, sklearn; print(sys.version); print(fedot.__file__); print(golem.__file__); print(lightgbm.__version__, sklearn.__version__)' >"$run/environment.txt"
    session="fedot.multiclass-quality-$variant.$task.fold$fold"
    set +e
    timeout --signal=TERM --kill-after=60s 7500s \
        "$benchmark_python" "$source_stage/automlbenchmark/runbenchmark.py" \
        FEDOT "$suite" 1h8c -m local -p 1 -s skip -i "$root/input" \
        -o "$run/results" -t "$task" -f "$fold" --session "$session" \
        --logging=console:info,app:debug,root:info \
        --extra="seed=$((104729 + 10#$fold))" --extra=f._portfolio=false \
        --extra=f.parallelization_mode=sequential --extra=f.n_jobs=8 \
        --extra=f.preset=best_quality --extra=f.with_tuning=true \
        --extra=f.logging_level=10 >"$run/run.log" 2>&1
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
