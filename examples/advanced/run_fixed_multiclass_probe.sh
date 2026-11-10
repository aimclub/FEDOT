#!/usr/bin/env bash
# Bounded, train-only fixed graph ablation. Never overwrite existing artifacts.
set -euo pipefail
root=${1:?benchmark root}
series=${2:?series}
mode=${3:-factorial}
fold=${4:-3}
[[ "$fold" =~ ^[0-9]$ ]]
[[ "$mode" == factorial || "$mode" == components || "$mode" == lgbm || "$mode" == horizon || "$mode" == confirm ]]
extra_args=(--save-predictions)
if [[ "$mode" == components ]]; then
    extra_args+=(--component-ablation)
elif [[ "$mode" == lgbm ]]; then
    extra_args+=(--lgbm-ablation)
elif [[ "$mode" == horizon ]]; then
    extra_args+=(--learning-rate-horizon)
elif [[ "$mode" == confirm ]]; then
    extra_args+=(--horizon-confirmation)
fi
[[ "$root" == "/var/essdata/$(id -un)/fedot-amlb-official-1h8c" ]]
[[ "$series" =~ ^[A-Za-z0-9_-]+$ ]]
diagnostics="$root/experiments/pr1467-diagnostics-20261109"
stage="$diagnostics/$series"
old="$diagnostics/multiclass-quality-20261110"
code="$stage/candidate"
benchmark_python="$root/venvs/automlbenchmark/bin/python"
framework_python="$root/automlbenchmark/frameworks/FEDOT/venv/bin/python"
exec 8>"$stage/queue.lock"
flock -n 8 || exit 7
run="$stage/runs/fixed-graph/Covertype-fold$fold"
mkdir -p "$run"
[[ ! -e "$run/STARTED" ]] || { echo 'Refusing to repeat probe'; exit 8; }
date -u +%FT%TZ >"$run/STARTED"
cp "$stage/candidate.commit" "$run/source.commit"
trap 'code=$?; if (( code != 0 )); then printf "exit=%s\n" "$code" >"$run/FAILED"; fi' EXIT
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
export XDG_CACHE_HOME="${root%/fedot-amlb-official-1h8c}/.cache/official-1h8c"
export XDG_CONFIG_HOME="$XDG_CACHE_HOME/openml-config"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export JOBLIB_TEMP_FOLDER="$root/tmp"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$diagnostics/vendor:$code"
cd "$run"
timeout --signal=TERM --kill-after=60s 1200s "$benchmark_python" \
    "$code/examples/advanced/multiclass_train_probe.py" prepare \
    --cache "$root/input" --task-id 7593 --fold "$fold" --output train.npz \
    --allow-numeric-binary-categories >prepare.log 2>&1
"$framework_python" -c 'import sys, fedot, golem, lightgbm, sklearn; print(sys.version); print(fedot.__file__); print(golem.__file__); print(lightgbm.__version__, sklearn.__version__)' >environment.txt
timeout --signal=TERM --kill-after=60s 7200s "$framework_python" \
    "$code/examples/advanced/fixed_multiclass_graph_probe.py" \
    --data train.npz --baseline-log "$old/runs/baseline/Covertype-fold3-large/run.log" \
    --candidate-log "$old/runs/candidate/Covertype-fold3-large/run.log" \
    --output fixed-graph.json "${extra_args[@]}" >run.log 2>&1
"$framework_python" - "$fold" "$mode" >validation.log <<'PY'
import json
import sys
with open('fixed-graph.json') as stream:
    result = json.load(stream)
assert result['task_id'] == 7593 and result['fold'] == int(sys.argv[1])
assert result['split'] == 'outer_train_inner_holdout'
expected = {'factorial': 8, 'components': 6, 'lgbm': 9, 'horizon': 8, 'confirm': 4}
assert len(result['results']) == expected[sys.argv[2]]
print('Verified task, fold, split and completed cell count')
PY
sha256sum fixed-graph.json >COMPLETE
date -u +%FT%TZ >"$stage/QUEUE_FINISHED"
