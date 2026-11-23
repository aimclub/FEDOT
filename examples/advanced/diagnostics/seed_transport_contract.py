"""H27: actual AMLB caller/callee carries explicit tuner seeds; zero model fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from examples.advanced.diagnostics.cache_startup_contract import save, verify
from examples.advanced.diagnostics.composer_seed_contract import CELLS, run_case


def flags(name):
    assert name in CELLS
    value = {'_paid_allrow_tuning': True}
    if name == 'original-seeded':
        value['_tuner_random_state'] = '104758'
    elif name == 'paid-seeded':
        value['_tuner_random_state'] = 104758
    elif name == 'paid-none':
        value['_tuner_random_state'] = None
    return value


def check_flags(name, received):
    expected = flags(name)
    assert set(received) == set(expected)
    for key, value in expected.items():
        assert type(received[key]) is type(value) and received[key] == value, key


def child(dataset, config):
    from examples.advanced.diagnostics import adapter_cv_probe
    from frameworks.shared.callee import result
    stage = Path(config.contract_stage)
    run = Path(config.contract_output)
    name = config.contract_name
    received = dict(config.framework_params)
    check_flags(name, received)
    assert config.seed == 104735 and config.type == 'classification'
    assert np.asarray(dataset.train.X).shape == (0, 2)
    assert np.asarray(dataset.test.X).shape == (0, 2)
    adapter = adapter_cv_probe.load_adapter(stage)
    original = adapter._make_fedot
    factories = []

    def transported_factory(local_config, *args, **kwargs):
        # H26's diagnostic generates an expected config. Feed the ACTUAL received
        # config to the unchanged factory instead, after checking typed parity.
        check_flags(name, local_config.framework_params)
        model = original(config, *args, **kwargs)
        factories.append(dict(received=received, seed=config.seed,
            composer_seed=getattr(model.api_composer, '_experimental_tuner_random_state', None),
            composer_has_seed=hasattr(model.api_composer, '_experimental_tuner_random_state')))
        return model

    if config.contract_smoke:
        graph = [dict(operation='logit', params={'C': 1.}, parents=[1]),
                 dict(operation='scaling', params={}, parents=[])]
    else:
        graph = json.loads((stage/'h23-entry.json').read_text())['graph']
    before = time.monotonic()
    with patch.object(adapter, '_make_fedot', transported_factory), \
            patch.object(adapter_cv_probe, 'load_adapter', lambda requested: adapter):
        record = run_case(stage, run, name, graph, smoke=config.contract_smoke)
    assert len(factories) == 1 and record['fits'] == 0
    child_record = dict(name=name, factories=factories, seconds=time.monotonic()-before,
        pid=os.getpid(), ppid=os.getppid(), executable=os.path.realpath(os.sys.executable),
        environment={key:os.environ.get(key) for key in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']})
    save(run/'child.json', child_record)
    return result(models_count=0, training_duration=0, predict_duration=0,
        transport_proof=hashlib.sha256((run/'child.json').read_bytes()).hexdigest())


def parent(stage, run, name, framework_python, smoke=False):
    from amlb.data import DatasetType
    from amlb.utils import Namespace
    from frameworks.shared import caller
    import frameworks.FEDOT as framework
    # No real dataset or labels enter the transport or the analytic objective.
    split = SimpleNamespace(X=np.empty((0, 2)), y_enc=np.empty((0, 1)))
    dataset = SimpleNamespace(type=DatasetType.multiclass, target=None,
        train=split, test=split, release=lambda: None)
    config = Namespace(name='h27-no-fit', type='classification', seed=104735,
        job_timeout_seconds=60, contract_stage=str(stage), contract_output=str(run),
        contract_name=name, contract_smoke=smoke, framework_params=flags(name))
    real = caller.run_in_venv

    def launch(caller_file, script_file, **kwargs):
        assert script_file == 'exec.py'
        return real(caller_file, str(stage/'seed_transport_contract.py'), 'child',
            python_exec=framework_python, **kwargs)

    class RetainedDirectory:
        def __init__(self, **kwargs):
            self.path = tempfile.mkdtemp(dir=run/'transport', **kwargs)
        def __enter__(self): return self.path
        def __exit__(self, *args): return False

    (run/'transport').mkdir()
    before = time.monotonic()
    with patch.object(caller, 'run_in_venv', launch), \
            patch.object(caller, 'TemporaryDirectory', RetainedDirectory), \
            patch.object(caller, 'rconfig', lambda: Namespace(root_dir=str(stage/'harness'),
                monitoring=Namespace(statistics=[]))):
        response = framework.run_fedot_tabular(dataset, config)
    assert response['models_count'] == 0
    proof = hashlib.sha256((run/'child.json').read_bytes()).hexdigest()
    assert response['transport_proof'] == proof
    save(run/'parent.json', dict(seconds=time.monotonic()-before, name=name, sent=flags(name),
        child_proof=proof, models_count=response['models_count'], pid=os.getpid()))


def audit(stage):
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from examples.advanced.diagnostics.fallback_rng_contract import canonical_history, analytic_score
    count = verify(stage)
    results = {}
    for name in CELLS:
        run = stage/'cells'/name
        read = lambda file: json.loads((run/file).read_text())
        record, c, p, process = [read(f) for f in ['result.json','child.json','parent.json','process.json']]
        reference = json.loads((stage/f'h26-{name}.json').read_text())
        check_flags(name, p['sent'])
        assert len(c['factories']) == 1
        factory = c['factories'][0]
        check_flags(name, factory['received'])
        seeded = name.endswith('-seeded')
        assert factory['seed'] == 104735
        assert factory['composer_seed'] == (104758 if seeded else None)
        assert factory['composer_has_seed'] == seeded
        assert c['pid'] != p['pid'] and c['ppid'] != c['pid']
        assert p['child_proof'] == hashlib.sha256((run/'child.json').read_bytes()).hexdigest()
        assert p['models_count'] == 0 and record['fits'] == 0
        assert process['exit_code'] == 0 and not process['timed_out'] and process['seconds'] <= 90
        assert p['seconds'] <= 80 and c['seconds'] <= 60
        expected_path = f'{stage}/harness:{stage}/vendor:{stage}/source:{stage}/harness'
        assert c['environment'] == record['environment'] == dict(PYTHONPATH=expected_path, FEDOT_OPERATION_DEFAULTS_OVERLAY=None)
        for key in ['initial','calls','fmin','builders','unseeded_requests','returned_graph','history','fits','seed','paid']:
            assert record[key] == reference[key], (name, key)
        assert record['history'] == canonical_history(OptHistory.load(str(run/'history.json')))
        assert record['unseeded_requests'] == (0 if seeded else 1 if name.startswith('paid-') else 2)
        for call in record['calls']:
            value = (1. if call['graph'][0]['params']['use_eval_set'] else .9) if record['paid'] else analytic_score(call['graph'])
            assert call['fitness'] == value
        policy = record['policy']
        if record['paid']:
            assert policy['status'] == 'evaluated' and policy['evaluations_count'] == 1
            assert policy['admission']['admitted'] and policy['admission']['remaining'] >= 110
            assert [v['cap'] for v in policy['calls']] == [300, 5, 5]
            assert policy['bounds']['refit_reserve'] == 324 and policy['bounds']['postfit_reserve'] == 162
            assert policy['charged_tuning_seconds'] <= policy['bounds']['allocation'] <= 1100
        else:
            assert policy['status'] == 'ineligible_original_path'
        results[name] = dict(process_seconds=process['seconds'], child_seconds=c['seconds'],
            result_sha=hashlib.sha256((run/'result.json').read_bytes()).hexdigest(), matches_h26=True)
    return dict(passed=True, verified_files=count, model_fits=0, cells=results,
        scope='Actual parent/caller/callee typed seed transport and analytic tuner parity; no model training or quality claim')


def suite(stage, framework_python, benchmark_python):
    from examples.advanced.diagnostics.run_followup import check_idle
    check_idle(); verify(stage)
    save(stage/'STARTED', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), pid=os.getpid()))
    (stage/'cells').mkdir()
    before = time.monotonic()
    for name in CELLS:
        run = stage/'cells'/name; run.mkdir()
        env = os.environ.copy(); env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
        env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',
            AMLB_PATH=str(stage/'harness'), PYTHONDONTWRITEBYTECODE='1')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[key]='8'
        start = time.monotonic()
        with (run/'worker.log').open('x') as log:
            worker = subprocess.Popen([benchmark_python,str(stage/'seed_transport_contract.py'),'parent',
                '--stage',str(stage),'--output',str(run),'--name',name,'--framework-python',framework_python],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out = False
            try: code = worker.wait(timeout=min(90,390-(time.monotonic()-before)))
            except subprocess.TimeoutExpired:
                timed_out=True; os.killpg(worker.pid,signal.SIGTERM)
                try: worker.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(worker.pid,signal.SIGKILL); worker.wait()
                code=worker.returncode
        save(run/'process.json',dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-start))
        assert code==0 and not timed_out
    report=audit(stage); report['stage_seconds']=time.monotonic()-before
    assert report['stage_seconds']<=390
    save(stage/'independent-audit.json',report)
    save(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['child','parent','suite','audit'])
    parser.add_argument('--stage',type=Path); parser.add_argument('--output',type=Path)
    parser.add_argument('--name',choices=CELLS); parser.add_argument('--framework-python')
    parser.add_argument('--benchmark-python'); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.mode=='child':
        from frameworks.shared.callee import call_run
        call_run(child)
    elif args.mode=='parent': parent(args.stage,args.output,args.name,args.framework_python,args.smoke)
    elif args.mode=='audit': print(json.dumps(audit(args.stage),indent=2))
    else:
        try: suite(args.stage,args.framework_python,args.benchmark_python)
        except BaseException as error: save(args.stage/'FAILED',dict(error=repr(error))); raise
