"""H16: actual AMLB parent/callee serialization and API-boundary fidelity, no fit."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from examples.advanced.diagnostics.cache_cost_probe import sha, write_json


def child_probe(dataset, config):
    import pandas as pd
    import fedot
    from fedot.core.pipelines.pipeline import Pipeline
    from frameworks.shared.callee import result
    from examples.advanced.diagnostics.adapter_fidelity_probe import summarize, CapturedBeforeSearch
    stage=Path(config.probe_stage)
    assert Path(fedot.__file__).resolve().is_relative_to((stage/'source').resolve())
    assert os.environ.get('FEDOT_OPERATION_DEFAULTS_OVERLAY') is None
    assert isinstance(dataset.train.X,pd.DataFrame)
    assert len(dataset.test.X)==len(dataset.test.y)==0
    with (stage/'received-frame.pkl').open('xb') as f:dataset.train.X.to_pickle(f)
    path=stage/'harness/frameworks/FEDOT/exec.py'
    spec=importlib.util.spec_from_file_location('h16_actual_adapter',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    original=adapter.Fedot;captured={}
    def factory(*args,**kwargs):
        automl=original(*args,**kwargs)
        def stop(data):
            captured['data']=data
            captured['params']={k:automl.params.get(k) for k in ['use_input_preprocessing','use_auto_preprocessing','cv_folds']}
            raise CapturedBeforeSearch()
        automl.api_composer.obtain_model=stop
        return automl
    with patch.object(adapter,'Fedot',factory),patch.object(Pipeline,'fit',side_effect=AssertionError('H16 forbids model fits')):
        try:adapter.run(dataset,config)
        except CapturedBeforeSearch:pass
    assert 'data' in captured
    data=captured['data']
    with (stage/'received-and-api.npz').open('xb') as f:
        np.savez_compressed(f,received_target=np.asarray(dataset.train.y),
                            features=np.asarray(data.features,dtype=float),target=np.asarray(data.target),idx=np.asarray(data.idx))
    record=dict(schema=summarize(data),params=captured['params'],model_fits=0,
                encoded_class_count=dataset.encoded_class_count,
                received_frame_sha256=sha(stage/'received-frame.pkl'),arrays_sha256=sha(stage/'received-and-api.npz'),
                fedot_file=fedot.__file__,adapter_sha256=sha(path),
                environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']})
    write_json(stage/'child.json',record)
    return result(models_count=0,training_duration=0,predict_duration=0,probe_sha256=sha(stage/'child.json'))


def transport(stage, framework_python):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from amlb.data import DatasetType
    from amlb.utils import Namespace
    from frameworks.shared import caller
    import frameworks.FEDOT as framework
    assert Path(framework.__file__).resolve().is_relative_to((stage/'harness').resolve())
    assert Path(caller.__file__).resolve().is_relative_to((stage/'harness').resolve())
    frame=pd.read_pickle(stage/'train-frame.pkl')
    with np.load(stage/'train.npz',allow_pickle=False) as z:y=z['target'].ravel()
    encoder=LabelEncoder().fit(y)
    # Actual parent executes target-only encoding. Proxy has only TRAIN data;
    # its empty test placeholder cannot access outer labels/features.
    target=SimpleNamespace(label_encoder=SimpleNamespace(classes=encoder.classes_,transform=encoder.transform))
    dataset=SimpleNamespace(type=DatasetType.multiclass,target=target,
        train=SimpleNamespace(X=frame,y=pd.DataFrame({'target':y})),
        test=SimpleNamespace(X=frame.iloc[:0].copy(),y=pd.DataFrame({'target':y[:0]})),release=lambda:None)
    config=Namespace(name='h16-train-only',type='classification',metric='logloss',cores=8,seed=104735,
                     max_runtime_seconds=3600,job_timeout_seconds=180,probe_stage=str(stage),
                     framework_params=dict(preset='best_quality',with_tuning=True,final_fit_use_remaining_time=False,_portfolio=False))
    real=caller.run_in_venv
    def launch(caller_file,script_file,**kwargs):
        assert script_file=='exec.py'
        return real(caller_file,str(Path(__file__).resolve()),'child',python_exec=framework_python,**kwargs)
    class RetainedDirectory:
        def __init__(self,**kwargs):self.path=tempfile.mkdtemp(dir=stage/'transport',**kwargs)
        def __enter__(self):return self.path
        def __exit__(self,*args):return False
    (stage/'transport').mkdir()
    before=time.monotonic()
    with patch.object(caller,'rconfig',lambda:Namespace(root_dir=str(stage/'harness'),monitoring=Namespace(statistics=[]))),\
         patch.object(caller,'TemporaryDirectory',RetainedDirectory),patch.object(caller,'run_in_venv',launch):
        response=framework.run_fedot_tabular(dataset,config)
    assert response['models_count']==0 and response['training_duration']==0
    write_json(stage/'parent.json',dict(seconds=time.monotonic()-before,response=response,
               caller_sha256=sha(stage/'harness/frameworks/shared/caller.py'),
               parent_sha256=sha(stage/'harness/frameworks/FEDOT/__init__.py'),
               target_encoder='sklearn LabelEncoder proxy on existing encoded TRAIN classes; no OpenML task reload'))


def audit(stage):
    import pandas as pd
    original=pd.read_pickle(stage/'train-frame.pkl');received=pd.read_pickle(stage/'received-frame.pkl')
    pd.testing.assert_frame_equal(original,received,check_exact=True)
    child=json.loads((stage/'child.json').read_text());parent=json.loads((stage/'parent.json').read_text())
    reference=json.loads((stage/'h9-result.json').read_text())
    assert child['model_fits']==0 and parent['response']['models_count']==0 and parent['response']['training_duration']==0
    assert parent['response']['probe_sha256']==sha(stage/'child.json')
    assert child['received_frame_sha256']==sha(stage/'received-frame.pkl') and child['arrays_sha256']==sha(stage/'received-and-api.npz')
    assert child['schema']==reference['api_frame'] and child['params']==reference['params']
    assert child['environment']['FEDOT_OPERATION_DEFAULTS_OVERLAY'] is None
    assert parent['seconds']<=240
    with np.load(stage/'train.npz',allow_pickle=False) as raw,np.load(stage/'received-and-api.npz',allow_pickle=False) as z,\
         np.load(stage/'h9-representations.npz',allow_pickle=False) as ref:
        y=raw['target'].ravel()
        assert z['received_target'].shape==(len(y),1)
        assert np.array_equal(z['received_target'].ravel(),y) and np.array_equal(z['target'].ravel(),y)
        assert child['encoded_class_count']==len(np.unique(y))
        assert np.array_equal(z['features'],ref['api_frame_features'],equal_nan=True)
        assert np.array_equal(z['idx'],ref['api_frame_idx']) and np.array_equal(z['target'].ravel(),ref['api_frame_target'])
    return dict(gate_passed=True,rows=len(original),columns=original.shape[1],model_fits=0,
                categorical_count=len(child['schema']['categorical_idx']),parent_seconds=parent['seconds'],
                received_array_sha256=sha(stage/'received-and-api.npz'),
                scope='Real parent/caller/callee and API boundary; TRAIN-only proxy dataset/encoded labels, no full task or quality result')


def run(stage, framework_python):
    import fcntl
    from examples.advanced.diagnostics.fresh_budget_probe import training_conflict
    with (stage.parents[2]/'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own=str(Path(__file__).resolve())
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:a=[v.decode(errors='replace') for v in p.read_bytes().split(b'\0') if v]
            except OSError:continue
            pid=int(p.parent.name);parent=pid==os.getppid() and a and Path(a[0]).name=='timeout' and own in a
            extra=any(Path(v).name in ['api_budget_probe.py','allrow_fit_probe.py','transport_fidelity_probe.py'] for v in a)
            if pid!=os.getpid() and not parent and (extra or training_conflict(pid,a,os.getpid(),os.getppid(),own)):raise RuntimeError('Host occupied: '+str(pid))
        m=json.loads((stage/'manifest.json').read_text());assert m['preflight_passed']
        for n,h in m['files'].items():assert sha(stage/n)==h,n
        write_json(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
        before=time.monotonic()
        try:
            transport(stage,framework_python)
            result=audit(stage);result['stage_seconds']=time.monotonic()-before;assert result['stage_seconds']<=300
            write_json(stage/'independent-audit.json',result)
            write_json(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        except BaseException as e:write_json(stage/'FAILED',dict(error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','child','audit']);p.add_argument('--stage',type=Path);p.add_argument('--framework-python')
    a=p.parse_args()
    if a.mode=='child':
        from frameworks.shared.callee import call_run
        call_run(child_probe)
    elif a.mode=='run':run(a.stage,a.framework_python)
    else:print(json.dumps(audit(a.stage)))
