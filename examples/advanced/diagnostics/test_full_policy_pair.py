import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from examples.advanced.diagnostics.run_paid_policy_pair import command, validate_run


def test_pair_has_one_new_fold_fixed_budget_and_only_declared_switch():
    root=Path('/root');stage=Path('/stage')
    a=command(root,stage,'baseline');b=command(root,stage,'candidate')
    assert a[a.index('-f')+1]==b[b.index('-f')+1]=='7'
    assert '1h8c' in a and '--extra=seed=104735' in a
    assert '--extra=f._paid_allrow_tuning=false' in a
    assert '--extra=f._paid_allrow_tuning=true' in b
    normalize=lambda args:[v.replace('baseline','SIDE').replace('candidate','SIDE')
        for v in args if not v.startswith('--extra=f._paid_allrow_tuning=')]
    assert normalize(a)==normalize(b)
    assert '--extra=f.final_fit_use_remaining_time=false' in a


@pytest.mark.parametrize('field,value',[('duration','3601'),('training_duration','3601'),('logloss','nan'),('info','failed')])
def test_invalid_full_job_cannot_pass(tmp_path,field,value):
    p=tmp_path/'results/fedot.h21-baseline.fabert.fold7/scores/results.csv';p.parent.mkdir(parents=True)
    row=dict(duration='3000',training_duration='2900',logloss='1',result='-1',info='')
    row[field]=value
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    with pytest.raises(AssertionError):validate_run(tmp_path,'baseline')


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'),reason='frozen API/composer host')
def test_observer_preserves_real_tuner_and_predictions(tmp_path):
    from copy import deepcopy
    import numpy as np
    import pandas as pd
    from sklearn.metrics import log_loss
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.pipelines.node import PipelineNode
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    stage=Path(os.environ['FEDOT_H9_ADAPTER']).parents[3]
    adapter=load_adapter(stage);original=adapter.Fedot
    def factory(*args,**kwargs):
        model=original(*args,**kwargs)
        cat=PipelineNode('catboost',nodes_from=[PipelineNode('scaling')])
        cat.parameters={**cat.parameters,'num_trees':12,'max_depth':2}
        pipe=Pipeline(cat);pipe.replace_n_jobs_in_nodes(8)
        model.api_composer.propose_and_fit_initial_assumption=lambda data:([deepcopy(pipe)],deepcopy(pipe))
        model.api_composer.compose_pipeline=lambda data,*args:(deepcopy(pipe),[deepcopy(pipe)],SimpleNamespace(history=OptHistory()))
        return model
    x=np.random.RandomState(42).normal(size=(60,4));x[:,2:]=np.arange(120).reshape(60,2)%3
    y=np.arange(60)%3;frame=pd.DataFrame(x,columns=['a','b','c','d'])
    outputs=[];observation=tmp_path/'observation'
    for enabled in [False,True]:
        params={'_paid_allrow_tuning':True}
        if enabled:params['_paid_policy_observation_dir']=str(observation)
        config=SimpleNamespace(type='classification',seed=104735,framework_params=params)
        with patch.object(adapter,'Fedot',factory):
            model=adapter._make_fedot(config,'neg_log_loss',dict(n_jobs=8,cv_folds=3,
                with_tuning=True,use_operations_cache=False,use_preprocessing_cache=False,
                use_predictions_cache=False,final_fit_use_remaining_time=False),5.4,.54)
        model.fit(frame.copy(),y.copy())
        outputs.append(model.predict_proba(frame,probs_for_all_classes=True))
    assert np.array_equal(*outputs)
    report=json.loads((observation/'tuning-exit.json').read_text())
    assert len(report['evaluations'])==3
    for cell in sorted((observation/'cv').iterdir()):
        files=sorted(cell.glob('prediction-*.npz'));assert len(files)==3
        losses=[]
        for p in files:
            with np.load(p,allow_pickle=False) as z:
                assert len(z['train_rows'])==40 and len(z['rows'])==20
                losses.append(log_loss(z['truth'],z['probs'],labels=z['classes']))
        assert abs(np.mean(losses)-json.loads((cell/'evaluation.json').read_text())['fitness'])<1e-12
    assert (observation/'final-fit.json').exists() and list((observation/'final-pools').glob('*.json'))


@pytest.mark.skipif(not os.environ.get('FEDOT_H21_TRANSPORT'),reason='benchmark parent and real framework child')
@pytest.mark.parametrize('enabled',[False,True])
def test_real_parent_transport_carries_flags_and_observer(enabled,tmp_path,monkeypatch):
    import tempfile
    import numpy as np
    from amlb.data import DatasetType
    from amlb.utils import Namespace
    from frameworks.shared import caller
    import frameworks.FEDOT as framework
    stage=Path(os.environ['FEDOT_H21_TRANSPORT']);real=caller.run_in_venv
    def launch(caller_file,script_file,**kwargs):
        assert script_file=='exec.py'
        return real(caller_file,str(stage/'full_policy_observer.py'),
            python_exec=os.environ['FEDOT_DIAGNOSTIC_PYTHON'],**kwargs)
    class RetainedDirectory:
        def __init__(self,**kwargs):self.path=tempfile.mkdtemp(dir=tmp_path,**kwargs)
        def __enter__(self):return self.path
        def __exit__(self,*args):return False
    monkeypatch.setattr(caller,'run_in_venv',launch)
    monkeypatch.setattr(caller,'TemporaryDirectory',RetainedDirectory)
    monkeypatch.setattr(caller,'rconfig',lambda:Namespace(root_dir=str(stage/'harness'),monitoring=Namespace(statistics=[])))
    split=SimpleNamespace(X=np.empty((0,2)),y_enc=np.empty((0,1)))
    dataset=SimpleNamespace(type=DatasetType.multiclass,target=None,train=split,test=split,release=lambda:None)
    config=Namespace(name='h21-preflight',type='classification',seed=104735,
        observer_stage=str(stage),job_timeout_seconds=90,framework_params={
            '_paid_allrow_tuning':enabled,'_paid_policy_observation_dir':str(tmp_path/'observer')})
    result=framework.run_fedot_tabular(dataset,config)
    assert result['models_count']==0 and result['probe'].flag==enabled
    assert result['probe'].observed and result['probe'].fits_recorded==0
    assert Path(result['probe'].fedot_file).is_relative_to(stage/'source')


@pytest.mark.skipif(not os.environ.get('FEDOT_H21_EXISTING_OBSERVATION'),reason='read-only audit of retained tiny preflight')
def test_independent_auditor_and_budget_corruption(tmp_path):
    import shutil
    from examples.advanced.diagnostics.audit_paid_policy_pair import audit_observation
    original=Path(os.environ['FEDOT_H21_EXISTING_OBSERVATION'])
    result=audit_observation(original)
    assert result['proper_selection'] and len(result['evaluations'])==3
    copied=tmp_path/'copy';shutil.copytree(original,copied)
    p=copied/'tuning-exit.json';value=json.loads(p.read_text())
    value['policy']['bounds']['allocation']+=1
    p.write_text(json.dumps(value))
    with pytest.raises(AssertionError):audit_observation(copied)
