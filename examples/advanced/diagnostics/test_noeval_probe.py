import json
import os
from pathlib import Path

import numpy as np
import pytest

from examples.advanced.diagnostics.noeval_probe import audit, dataset, MODEL
from examples.advanced.diagnostics.cache_cost_probe import sha, probability_loss


def fixture(stage, fault=None):
    (stage/'cells').mkdir()
    for variant in ('baseline', 'candidate'):
        p = stage/variant/MODEL
        p.parent.mkdir(parents=True)
        p.write_text(variant)
    for kind in ('classification', 'regression'):
        _, y, train, test = dataset(kind)
        for use_eval in (True, False):
            for variant in ('baseline', 'candidate'):
                p = stage/'cells'/f'{kind}_{int(use_eval)}_{variant}.json'
                r = dict(module_sha256=sha(stage/variant/MODEL), callback_count=1,
                         thread_count=8, seconds=.1, model_params={'use_best_model':use_eval})
                if variant == 'baseline' and not use_eval:
                    r.update(status='expected_alias_error', error='only one od_wait early_stopping_rounds')
                else:
                    pred = np.full((60, 7), 1/7) if kind == 'classification' else y[test].copy()
                    classes = np.arange(7) if kind == 'classification' else np.array([])
                    score = probability_loss(pred, y[test], classes) if kind == 'classification' else 0.
                    rows = test[::-1] if fault == 'rows' else test
                    np.savez(p.with_suffix('.npz'),prediction=pred,truth=y[test],classes=classes,
                             train_rows=train,test_rows=rows)
                    r.update(status='fitted',trees=12,score=score,prediction_sha256=sha(p.with_suffix('.npz')),
                             pool_rows=[192,48] if use_eval else [240])
                    if fault == 'unused_detector' and not use_eval:r['model_params']['od_wait']=100
                    if fault == 'wrong_rows' and not use_eval:r['pool_rows']=[192]
                    if fault == 'callbacks':r['callback_count']=0
                p.write_text(json.dumps(r))


def test_complete_audit(tmp_path):
    fixture(tmp_path)
    assert audit(tmp_path)['gate_passed']


@pytest.mark.parametrize('fault',['rows','unused_detector','wrong_rows','callbacks'])
def test_audit_rejects_corruption(tmp_path,fault):
    fixture(tmp_path,fault)
    with pytest.raises(AssertionError):audit(tmp_path)


@pytest.mark.skipif(not os.environ.get('H13_STAGE'),reason='frozen host runtime only')
def test_frozen_constructors_and_native_split():
    import importlib.util
    from fedot.core.data.data import InputData
    from fedot.core.data.data_split import train_test_data_setup
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.operations.operation_parameters import OperationParameters
    from fedot.core.pipelines.node import PipelineNode
    stage=Path(os.environ['H13_STAGE'])
    for kind in ('classification','regression'):
        x,y,train,_=dataset(kind)
        data=InputData(idx=train,features=x[train],target=y[train],task=Task(getattr(TaskTypesEnum,kind)),data_type=DataTypesEnum.table)
        a,b=train_test_data_setup(data)
        assert (len(a.idx),len(b.idx))==(192,48)
        params={**PipelineNode('catboost' if kind=='classification' else 'catboostreg').parameters,
                'use_eval_set':False,'use_best_model':True,'fit_time_limit':30,'n_jobs':8}
        for variant in ('baseline','candidate'):
            spec=importlib.util.spec_from_file_location('test_'+variant,stage/variant/MODEL)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            impl=getattr(module,'FedotCatBoost'+kind.title()+'Implementation')(OperationParameters(**params))
            model=impl.model.get_params()
            assert len(impl.fit_callbacks)==1 and model['thread_count']==8
            if variant=='baseline':assert 'od_wait' in model and model['early_stopping_rounds'] is False
            else:assert not set(model) & {'early_stopping_rounds','od_wait','od_type','od_pval'}
