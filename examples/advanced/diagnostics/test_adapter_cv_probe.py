import json
import os
from pathlib import Path

import numpy as np
import pytest

from examples.advanced.diagnostics.adapter_cv_probe import paired_audit, LABELS, load_adapter
from examples.advanced.diagnostics.adapter_fidelity_probe import capture
from examples.advanced.diagnostics.cache_cost_probe import sha, worker


def fixture(stage, fault=None):
    (stage/'cells').mkdir()
    y=np.arange(7413)%7
    np.savez(stage/'train.npz',features=np.zeros((7413,2)),target=y)
    arrays={'classes':np.arange(7)}
    for i,rows in enumerate(np.array_split(np.arange(7413),3)):
        p=np.full((len(rows),7),.2/6);p[np.arange(len(rows)),y[rows]]=.8
        arrays.update({f'rows_{i}':rows,f'train_rows_{i}':np.setdiff1d(np.arange(7413),rows),
                       f'truth_{i}':y[rows],f'probs_{i}':p})
    np.savez(stage/'h7-reference.npz',**arrays)
    (stage/'h7-independent-audit.json').write_text(json.dumps(dict(
        cells={'fresh_a':{'graph_id_before':'native'}},independent_losses={'fresh_a':float(-np.log(.8))})))
    for label in LABELS:
        copied={k:v.copy() for k,v in arrays.items()}
        if label==LABELS[1]:
            if fault=='rows':copied['rows_0']=copied['rows_0'][::-1]
            if fault=='probabilities':copied['probs_0'][0]=0
            if fault=='classes':copied['classes']=copied['classes'][::-1]
        p=stage/'cells'/(label+'.npz');np.savez(p,**copied)
        record=dict(valid=True,folds=[{'fitted_nodes':[]}]*3,cache_enabled=False,
                    graph_id_before='native',graph_id_after='bad' if fault=='graph' else 'native',
                    npz_sha256=sha(p),data_sha256=sha(stage/'train.npz'),seconds=1,fitness=float(-np.log(.8)))
        p.with_suffix('.json').write_text(json.dumps(record))
        p.with_suffix('.process.json').write_text(json.dumps(dict(exit_code=0,seconds=700 if fault=='budget' else 2)))


def test_audit_accepts_paired_complete_cv(tmp_path):
    fixture(tmp_path)
    result=paired_audit(tmp_path)
    assert result['api_pair_max_difference']==0
    assert not result['representation_quality_gate']


@pytest.mark.parametrize('fault',['rows','probabilities','classes','graph','budget'])
def test_audit_rejects_corruption(tmp_path,fault):
    fixture(tmp_path,fault)
    with pytest.raises((AssertionError,ValueError)):
        paired_audit(tmp_path)


def test_host_captured_data_enters_real_cv(tmp_path):
    path=os.environ.get('FEDOT_H9_ADAPTER')
    if not path:pytest.skip('Host-only frozen adapter and native models')
    import importlib.util
    spec=importlib.util.spec_from_file_location('h10_test_adapter',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    x=np.random.RandomState(9).normal(size=(60,4)).astype('float32');y=np.arange(60)%3
    np.savez(tmp_path/'train.npz',features=x,target=y);(tmp_path/'cells').mkdir()
    data=capture(adapter,x,y)['data']
    worker(tmp_path,'fresh_api_smoke',smoke=True,spec={'cap':30},input_data=data)
    r=json.loads((tmp_path/'cells/fresh_api_smoke.json').read_text())
    assert r['valid'] and len(r['folds'])==3 and not r['cache_enabled']
    data.idx=data.idx[::-1]
    with pytest.raises(ValueError,match='preserve H7'):
        worker(tmp_path,'fresh_bad',smoke=True,spec={'cap':30},input_data=data)
