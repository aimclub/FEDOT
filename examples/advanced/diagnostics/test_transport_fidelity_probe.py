import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.advanced.diagnostics.transport_fidelity_probe import audit, transport, sha


def fixture(stage,fault=None):
    frame=pd.DataFrame({'a':np.arange(6,dtype=np.float64),'b':np.arange(6,dtype=np.uint8)})
    frame.to_pickle(stage/'train-frame.pkl')
    if fault=='dtype':frame['b']=frame['b'].astype('int64')
    if fault=='columns':frame.columns=['changed','b']
    frame.to_pickle(stage/'received-frame.pkl')
    y=np.arange(6)%3;np.savez(stage/'train.npz',target=y)
    x=frame.to_numpy(dtype=float)
    np.savez(stage/'h9-representations.npz',api_frame_features=x,api_frame_target=y,api_frame_idx=np.arange(6))
    np.savez(stage/'received-and-api.npz',features=x+1 if fault=='values' else x,target=y[:,None],
             received_target=y[::-1,None] if fault=='target' else y[:,None],idx=np.arange(6))
    schema=dict(categorical_idx=[1]);params=dict(use_input_preprocessing=True)
    (stage/'h9-result.json').write_text(json.dumps(dict(api_frame=schema,params=params)))
    child=dict(schema=schema,params=params,model_fits=1 if fault=='fit' else 0,encoded_class_count=3,
               environment={'FEDOT_OPERATION_DEFAULTS_OVERLAY':None},received_frame_sha256=sha(stage/'received-frame.pkl'),arrays_sha256=sha(stage/'received-and-api.npz'))
    (stage/'child.json').write_text(json.dumps(child))
    (stage/'parent.json').write_text(json.dumps(dict(seconds=1,response=dict(models_count=0,training_duration=0,probe_sha256=sha(stage/'child.json')))))


def test_paired_transport_audit(tmp_path):
    fixture(tmp_path)
    assert audit(tmp_path)['gate_passed']


@pytest.mark.parametrize('fault',['dtype','columns','values','target','fit'])
def test_transport_auditor_rejects_corruption(tmp_path,fault):
    fixture(tmp_path,fault)
    with pytest.raises(AssertionError):audit(tmp_path)


@pytest.mark.skipif(not os.environ.get('H16_STAGE'),reason='host benchmark parent and framework child')
def test_real_parent_callee_api_without_fit(tmp_path):
    runtime=Path(os.environ['H16_STAGE'])
    for name in ['source','harness']:(tmp_path/name).symlink_to(runtime/name,target_is_directory=True)
    x=np.random.RandomState(104742).normal(size=(60,4));x[:,2:]=np.arange(120).reshape(60,2)%3
    y=np.arange(60)%3
    frame=pd.DataFrame(x,columns=['numeric_a','numeric_b','category_a','category_b'])
    for name in ['category_a','category_b']:frame[name]=frame[name].astype('uint8')
    frame.to_pickle(tmp_path/'train-frame.pkl');np.savez(tmp_path/'train.npz',target=y)
    transport(tmp_path,os.environ['FEDOT_DIAGNOSTIC_PYTHON'])
    pd.testing.assert_frame_equal(frame,pd.read_pickle(tmp_path/'received-frame.pkl'),check_exact=True)
    r=json.loads((tmp_path/'child.json').read_text())
    assert r['model_fits']==0 and r['schema']['categorical_idx']==[2,3]
    assert r['encoded_class_count']==3 and r['environment']['FEDOT_OPERATION_DEFAULTS_OVERLAY'] is None
    with np.load(tmp_path/'received-and-api.npz') as z:
        assert np.array_equal(z['received_target'],y[:,None]) and np.array_equal(z['idx'],np.arange(60))
    assert len(list((tmp_path/'transport').iterdir()))==1, 'Transport artifacts must be retained'
