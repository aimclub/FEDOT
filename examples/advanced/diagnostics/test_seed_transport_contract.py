import os
from pathlib import Path
import pytest
from examples.advanced.diagnostics.seed_transport_contract import flags, check_flags, parent


@pytest.mark.parametrize('name',['original-none','original-seeded','paid-none','paid-seeded'])
def test_exact_transport_forms(name):
    check_flags(name,flags(name))
    broken=flags(name);broken['_paid_allrow_tuning']=1
    with pytest.raises(AssertionError):check_flags(name,broken)


@pytest.mark.parametrize('name,value',[('original-none',None),('original-seeded',104758),('paid-seeded','104758')])
def test_missing_null_and_seed_types_are_distinct(name,value):
    broken=flags(name);broken['_tuner_random_state']=value
    with pytest.raises(AssertionError):check_flags(name,broken)


@pytest.mark.skipif(not os.environ.get('FEDOT_H27_STAGE'),reason='real benchmark parent/framework child')
@pytest.mark.parametrize('name',['original-seeded','paid-seeded'])
def test_real_seeded_transport_smoke(name,tmp_path):
    import json
    stage=Path(os.environ['FEDOT_H27_STAGE'])
    run=tmp_path/name;run.mkdir()
    parent(stage,run,name,os.environ['FEDOT_DIAGNOSTIC_PYTHON'],smoke=True)
    r=json.loads((run/'result.json').read_text())
    c=json.loads((run/'child.json').read_text())
    assert r['fits']==0 and r['unseeded_requests']==0
    assert c['factories'][0]['composer_seed']==104758
    assert len(r['calls'])==(3 if name.startswith('paid-') else 6)
