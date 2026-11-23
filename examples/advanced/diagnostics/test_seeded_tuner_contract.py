import os
from pathlib import Path
import pytest
from examples.advanced.diagnostics.seeded_hyperopt_patch import patched


def test_patch_requires_exact_source_and_is_not_reapplied():
    with pytest.raises(AssertionError):patched('hyperopt_tuner.py','changed source')
    source='algo=self.algo,\n'
    result=patched('simultaneous.py',source)
    assert 'rstate=self._hyperopt_rstate' in result
    with pytest.raises(AssertionError):patched('simultaneous.py',result)


@pytest.mark.skipif(not os.environ.get('FEDOT_H25_STAGE'),reason='isolated patched host vendor')
def test_real_builder_seed_replay_and_generator_continuity(tmp_path):
    from examples.advanced.diagnostics.seeded_tuner_contract import run_case
    stage=Path(os.environ['FEDOT_H25_STAGE'])
    graph=[dict(operation='logit',params={'C':1.0},parents=[1]),dict(operation='scaling',params={},parents=[])]
    records=[]
    for name in ['seed-a-first','seed-a-second']:
        output=tmp_path/name;output.mkdir();records.append(run_case(stage,output,name,graph))
    for key in ['calls','returned_graph','history','fmin','requests']:assert records[0][key]==records[1][key]


@pytest.mark.skipif(not os.environ.get('FEDOT_H25_STAGE'),reason='isolated patched host vendor')
def test_opt_in_default_and_independent_instances():
    import numpy as np
    from golem.core.tuning.simultaneous import SimultaneousTuner
    from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
    args=dict(objective_evaluate=lambda graph:None,search_space=PipelineSearchSpace())
    assert SimultaneousTuner(**args)._hyperopt_rstate is None
    a=SimultaneousTuner(**args,random_state=123);b=SimultaneousTuner(**args,random_state=123)
    assert a._hyperopt_rstate is not b._hyperopt_rstate
    assert np.array_equal(a._hyperopt_rstate.integers(100000,size=10),b._hyperopt_rstate.integers(100000,size=10))
    with pytest.raises(ValueError):SimultaneousTuner(**args,random_state=-1)
