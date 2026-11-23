import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
from golem.core.tuning.tuner_interface import BaseTuner

from examples.advanced.diagnostics import calibrated_tuner_probe as probe


def test_calibration_and_reserve():
    assert probe.calibrated_cap(200)==260
    assert probe.calibrated_cap(290)==300
    assert probe.calibrated_cap(.4)==5
    assert probe.admitted(620,260) and not probe.admitted(619,260)
    for bad in [0,-1,float('inf'),float('nan')]:
        with pytest.raises(ValueError):probe.calibrated_cap(bad)


@pytest.mark.skipif(not hasattr(BaseTuner,'_record_tuning_results'),reason='requires frozen host GOLEM history implementation')
@pytest.mark.parametrize('scores,budget,expected_calls,selected',[
    ([1,.9,.9],1000,3,True),([1,1.1,1.1],1000,3,False),
    ([1,.999,.999],1000,3,False),([None],1000,1,False),
    ([1],200,1,False),([1,None,None],1000,3,False)])
def test_real_tuner_control_flow(scores,budget,expected_calls,selected):
    from datetime import timedelta
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from golem.core.optimisers.graph import OptGraph, OptNode
    def record_graph(graph):
        return [dict(operation=n.name,params=dict(n.parameters),
                     parents=[graph.nodes.index(p) for p in (n.nodes_from or [])]) for n in graph.nodes]
    scale=OptNode({'name':'scaling','params':{'with_mean':True,'with_std':True}})
    cat=OptNode({'name':'catboost','params':{'num_trees':3000,'learning_rate':.03,'use_eval_set':True,'n_jobs':8}},nodes_from=[scale])
    pipe=OptGraph(cat)
    before=record_graph(pipe)
    calls=[]
    class Evaluator:pass
    def record(evaluator,graph,phase,cap):
        if budget==200:assert 190<cap<195
        else:assert cap==(300 if not calls else 260)
        calls.append(record_graph(graph))
        score=scores[len(calls)-1]
        return (SingleObjFitness(score) if score is not None else SingleObjFitness()),200
    history=OptHistory()
    tuner=probe.make_tuner_class()(objective_evaluate=Evaluator(),deadline=time.monotonic()+budget,
        recorder=record,search_space=probe.fixed_space(before),
        iterations=1,timeout=timedelta(seconds=1000),n_jobs=8,history=history,deviation=1.)
    returned=tuner.tune(pipe,show_progress=False)
    assert len(calls)==expected_calls
    expected=[{**node,'params':dict(node['params'])} for node in before]
    next(n for n in expected if n['operation']=='catboost')['params']['use_eval_set']=False
    if expected_calls==3:
        assert calls==[before,expected,expected]
        assert tuner.evaluations_count==1 and len(history.generations)==3
    assert record_graph(returned)==(expected if selected else before)
    assert record_graph(history.generations[-1][0].graph)==record_graph(returned)
    if scores[0] is None:assert not history.generations[-1][0].fitness.valid


def test_actual_hyperopt_space_retains_every_parameter():
    from golem.core.optimisers.graph import OptGraph,OptNode
    from golem.core.tuning.hyperopt_tuner import get_node_parameters_for_hyperopt
    from hyperopt.pyll.stochastic import sample
    nodes=[dict(operation='catboost',params=dict(num_trees=3000,use_eval_set=True,learning_rate=.03)),
           dict(operation='scaling',params=dict(with_mean=True,with_std=False))]
    space=probe.fixed_space(nodes)
    scale=OptNode({'name':'scaling','params':nodes[1]['params']})
    graph=OptGraph(OptNode({'name':'catboost','params':nodes[0]['params']},nodes_from=[scale]))
    parameters={}
    for i,node in enumerate(graph.nodes):
        parameters.update(get_node_parameters_for_hyperopt(space,i,node)[0])
    BaseTuner.set_arg_graph(graph,sample(parameters))
    assert graph.nodes[0].parameters==dict(num_trees=3000,use_eval_set=False,learning_rate=.03)
    assert graph.nodes[1].parameters==nodes[1]['params']
    with pytest.raises(ValueError):probe.fixed_space([nodes[0],nodes[0]])


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'),reason='isolated host API and H13 patch required')
def test_actual_builder_api_cv_history_and_pools(tmp_path):
    import pandas as pd
    harness=Path(os.environ['FEDOT_H9_ADAPTER']).parents[2]
    (tmp_path/'harness').symlink_to(harness,target_is_directory=True)
    (tmp_path/'cells').mkdir()
    x=np.random.RandomState(42).normal(size=(60,4)).astype('float32')
    x[:,2:]=np.arange(120).reshape(60,2)%3
    y=np.arange(60)%3
    np.savez(tmp_path/'train.npz',features=x,target=y)
    pd.DataFrame(x,columns=['a','b','c','d']).to_pickle(tmp_path/'train-frame.pkl')
    result=probe.run_tuner(tmp_path,smoke=True)
    assert len(result['calls'])==3 and all(c['valid'] for c in result['calls'])
    audit=probe.audit(tmp_path,smoke=True)
    assert audit['integration_valid'] and audit['history_generations']==3
    # Independent auditor must reject a different returned graph or charged time.
    path=tmp_path/'result.json'
    original=path.read_text()
    broken=json.loads(original);broken['policy_seconds']=1101
    path.write_text(json.dumps(broken))
    with pytest.raises(AssertionError):probe.audit(tmp_path,smoke=True)
    path.write_text(original)
