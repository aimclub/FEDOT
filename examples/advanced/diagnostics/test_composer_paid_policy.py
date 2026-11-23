from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import os
import pytest

from examples.advanced.diagnostics.composer_paid_policy import resources,patch_composer,patch_adapter,ComposerPaidPolicy


def test_real_wall_budget_not_double_subtracted():
    r=resources(3240,100,2000,324)
    assert r['remaining']==1340 and r['allocation']==854
    assert r['tuner_deadline']==2854 and r['api_deadline']==3340
    assert r['refit_reserve']+r['postfit_reserve']==486
    assert resources(3240,100,1500,324)['allocation']==1100


@pytest.mark.parametrize('args',[(None,1,2,3),(3240,None,2,3),(3240,1,2,None),
    (3240,2,1,3),(3240,1,float('nan'),3),(-1,1,2,3),(100,1,99,50)])
def test_invalid_or_exhausted_budget_skips(args):
    assert resources(*args) is None


def test_composer_patch_only_adds_opt_in_dispatch():
    original=Path(os.environ.get('FEDOT_H18_BASE_COMPOSER','fedot/api/api_utils/api_composer.py')).read_text(encoding='utf8')
    patched=patch_composer(original)
    start=patched.index("        experimental_policy = getattr(self, '_experimental_paid_policy', None)")
    end=patched.index('        timeout_for_tuning_sec',start)
    assert patched[:start]+patched[end:]==original
    with pytest.raises(AssertionError):patch_composer(patched)


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'),reason='frozen host adapter/runtime')
def test_adapter_explicit_opt_in_and_no_flag_parity():
    import importlib.util
    path=Path(os.environ['FEDOT_H9_ADAPTER'])
    spec=importlib.util.spec_from_file_location('h18_test_adapter',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    def make(flag):
        c=SimpleNamespace(type='classification',seed=104735,framework_params={} if flag is None else {'_paid_allrow_tuning':flag})
        return adapter._make_fedot(c,'neg_log_loss',{'n_jobs':8,'use_operations_cache':False,
            'use_preprocessing_cache':False,'use_predictions_cache':False},5.4,.54)
    for flag in [None,False]:assert not hasattr(make(flag).api_composer,'_experimental_paid_policy')
    assert isinstance(make(True).api_composer._experimental_paid_policy,ComposerPaidPolicy)


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'),reason='frozen host composer/runtime')
def test_exhausted_budget_dispatch_does_not_build_or_evaluate(monkeypatch):
    from fedot.api.api_utils.api_composer import ApiComposer
    from fedot.core.pipelines.tuning import tuner_builder
    from examples.advanced.diagnostics import composer_paid_policy as policy
    from fedot.api.time import ApiTime
    params=SimpleNamespace(timeout=1,get=lambda *args:False,
        composer_requirements=SimpleNamespace(max_graph_fit_time=timedelta(seconds=30)))
    model=SimpleNamespace(params=params,_fit_started_at=0)
    composer=SimpleNamespace(params=params,was_tuned=True,timer=ApiTime(time_for_automl=1,with_tuning=True))
    composer._experimental_paid_policy=ComposerPaidPolicy(model)
    monkeypatch.setattr(policy,'eligible',lambda *args:True)
    monkeypatch.setattr(tuner_builder,'TunerBuilder',lambda *args:pytest.fail('Unexpected tuner build'))
    graph=object()
    assert ApiComposer.tune_final_pipeline(composer,None,graph,None) is graph
    assert not composer.was_tuned and composer._experimental_paid_policy.record['calls']==[]


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'),reason='frozen host composer/runtime')
def test_ineligible_dispatch_falls_through_original(monkeypatch):
    from fedot.api.api_utils.api_composer import ApiComposer
    from examples.advanced.diagnostics import composer_paid_policy as policy
    class ReachedOriginal(Exception):pass
    def original():raise ReachedOriginal()
    params=SimpleNamespace()
    composer=SimpleNamespace(params=params,timer=SimpleNamespace(determine_resources_for_tuning=original))
    composer._experimental_paid_policy=ComposerPaidPolicy(SimpleNamespace(params=params))
    monkeypatch.setattr(policy,'eligible',lambda *args:False)
    with pytest.raises(ReachedOriginal):ApiComposer.tune_final_pipeline(composer,None,object(),None)


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'),reason='frozen host real API final refit')
def test_tiny_real_api_composer_final_refit(tmp_path):
    from examples.advanced.diagnostics.composer_budget_probe import run_case,audit
    (tmp_path/'harness').symlink_to(Path(os.environ['FEDOT_H9_ADAPTER']).parents[2],target_is_directory=True)
    run_case(tmp_path,smoke=True)
    assert audit(tmp_path)['gate_passed']
