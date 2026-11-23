"""H18 experimental adapter opt-in. Installed ONLY in immutable candidate copies."""
from datetime import timedelta
import math
from pathlib import Path
import time


def resources(api_seconds, started, now, refit_seconds):
    values=(api_seconds,started,now,refit_seconds)
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        return None
    if api_seconds<=0 or refit_seconds<=0 or now<started:
        return None
    postfit=max(5.,api_seconds*.05)
    remaining=started+api_seconds-now
    allocation=min(1100.,remaining-refit_seconds-postfit)
    # Three >=5s evaluations and two 50s overhead reserves must fit initially.
    if allocation<115:
        return None
    initial_cap=min(300.,(allocation-100.)/3.)
    return dict(api_deadline=started+api_seconds,remaining=remaining,refit_reserve=refit_seconds,
                postfit_reserve=postfit,allocation=allocation,initial_cap=initial_cap,
                tuner_deadline=now+allocation)


def eligible(pipeline, data, composer):
    if getattr(getattr(data.task,'task_type',None),'value',None)!='classification':
        return False
    if composer.params.composer_requirements.cv_folds!=3 or len(composer.metrics)!=1:
        return False
    if len(pipeline.nodes)!=2 or pipeline.root_node.operation.operation_type!='catboost':
        return False
    parents=pipeline.root_node.nodes_from
    return (len(parents or [])==1 and parents[0].operation.operation_type=='scaling'
            and not parents[0].nodes_from and pipeline.root_node.parameters.get('use_eval_set') is True)


def install(model):
    """Called by the patched real adapter only for _paid_allrow_tuning=True."""
    if hasattr(model.api_composer,'_experimental_paid_policy'):
        raise ValueError('Policy already installed')
    model.api_composer._experimental_paid_policy=ComposerPaidPolicy(model)


class ComposerPaidPolicy:
    def __init__(self, model):
        self.model=model
        self.record=None

    def __call__(self, composer, data, pipeline, history):
        from examples.advanced.diagnostics.calibrated_tuner_probe import make_tuner_class,parameter_space,graph_record
        from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
        if not eligible(pipeline,data,composer):
            self.record=dict(status='ineligible_original_path')
            return NotImplemented
        req=composer.params.composer_requirements
        refit=getattr(req,'max_graph_fit_time',None)
        total=getattr(self.model.params,'timeout',None)
        started=getattr(self.model,'_fit_started_at',None)
        now=time.monotonic()
        bounds=resources(total*60 if total is not None else None,started,now,
                         refit.total_seconds() if refit is not None else None)
        if bounds is None or self.model.params.get('final_fit_use_remaining_time',False):
            composer.was_tuned=False
            self.record=dict(status='insufficient_or_unsupported_budget',calls=[])
            return pipeline
        self.record=dict(status='running',bounds=bounds,initial_graph=graph_record(pipeline),calls=[])

        def recorder(evaluator,graph,phase,cap):
            before=time.monotonic()
            assert evaluator._operations_cache is None and evaluator._preprocessing_cache is None and evaluator._predictions_cache is None
            original=graph_record(graph)
            fitness=evaluator(graph)
            seconds=time.monotonic()-before
            self.record['calls'].append(dict(phase=phase,cap=cap,seconds=seconds,graph=original,
                                            valid=bool(fitness.valid),fitness=float(fitness.value) if fitness.valid else None))
            return fitness,seconds

        tuning_start=time.monotonic()
        try:
            with composer.timer.launch_tuning():
                tuner=(TunerBuilder(data.task).with_tuner(make_tuner_class())
                       .with_metric(composer.metrics[0]).with_requirements(req)
                       .with_eval_time_constraint(req.max_graph_fit_time)
                       .with_iterations(1).with_timeout(timedelta(seconds=bounds['allocation']))
                       .with_evaluation_time_constraint(timedelta(seconds=bounds['initial_cap']))
                       .with_search_space(parameter_space(pipeline)).with_history(history)
                       .with_additional_params(deadline=bounds['tuner_deadline'],recorder=recorder,
                                               ceiling=bounds['initial_cap'],overhead=50,deviation=1.).build(data))
                returned=tuner.tune(pipeline,show_progress=False)
                composer.was_tuned=tuner.was_tuned
                self.record.update(status='evaluated',returned_graph=graph_record(returned),
                                   admission=tuner.admission,evaluations_count=tuner.evaluations_count,
                                   init_metric=tuner.init_metric,obtained_metric=tuner.obtained_metric,
                                   was_tuned=tuner.was_tuned)
                return returned
        finally:
            # ApiTime's ordinary context does not account an exceptional exit.
            elapsed=time.monotonic()-tuning_start
            composer.timer.tuning_spend_time=timedelta(seconds=elapsed)
            self.record['charged_tuning_seconds']=elapsed
            self.record['remaining_after_tuning']=bounds['api_deadline']-time.monotonic()


def patch_composer(source):
    old='''        """ Launch tuning procedure for obtained pipeline by composer """
        timeout_for_tuning_sec = max(0, self.timer.determine_resources_for_tuning())'''
    new='''        """ Launch tuning procedure for obtained pipeline by composer """
        experimental_policy = getattr(self, '_experimental_paid_policy', None)
        if experimental_policy is not None:
            experimental_result = experimental_policy(self, train_data, pipeline_gp_composed, history)
            if experimental_result is not NotImplemented:
                return experimental_result
        timeout_for_tuning_sec = max(0, self.timer.determine_resources_for_tuning())'''
    assert source.count(old)==1
    return source.replace(old,new)


def patch_adapter(source):
    old='''def _make_fedot(
    config, scoring_metric, training_params, runtime_min, max_pipeline_fit_time
):
    return Fedot(
        problem=config.type,
        timeout=runtime_min,
        metric=scoring_metric,
        seed=config.seed,
        max_pipeline_fit_time=max_pipeline_fit_time,
        **training_params,
    )'''
    new=old.replace('    return Fedot(', '    model = Fedot(')+'''
    if _as_bool(config.framework_params.get("_paid_allrow_tuning", False)):
        from examples.advanced.diagnostics.composer_paid_policy import install
        install(model)
    return model'''
    assert source.count(old)==1
    return source.replace(old,new)


def apply(stage):
    stage=Path(stage)
    assert not (stage/'STARTED').exists()
    for relative,patch in [('source/fedot/api/api_utils/api_composer.py',patch_composer),
                           ('harness/frameworks/FEDOT/exec.py',patch_adapter)]:
        p=stage/relative
        p.write_text(patch(p.read_text()),encoding='utf8')
