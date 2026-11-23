"""H26 explicit adapter/composer seed opt-in, isolated copies only."""
import difflib
import hashlib
from pathlib import Path


def parse_seed(value):
    if value is None:
        return None
    if isinstance(value,str) and value.isascii() and value.isdecimal():
        value=int(value)
    if isinstance(value,bool) or not isinstance(value,int) or value<0:
        raise ValueError('_tuner_random_state must be a nonnegative integer or null')
    return value


def replace_once(source,old,new):
    assert source.count(old)==1
    return source.replace(old,new)


def apply(stage):
    stage=Path(stage);assert not (stage/'STARTED').exists()
    edits={
        'harness/frameworks/FEDOT/exec.py':[
            ('def _make_fedot(\n    config, scoring_metric, training_params, runtime_min, max_pipeline_fit_time\n):\n    model = Fedot(',
             'def _make_fedot(\n    config, scoring_metric, training_params, runtime_min, max_pipeline_fit_time\n):\n    from examples.advanced.diagnostics.composer_seed_patch import parse_seed\n    tuner_seed = parse_seed(config.framework_params.get("_tuner_random_state"))\n    model = Fedot('),
            ('    observation_dir = config.framework_params.get("_paid_policy_observation_dir")',
             '    if tuner_seed is not None:\n        model.api_composer._experimental_tuner_random_state = tuner_seed\n    observation_dir = config.framework_params.get("_paid_policy_observation_dir")')],
        'source/fedot/api/api_utils/api_composer.py':[
            ('                 .with_history(history))\n\n        bounds = bounded_tuning_resources(',
             '                 .with_history(history))\n        tuner_seed = getattr(self, "_experimental_tuner_random_state", None)\n        if tuner_seed is not None:\n            builder = builder.with_additional_params(random_state=tuner_seed)\n\n        bounds = bounded_tuning_resources(')],
        'source/examples/advanced/diagnostics/composer_paid_policy.py':[
            ('ceiling=bounds[\'initial_cap\'],overhead=50,deviation=1.).build(data))',
             'ceiling=bounds[\'initial_cap\'],overhead=50,deviation=1.,\n                                               random_state=getattr(composer, "_experimental_tuner_random_state", None)).build(data))')]}
    hashes={};diff=[]
    for name,replacements in edits.items():
        p=stage/name;old=p.read_text();new=old
        for a,b in replacements:new=replace_once(new,a,b)
        hashes[name]=dict(before=hashlib.sha256(p.read_bytes()).hexdigest())
        p.write_text(new);hashes[name]['after']=hashlib.sha256(p.read_bytes()).hexdigest()
        diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=name,tofile=name))
    with (stage/'composer-seed.patch').open('x') as out:out.write(''.join(diff))
    return hashes
