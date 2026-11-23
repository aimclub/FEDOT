"""H25 opt-in RNG patch, applicable only to a new isolated vendor copy."""
import difflib
import hashlib
from pathlib import Path


def patched(name, source):
    assert 'rstate=self._hyperopt_rstate' not in source
    if name=='hyperopt_tuner.py':
        old='algo: Callable = tpe.suggest, **kwargs):'
        assert source.count(old)==1
        source=source.replace(old,'algo: Callable = tpe.suggest, random_state: Optional[int] = None, **kwargs):')
        old='        self.algo = algo\n'
        assert source.count(old)==1
        source=source.replace(old,old+'        self._hyperopt_rstate = None if random_state is None else np.random.default_rng(random_state)\n')
    else:
        assert name=='simultaneous.py'
    old='algo=self.algo,\n'
    assert source.count(old)==1
    indentation='             ' if name=='hyperopt_tuner.py' else '                         '
    return source.replace(old,old+indentation+'rstate=self._hyperopt_rstate,\n')


def apply(stage):
    stage=Path(stage)
    assert not (stage/'STARTED').exists()
    changes=[];hashes={}
    for name in ['hyperopt_tuner.py','simultaneous.py']:
        p=stage/'vendor/golem/core/tuning'/name
        old=p.read_text();new=patched(name,old)
        changes.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=name,tofile=name))
        hashes[name]=dict(before=hashlib.sha256(p.read_bytes()).hexdigest())
        p.write_text(new)
        hashes[name]['after']=hashlib.sha256(p.read_bytes()).hexdigest()
    with (stage/'seeded-hyperopt.patch').open('x') as out:out.write(''.join(changes))
    return hashes
