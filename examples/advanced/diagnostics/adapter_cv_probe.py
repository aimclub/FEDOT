"""H10: fixed native CV under actual API data preparation, two cells only."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from examples.advanced.diagnostics.adapter_fidelity_probe import capture, summarize
from examples.advanced.diagnostics.cache_cost_probe import worker, sha, write_json, probability_loss
from examples.advanced.diagnostics.fresh_budget_probe import training_conflict

LABELS = ('fresh_api32', 'fresh_api_frame')


def load_adapter(stage):
    path = stage/'harness/frameworks/FEDOT/exec.py'
    spec = importlib.util.spec_from_file_location('h10_actual_adapter',path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def cell(stage, label):
    import pandas as pd
    assert label in LABELS
    with np.load(stage/'train.npz',allow_pickle=False) as z:
        x,y=z['features'].copy(),z['target'].copy()
    features=x if label=='fresh_api32' else pd.read_pickle(stage/'train-frame.pkl')
    captured=capture(load_adapter(stage),features,y)
    write_json(stage/'cells'/(label+'.input.json'),
               dict(schema=summarize(captured['data']),params=captured['params']))
    worker(stage,label,spec={'cap':600},input_data=captured['data'])


def paired_audit(stage):
    with np.load(stage/'h7-reference.npz',allow_pickle=False) as z:
        ref={k:z[k].copy() for k in z.files}
    old=json.loads((stage/'h7-independent-audit.json').read_text())
    rows=np.concatenate([ref[f'rows_{i}'] for i in range(3)])
    assert np.array_equal(np.sort(rows),np.arange(7413)), 'H7 row coverage'
    with np.load(stage/'train.npz',allow_pickle=False) as z:
        y=z['target'].ravel()
    results={}
    predictions={}
    for label in LABELS:
        p=stage/'cells'/(label+'.json')
        r=json.loads(p.read_text())
        process=json.loads(p.with_suffix('.process.json').read_text())
        assert process['exit_code']==0 and process['seconds']<=665
        assert r['valid'] and len(r['folds'])==3 and not r['cache_enabled']
        assert r['graph_id_before']==r['graph_id_after']==old['cells']['fresh_a']['graph_id_before']
        assert r['npz_sha256']==sha(p.with_suffix('.npz')) and r['data_sha256']==sha(stage/'train.npz')
        losses=[];diff=[];predictions[label]=[]
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
            assert np.array_equal(z['classes'],ref['classes'])
            for i in range(3):
                for key in [f'rows_{i}',f'train_rows_{i}',f'truth_{i}']:
                    assert np.array_equal(z[key],ref[key]),key
                assert np.array_equal(z[f'truth_{i}'],y[z[f'rows_{i}']])
                probs=z[f'probs_{i}']
                losses.append(probability_loss(probs,z[f'truth_{i}'],z['classes']))
                diff.append(float(np.max(np.abs(probs-ref[f'probs_{i}']))))
                predictions[label].append(probs.copy())
        assert abs(np.mean(losses)-r['fitness'])<1e-10
        results[label]=dict(cv=float(np.mean(losses)),fold_losses=losses,seconds=r['seconds'],
                            process_seconds=process['seconds'],max_difference_from_direct=max(diff),
                            fitted=[f['fitted_nodes'] for f in r['folds']])
    native=old['independent_losses']['fresh_a']
    return dict(cells=results,direct_h7_cv=native,
                api_pair_max_difference=max(float(np.max(np.abs(a-b))) for a,b in
                                           zip(predictions[LABELS[0]],predictions[LABELS[1]])),
                api32_relative_improvement=1-results[LABELS[0]]['cv']/native,
                representation_quality_gate=results[LABELS[0]]['cv']<=.99*native,
                scope='train-only same splits; no independent quality or full-job admission')


def suite(stage):
    import fcntl
    started=time.monotonic()
    with (stage.parents[2]/'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:args=[a.decode(errors='replace') for a in p.read_bytes().split(b'\0') if a]
            except OSError:continue
            pid=int(p.parent.name)
            conflict=training_conflict(pid,args,os.getpid(),os.getppid(),str(Path(__file__).resolve()))
            own_parent=pid==os.getppid() and args and Path(args[0]).name=='timeout' and str(Path(__file__).resolve()) in args
            extra=any(Path(a).name in ('adapter_cv_probe.py','adapter_fidelity_probe.py') for a in args)
            if pid!=os.getpid() and not own_parent and (conflict or extra):
                raise RuntimeError('Host occupied: '+str(pid))
        manifest=json.loads((stage/'manifest.json').read_text())
        assert manifest['preflight_passed']
        for n,h in manifest['files'].items():assert sha(stage/n)==h,n
        write_json(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
        try:
            for label in LABELS:
                before=time.monotonic()
                with (stage/'cells'/(label+'.log')).open('x') as log:
                    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--stage',str(stage),'--label',label],
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=p.wait(timeout=660)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                        code=p.returncode
                write_json(stage/'cells'/(label+'.process.json'),dict(exit_code=code,seconds=time.monotonic()-before))
                if code!=0:raise RuntimeError('CV worker failed: '+label)
            audit=paired_audit(stage)
            audit['stage_seconds']=time.monotonic()-started
            assert audit['stage_seconds']<=1400
            write_json(stage/'independent-audit.json',audit)
            write_json(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        except BaseException as error:
            write_json(stage/'FAILED',dict(error=repr(error)))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['suite','worker','audit'])
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--label')
    a=p.parse_args()
    if a.mode=='suite':suite(a.stage)
    elif a.mode=='worker':cell(a.stage,a.label)
    else:print(json.dumps(paired_audit(a.stage)))
