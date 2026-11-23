"""Read-only failure-path audit; never reads outer labels or predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


MANIFEST = '2811371bc5db136402aa89a1e2bd5d076eba4553e5684e50bed308085ffd60c1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(stage):
    manifest = stage/'manifest.json'
    assert sha(manifest) == MANIFEST
    files = json.loads(manifest.read_text())['files']
    assert len(files) == 1496
    for name, digest in files.items():
        path = stage/name
        assert path.resolve().is_relative_to(stage.resolve())
        assert sha(path) == digest, name
    queue = json.loads((stage/'QUEUE_FINISHED').read_text())
    assert not (stage/'QUEUE_FAILED').exists()
    sides = {}
    for side in ['baseline', 'candidate']:
        run = stage/'runs'/side
        assert (run/'FAILED').exists() and not (run/'COMPLETE').exists()
        process = json.loads((run/'process.json').read_text())
        assert process['exit_code'] == 0 and not process['timed_out']
        assert process['seconds'] < 3900
        assert not (run/'cache').exists() and not (run/'observation').exists()
        log = (run/'run.log').read_text()
        chain = ['result = run_fn(ds, config)', 'fedot = _make_fedot(', 'model = Fedot(',
                 'self.api_composer = ApiComposer', 'self.init_cache()',
                 'self.operations_cache = OperationsCache',
                 'if cache_dir is None or Path(cache_dir).samefile(default_fedot_data_dir()):',
                 'st = self.stat()', 'FileNotFoundError:']
        offset = log.index('Traceback')
        for token in chain:
            offset = log.index(token, offset) + len(token)
        assert str(run/'cache') in log[offset:offset+500]
        # Read error/status columns only; do not interpret missing quality as infinite loss.
        score = run/'results'/f'fedot.h21-{side}.fabert.fold7/scores/results.csv'
        with score.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert 'NoResultError' in rows[0]['info'] and str(run/'cache') in rows[0]['info']
        sides[side] = dict(process=process, error=rows[0]['info'],
            failure_before_fit=True, fit_observation_absent=True,
            evidence_sha256={name: sha(run/name) for name in
                             ['STARTED', 'FAILED', 'process.json', 'run.log']})
    adapter = (stage/'harness/frameworks/FEDOT/exec.py').read_text()
    start = adapter.index('    fedot = _make_fedot(')
    assert adapter.index('    with Timer() as training:', start) > start
    assert adapter.index('        fedot.fit(', start) > start
    return dict(manifest_sha256=MANIFEST, verified_files=len(files), queue=queue,
        sides=sides, mechanism_gate=False, outer_descriptive_gate=None,
        classification='Both framework jobs failed in constructor before Fedot.fit; no quality result',
        zero_fit_basis='Traceback plus adapter control flow, not an installed fit counter',
        original_auditor_limitation='Success-path auditor requires absent observation/fit.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().stage), indent=2))
