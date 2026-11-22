"""Create an exclusive NEW runtime stage; never modify the source runtime.

Run with system Python, before any framework imports. Source archive and its
expected SHA must be supplied by the caller after committing the protocol.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, series, archive, expected_sha, commit):
    if sys.version_info < (3, 9):
        raise RuntimeError('Use the framework Python (>=3.9), not system Python')
    root = root.resolve()
    if root != Path('/var/essdata') / os.environ['USER'] / 'fedot-amlb-official-1h8c':
        raise ValueError('Unexpected benchmark root')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', series):
        raise ValueError('Invalid series')
    if not re.fullmatch(r'[a-f0-9]{40}', commit) or sha(archive) != expected_sha:
        raise ValueError('Archive identity mismatch')
    diag = root / 'experiments/pr1467-diagnostics-20261109'
    stage = diag / series
    stage.mkdir(exist_ok=False)
    (stage / 'archive.sha256').write_text(expected_sha + '\n')
    for variant in ('baseline', 'candidate'):
        code = stage / variant
        code.mkdir()
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                destination = (code / member.name).resolve()
                if not destination.is_relative_to(code):
                    raise ValueError('Unsafe archive entry')
                if member.issym() or member.islnk():
                    link_base = destination.parent if member.issym() else code
                    if not (link_base / member.linkname).resolve().is_relative_to(code):
                        raise ValueError('Archive link escapes the new source tree')
                elif not (member.isfile() or member.isdir()):
                    raise ValueError('Unsupported archive member type')
            bundle.extractall(code)
        (stage / (variant + '.commit')).write_text(commit + '\n')
        vendor = stage / 'vendor' / variant
        shutil.copytree(diag / 'vendor', vendor,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        individual = vendor / 'golem/core/optimisers/opt_history_objects/individual.py'
        original = '7b0d1de39ad586774883025bc1c690ac1539634a9f71b43a249a059c123af527'
        corrected = '1d4f11ca785b4cb60f33d717db08c649104054b5fabeab41771085309ff61cb0'
        if sha(individual) == original:
            subprocess.run(['git', 'apply', '--ignore-space-change',
                            str(code / 'examples/advanced/diagnostics/golem_metadata_cow.patch')],
                           cwd=vendor, check=True)
        if sha(individual) != corrected:
            raise ValueError('Unrecognized metadata implementation')
    old_harness = root / 'experiments/pr1467-5fbcaf16/automlbenchmark'
    harness = stage / 'harness'
    # Follow the top-level deployment symlink into a NEW real tree. Never follow
    # a framework venv; those are read-only interpreter dependencies.
    shutil.copytree(old_harness.resolve(), harness, symlinks=True,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc',
                                                 'venv', '.venv', 'results', 'output'))
    (harness / 'frameworks/FEDOT/venv').symlink_to(
        root / 'automlbenchmark/frameworks/FEDOT/venv', target_is_directory=True)
    adapter = harness / 'frameworks/FEDOT/__init__.py'
    if sha(adapter) == '497587c2e49990e8c244efdbb2c75683907649f0ffa522653dc1373ac582c4e9':
        subprocess.run(['git', 'apply', '--ignore-space-change',
                        str(stage / 'candidate/examples/advanced/diagnostics/amlb_overlay_transport.patch')],
                       cwd=harness, check=True)
    if sha(adapter) != '96dcc35e61a13c0f35c1d9b551e7bfd27992c280d1eb29b73af69e353ae14c2c':
        raise ValueError('Unexpected harness adapter')
    runtime = ['harness/frameworks/FEDOT/__init__.py', 'harness/frameworks/FEDOT/exec.py',
               'harness/frameworks/shared/caller.py']
    for variant in ('baseline', 'candidate'):
        runtime.extend(f'vendor/{variant}/golem/core/{part}' for part in (
            'optimisers/opt_history_objects/individual.py', 'tuning/tuner_interface.py'))
    (stage / 'runtime.sha256').write_text(''.join(f'{sha(stage / p)}  {p}\n' for p in runtime))
    # Full Python source map guards against changes outside the abbreviated
    # compatibility manifest, including unintentional changes to models.
    source_map = {}
    for name in ('baseline', 'candidate', 'vendor', 'harness'):
        for path in sorted((stage / name).rglob('*.py')):
            if not path.is_symlink() and 'venv' not in path.relative_to(stage).parts:
                source_map[str(path.relative_to(stage))] = sha(path)
    (stage / 'all-source-sha256.json').write_text(json.dumps(source_map, indent=2))
    (stage / 'PREPARED').write_text(commit + '\n')
    print(json.dumps(dict(stage=str(stage), source_commit=commit,
                          archive_sha256=expected_sha,
                          runtime_sha256=sha(stage / 'runtime.sha256'),
                          all_sources_sha256=sha(stage / 'all-source-sha256.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--series', required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    prepare(args.root, args.series, args.archive, args.sha256, args.commit)
