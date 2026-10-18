"""Archive the one completed fake-only CPU attempt; never runs its tests."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

R = Path('/Users/ryanli/code/minivllm')
SOURCE = R / '.artifacts/m5-ultra-256gb/gdn-canary-driver-cpu-v1'
OUT = R / 'docs/experiments/m5-ultra-256gb/gdn-canary-driver-cpu-v1'
RESULT_SHA = '9b67022100f2f0c7914505e8d48ce64068048a40b397687a8dcd420353400fad'


def need(ok, message):
    if not ok: raise ValueError(message)


def info(path):
    raw = Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def run():
    need(not OUT.exists(), 'fresh archive required')
    need(info(SOURCE / 'result.json')['sha256'] == RESULT_SHA, 'exact completed CPU result')
    result = json.loads((SOURCE / 'result.json').read_bytes())
    need(result['status'] == 'PASS' and result['tests'] == 12
         and result['failures'] == result['errors'] == result['skipped'] == 0
         and result['source_unchanged'] is True and result['denied_operations'] == []
         and result['forbidden_modules_loaded'] == [] and len(result['test_outcomes']) == 12
         and all(row['status'] == 'PASS' for row in result['test_outcomes']), 'actual twelve fake CPU tests')
    paths = sorted(SOURCE.rglob('*'))
    need(not any(p.is_symlink() for p in paths), 'no symlinks in CPU evidence')
    files = [p for p in paths if p.is_file()]
    need(all(p.is_file() or p.is_dir() for p in paths), 'regular source tree')
    inventory = json.loads((SOURCE / 'artifact-manifest.json').read_bytes())
    need(inventory['schema'] == 'cpu-artifact-inventory-v1'
         and set(inventory['files']) == {str(p.relative_to(SOURCE)) for p in files if p.name != 'artifact-manifest.json'},
         'complete CPU inventory membership')
    for name, expected in inventory['files'].items():
        path = SOURCE / name
        need(path.resolve().is_relative_to(SOURCE.resolve()) and info(path) == expected, 'CPU inventory: ' + name)
    need(result['source_before'] == result['source_after'], 'unchanged CPU source inventory')
    for name, expected in result['source_before'].items():
        need(info(SOURCE / 'frozen-source' / name.lstrip('/')) == expected, 'actual frozen source: ' + name)
    OUT.mkdir()
    (OUT / 'archive.py').write_bytes(Path(__file__).read_bytes())
    entries = {str(Path('cpu') / p.relative_to(SOURCE)): dict(source=str(p), **info(p)) for p in files}
    entries['archive.py'] = dict(source=str(OUT / 'archive.py'), **info(OUT / 'archive.py'))
    write(OUT / 'manifest.json', dict(schema='gdn-canary-driver-cpu-archive-manifest-v1', files=entries))
    archive = OUT / 'evidence.tar.xz'
    with tarfile.open(archive, 'w:xz', dereference=True) as tar:
        for name, entry in entries.items(): tar.add(entry['source'], arcname=name, recursive=False)
    with tarfile.open(archive, 'r:xz') as tar:
        members = tar.getmembers()
        need(len(members) == len(entries) and {m.name for m in members} == set(entries), 'exact tar membership')
        for member in members:
            need(member.isfile(), 'regular tar member')
            entry = entries[member.name]; raw = tar.extractfile(member).read()
            need(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'], 'tar payload')
            need(info(entry['source']) == {k: entry[k] for k in ('sha256', 'bytes')}, 'original file changed')
    need({str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()}
         == {str(p.relative_to(SOURCE)) for p in files}, 'CPU tree membership changed')
    for name in ('result.json', 'stdout.txt', 'stderr.txt'):
        (OUT / name).write_bytes((SOURCE / name).read_bytes())
    write(OUT / 'archive-verification.json', dict(schema='gdn-canary-driver-cpu-archive-verification-v1',
        status='PASS', archive=info(archive), members=len(entries), raw_bytes=sum(x['bytes'] for x in entries.values()),
        result_sha256=RESULT_SHA, tests=12, complete_original_files_rechecked=True,
        every_member_reopened_and_hash_verified=True, gpu_qualified=False, numerical_qualified=False))
    (OUT / 'README.md').write_text(
        '# GDN canary driver: CPU wiring evidence\n\n'
        'One guarded attempt ran all 12 whole-module fake CPU tests: PASS, zero failures, errors or skips. '
        'No backend imports, real subprocesses, network, signals, services or GPU work were allowed. '
        'The historical oMLX restoration was an evidence prerequisite, not an observation of the current service.\n\n'
        'The full runner, command, stdout/stderr, per-test outcomes and before/after source snapshots are in '
        '`evidence.tar.xz`. `manifest.json` maps each original file to its exact archive member, SHA256 and size; '
        '`archive-verification.json` records complete member and original-file rechecks. '
        'Tests cover default-inert execution, fixed CLI without a child --run, source/PREPARED binding, '
        'failed/unknown launch and finally-recovery routing. They do not prove device execution, numerical '
        'correctness, physical resource safety or real restoration. Original artifacts remain untouched.\n')
    print(json.dumps(json.loads((OUT / 'archive-verification.json').read_bytes())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run: run()
    else: print('DESCRIPTION_ONLY; no archive created.')
