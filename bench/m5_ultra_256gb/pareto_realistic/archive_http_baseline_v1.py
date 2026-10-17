"""Archive the fixed HTTP v1 readiness rejection; never run a backend or mutate raw.

Requires the completed independent restoration audit. --run is explicit; without
it this script only prints the fixed inputs/destination. It never overwrites an
archive, reads model weights, launches processes, or changes the service.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARTIFACTS = ROOT / '.artifacts/m5-ultra-256gb'
PREREG = HERE / 'prereg-http-baseline-v1'
DESTINATION = ROOT / 'docs/experiments/m5-ultra-256gb/E023-http-baseline-v1'
TREES = {
    'raw/owner': ARTIFACTS / 'http-baseline-v1-owner',
    'raw/cold-journal': ARTIFACTS / 'http-baseline-v1-cold-journal',
    'raw/execution': ARTIFACTS / 'http-baseline-v1-execution',
    'preregistration': PREREG,
}
PLAN_SHA = 'd641a95b7a454ce45ea01c3f3b3b0bc59f08ec71de0f0d5c79973315f7861429'
CONTROLLER_SHA = '461355834222ad6820cc054c7a9450fd1b128cdb5e11491e2fdc915a08ae6974'
LISTENER_SHA = '5f8c8c5a0a539ac5cd247cffe62fd35306d17f5276983d246d9b8b94bc8bd73b'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest_stream(stream):
    digest = hashlib.sha256(); size = 0
    while block := stream.read(8 << 20):
        digest.update(block); size += len(block)
    return digest.hexdigest(), size


def file_info(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular non-symlink input required: ' + str(path))
    before = path.stat()
    with path.open('rb') as stream:
        digest, size = digest_stream(stream)
    after = path.stat()
    state = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    need(state(before) == state(after) and size == after.st_size, 'input changed while hashing: ' + str(path))
    return dict(sha256=digest, bytes=size, device=after.st_dev, inode=after.st_ino, mtime_ns=after.st_mtime_ns)


def read(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def tree_files(root):
    need(root.is_dir() and not root.is_symlink(), 'existing regular input directory required: ' + str(root))
    result = []
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(), 'symlink not silently followed: ' + str(path))
        need(path.is_dir() or path.is_file(), 'special file not archived: ' + str(path))
        if path.is_file():
            result.append(path)
    return result


def verify_fixed_run(audit_path):
    """Bind README facts to producer/raw evidence and the independent recovery receipt."""
    owner = TREES['raw/owner']; cold = TREES['raw/cold-journal']; execution = TREES['raw/execution']
    capture = owner / 'phases/c1-capture/capture'
    plan = read(PREREG / 'owner-plan.json')
    need(file_info(PREREG / 'owner-plan.json')['sha256'] == PLAN_SHA
         and file_info(owner / 'owner-plan.json')['sha256'] == PLAN_SHA, 'fixed v1 owner plan')
    complete = read(owner / 'complete.json'); result = read(capture / 'complete.json')
    need(complete['status'] == 'FAIL' and complete['owner_pid'] == 81158
         and complete['owner_plan_sha256'] == PLAN_SHA and complete['phases'] == []
         and complete['all_children_reaped'] is True and complete['numerical_replay_complete'] is False
         and complete['independently_qualified'] is False and complete['performance_qualified'] is False,
         'fixed failed owner facts differ')
    need(result['status'] == 'FAIL' and result['cleanup_complete'] is True and result['server_reaped'] is True
         and result['errors'] == [{'phase': 'readiness', 'type': 'ValueError',
                                  'message': 'ready listener is not the owned localhost server'}], 'fixed readiness rejection')
    need(not (capture / 'group.json').exists() and not (capture / 'client').exists()
         and not any((owner / 'phases' / name).exists() for name in ('c1-replay', 'c8-capture', 'c8-replay')),
         'unexpected client requests or later phase')
    trace_raw = (capture / 'trace.jsonl').read_bytes()
    trace = [json.loads(line) for line in trace_raw.splitlines()]
    need(trace_raw.endswith(b'\n') and len(trace) == 1 and trace[0]['event'] == 'startup', 'trace is not startup-only')
    events = [json.loads(line) for line in (capture / 'events.jsonl').read_bytes().splitlines()]
    need(not any(row.get('label') == 'checkpoint' and row.get('phase') in ('ready', 'capturing', 'after-client')
                 for row in events), 'unexpected request-stage event')
    listener = capture / 'listener.json'; fixture = HERE / 'http_capture_v2_listener_fixture.json'
    need(file_info(listener)['sha256'] == file_info(fixture)['sha256'] == LISTENER_SHA
         and listener.read_bytes() == fixture.read_bytes(), 'actual retained listener fixture mismatch')
    raw_listener = read(listener)
    need(raw_listener['returncode'] == 0 and raw_listener['stderr'] == ''
         and raw_listener['stdout'] == 'p81230\nf4\nn127.0.0.1:18023\n', 'actual listener raw changed')
    children = read(owner / 'children.json')['children']
    need(len(children) == 1 and children[0]['phase'] == 'c1-capture' and children[0]['pid'] == 81230,
         'exact one observed server child required')
    release = read(owner / 'phases/c1-capture/release.json')
    need(release['pid'] == 81230 and release['returncode'] == -15 and release['waited_reaped'] is True
         and release['strict_absence'] is True, 'server direct reap receipt')
    final = read(execution / 'final.json'); journal_path = cold / 'journal.json'; journal = read(journal_path)
    need(final['journal_state'] == 'RESTORED' and final['stop_returncode'] == 0
         and final['owner_returncode'] == 1 and final['recovery_returncode'] == 0, 'execution/recovery result')
    need(journal['state'] == 'RESTORED' and journal['controller_sha256'] == CONTROLLER_SHA
         and journal['replacement']['pid'] == 81337 and journal['replacement']['ppid'] == 1
         and journal['replacement']['comm'] == 'omlx-server'
         and journal['exact_runtime_signature_restored'] is False and journal['warm_kv_restored'] is False,
         'final persisted-visible service restoration')
    audit = read(audit_path)
    need(audit['schema'] == 'cold-service-restoration-audit-v1' and audit['status'] == 'PASS'
         and audit['qualified'] is True and audit['numerical_qualified'] is False
         and audit['performance_qualified'] is False and audit['ledger_opened'] is False
         and audit['journal_sha256'] == file_info(journal_path)['sha256']
         and audit['owner_plan_sha256'] == PLAN_SHA and audit['controller_sha256'] == CONTROLLER_SHA,
         'independent restoration-only audit binding')
    restored = audit['service_restore']
    need(restored['status'] == 'PERSISTED_VISIBLE_RECEIPTS_CONSISTENT'
         and restored['replacement'] == journal['replacement']
         and restored['raw_log_sha256'] == journal['log_window']['sha256']
         and restored['raw_log_bytes'] == journal['log_window']['bytes']
         and restored['http_worker_returncode'] == 0
         and restored['exact_runtime_signature_restored'] is False and restored['warm_kv_restored'] is False,
         'independent restoration facts differ')
    need(audit['owner_reap']['status'] == 'ALL_OBSERVED_CHILDREN_REAPED', 'independent reap audit required')
    return plan, journal, audit, dict(
        schema='E023-http-baseline-v1-failure-summary', status='REJECTED_BEFORE_HTTP_REQUESTS',
        owner_plan_sha256=PLAN_SHA, owner_pid=81158, inference_http_requests_started=0,
        observed_server_children=1, server_pid=81230, server_returncode=-15,
        server_waited_reaped=True, server_strict_absence=True,
        failure='v1 exact lsof field sequence rejected the implicit f4 descriptor at readiness',
        listener_sha256=LISTENER_SHA, numerical_qualified=False, performance_qualified=False,
        restoration_status='RESTORED', restored_identity=journal['replacement'],
        exact_runtime_signature_restored=False, warm_kv_restored=False,
        restoration_journal_sha256=file_info(journal_path)['sha256'],
        restoration_audit_sha256=file_info(audit_path)['sha256'],
        scope='Failure evidence only. Zero inference requests, no replay or numerical/performance conclusion.')


def archive(audit_path, out):
    need(not out.exists() and out.parent.is_dir(), 'new archive directory in existing parent required')
    audit_path = audit_path.resolve()
    plan, journal, audit, summary = verify_fixed_run(audit_path)
    inputs = {}
    tree_snapshot = {}
    for prefix, root in TREES.items():
        paths = tree_files(root)
        tree_snapshot[prefix] = [str(path.relative_to(root)) for path in paths]
        for path in paths:
            inputs[prefix + '/' + str(path.relative_to(root))] = path
    # Current bytes are admitted only against the v1 registered digests; later
    # v2 files cannot silently replace the failed run's implementation.
    pins = {**plan['execution']['pins'], **read(PREREG / 'receipt.json')['source_pins']}
    support = HERE.parent / 'omlx_cold_restart/controller.py'
    pins[str(support)] = 'aad297fb08515d6304fd3ae435301318650e8443ad22628c1ed2e4791887f0f3'
    for source, expected in pins.items():
        path = Path(source)
        need(path.is_relative_to(ROOT) and file_info(path)['sha256'] == expected, 'v1 registered source changed: ' + source)
        inputs['registered-source/' + str(path.relative_to(ROOT))] = path
    inputs['regression/http_capture_v2_listener_fixture.json'] = HERE / 'http_capture_v2_listener_fixture.json'
    inputs['sources/archive_http_baseline_v1.py'] = Path(__file__).resolve()
    # Locate audit source from its explicit receipt digest, never infer a passing
    # audit merely from a filename. The adjacent root-reviewed auditor is retained.
    for key, path in (
        ('audit_source_sha256', HERE.parent / 'omlx_cold_restart/audit_restore_v3.py'),
        ('raw_auditor_sha256', HERE / 'audit_absent_qualification.py'),
    ):
        need(path.is_file() and file_info(path)['sha256'] == audit[key], 'restoration auditor source pin')
        inputs['sources/' + path.name] = path
    audit_test = HERE.parent / 'omlx_cold_restart/test_audit_restore_v3_cpu.py'
    need(file_info(audit_test)['sha256'] == '1794bccdaeaac6229b02844b927f841f590393c8fc6f910855756421688802c8',
         'restoration auditor CPU-test pin')
    inputs['sources/' + audit_test.name] = audit_test
    out.mkdir()
    journal_source = TREES['raw/cold-journal'] / 'journal.json'
    for name, original in (('restored-journal.json', journal_source), ('restoration-audit.json', audit_path)):
        with (out / name).open('xb') as stream:
            stream.write(original.read_bytes())
        need((out / name).read_bytes() == original.read_bytes(), 'exact recovery sidecar copy')
        inputs[name] = out / name
    write_new(out / 'summary.json', summary)
    inputs['summary.json'] = out / 'summary.json'
    readme = '''# E023 HTTP baseline v1: readiness rejection

This attempt stopped before any inference HTTP request. The server startup trace
contains one startup record and no decode/prefill steps. The v1 listener check
rejected the actual macOS output `p81230`, `f4`, `n127.0.0.1:18023` because it
expected only PID and endpoint fields. This is a harness-format failure.
The installed `/usr/share/man/man8/lsof.8` documents that a file set begins
with `f` (lines 3201–3202) and that the file-descriptor field is always selected
(line 3256). The v2 parser admits that one numeric field while still requiring
exactly one owned PID and one exact localhost endpoint.

One server child, PID **81230**, was terminated, waited/reaped and confirmed absent.
No replay or C8 phase started. The original service was restored as PID **81337**;
the independent restoration-only audit is retained verbatim. The restoration
contract covers persisted settings and the visible loaded model. It does not claim
the same hidden runtime signature or warm KV state.

There is **no numerical qualification or performance conclusion** from this run.
The v2 regression fixture is byte-identical to the failed run's raw listener receipt.

`evidence.tar.xz` contains all regular files from the owner, cold-journal and
execution directories, all v1 preregistration files and registered source, plus
the regression fixture and archive/restoration audit source. Runtime binaries and
weights were never copied into those evidence directories; their retained pin
receipts remain included. Nothing in the original evidence is overwritten.

`manifest.json` lists every archive member's SHA-256, size and original path/stat.
`archive-verification.json` records a full extraction and per-member verification,
followed by a second hash/stat check of every original input and directory member
list. `restored-journal.json` and `restoration-audit.json` are exact-byte sidecars
for the next recovery cycle, not rewritten summaries.
'''
    (out / 'README.md').write_text(readme)
    inputs['README.md'] = out / 'README.md'
    records = {name: {'source': str(path), **file_info(path)} for name, path in sorted(inputs.items())}
    manifest = dict(schema='E023-http-failure-archive-manifest-v1', members=records,
                    members_count=len(records), raw_bytes=sum(row['bytes'] for row in records.values()),
                    scope='All tar payload members; manifest and final archive-verification are external nonrecursive receipts.')
    write_new(out / 'manifest.json', manifest)
    package = out / 'evidence.tar.xz'
    with tarfile.open(package, 'w:xz', dereference=True, preset=6) as tar:
        for name, path in sorted(inputs.items()):
            tar.add(path, arcname=name, recursive=False)
    # Do not extract arbitrary tar paths or links. Copy each verified regular
    # member into a temporary directory, then independently hash those bytes.
    verified = []
    with tempfile.TemporaryDirectory(prefix='verify-', dir=out) as temporary:
        target = Path(temporary)
        with tarfile.open(package, 'r:xz') as tar:
            seen = set()
            for member in tar:
                name = member.name; relative = PurePosixPath(name)
                need(member.isfile() and not relative.is_absolute() and '..' not in relative.parts
                     and name in records and name not in seen, 'unexpected/unsafe/duplicate archive member')
                seen.add(name); expected = records[name]
                need(member.size == expected['bytes'], 'archive member size mismatch: ' + name)
                dest = target.joinpath(*relative.parts); dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, dest.open('xb') as output:
                    shutil.copyfileobj(source, output, 8 << 20)
                actual = file_info(dest)
                need((actual['sha256'], actual['bytes']) == (expected['sha256'], expected['bytes']),
                     'extracted member SHA mismatch: ' + name)
                verified.append(name)
            need(seen == set(records), 'archive omitted members')
    for name, path in inputs.items():
        need(file_info(path) == {k: v for k, v in records[name].items() if k != 'source'}, 'original changed: ' + str(path))
    for prefix, root in TREES.items():
        need(tree_snapshot[prefix] == [str(p.relative_to(root)) for p in tree_files(root)], 'input tree membership changed')
    need((out / 'restored-journal.json').read_bytes() == journal_source.read_bytes()
         and (out / 'restoration-audit.json').read_bytes() == audit_path.read_bytes(), 'final exact-copy verification')
    package_info = file_info(package)
    verification = dict(schema='E023-http-failure-archive-verification-v1', status='PASS',
        archive={'path': package.name, 'sha256': package_info['sha256'], 'bytes': package_info['bytes']},
        manifest_sha256=file_info(out / 'manifest.json')['sha256'], members_count=len(verified),
        extracted_members_all_sha_verified=True, original_files_all_rehashed=True,
        original_tree_membership_rechecked=True, exact_recovery_sidecars_verified=True,
        numerical_qualified=False, performance_qualified=False, summary_sha256=file_info(out / 'summary.json')['sha256'])
    write_new(out / 'archive-verification.json', verification)
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restoration-audit', type=Path,
        default=ARTIFACTS / 'http-baseline-v1-restoration-audit.json')
    parser.add_argument('--out', type=Path, default=DESTINATION)
    parser.add_argument('--run', action='store_true', help='Create the new archive only after the recovery audit exists.')
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({'action': 'PREPARE_ONLY_NO_ARCHIVE', 'inputs': {k: str(v) for k, v in TREES.items()},
                          'restoration_audit': str(args.restoration_audit), 'out': str(args.out)}, indent=2))
        return 0
    print(json.dumps(archive(args.restoration_audit.resolve(), args.out.resolve()), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
