"""Archive the closed Lithos D512 formal-v1 cycle; no work without --run.

Run only after the root owner, recovery and independent audits have finished.
The default formal audit is analysis/independent-formal.json. A failed trial can
instead be archived with --restoration-audit analysis/restoration-audit.json;
that must be a real independent PASS, not an inferred recovery status. Missing
formal evidence stays NOT_RUN and cannot grant performance qualification.

This copies and verifies evidence bytes. It does not execute an auditor, touch
services, import inference runtimes, or read model/pack/runtime payloads. A failed
archive attempt leaves its new partial destination; no output is overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT = Path('/Users/ryanli/code/minivllm')
LITHOS = Path('/Users/ryanli/code/lithos-metal')
ART = ROOT / '.artifacts/m5-ultra-256gb'
NAME = 'lithos-d512-formal-v1'
PREREG = ROOT / 'bench/m5_ultra_256gb/lithos_d512_formal/prereg-formal-v1'
PLAN_SHA = 'ab8d035bf53f65855143ac9c5f2713d0a119bcd81a9504766a0aa8b376612523'
DEFAULT_OUT = ROOT / 'docs/experiments/m5-ultra-256gb/E023-lithos-formal-v1'
TREE_KEYS = ('owner', 'cold-journal', 'execution', 'analysis', 'cpu-prepare')
PHASES = ('r1-c1', 'r1-c8', 'r2-c8', 'r2-c1', 'r3-c1', 'r3-c8')
AUDIT_SHAS = {
    'audit_formal.py': '768dc970dca201a700c94e07e17c5fdd399c903f444272ea150eb6336575c9c2',
    'audit_resources.py': '398ff7a52dba307418f4c2725f33988b1571d65b11e3f46c9e97759f3a167c9e',
    'audit_restore.py': '41d260e63ce13c5c5912217ae74169e5c3cd118af32f3667aefffc3637a2d66f',
}
FORBIDDEN_SUFFIXES = ('.safetensors', '.pack', '.dylib', '.so', '.bin')
EXCLUDED_OFFICIAL_PAYLOADS = {
    str(LITHOS / 'monolith/runtime/_native.cpython-312-darwin.so'): dict(
        sha256='a1932813e5a9769edc2281b3070da7e84de6e02169ac5b3749e857492767e99f',
        bytes=221536,
        reason='Known native runtime extension is classified in the inherited lithos_source inventory; preserve its supplied identity and owner pre/post verification receipts without copying or rehashing the binary payload.'),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def info(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular input required: ' + str(path))
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    before = path.stat()
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        need(signature(os.fstat(stream.fileno())) == signature(before), 'input replaced before open: ' + str(path))
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
        need(signature(os.fstat(stream.fileno())) == signature(before), 'open input changed: ' + str(path))
    after = path.stat()
    need(signature(before) == signature(after) and size == after.st_size, 'input changed: ' + str(path))
    return dict(sha256=digest.hexdigest(), bytes=size, device=after.st_dev,
                inode=after.st_ino, mtime_ns=after.st_mtime_ns)


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())


def files(root):
    need(root.is_dir() and not root.is_symlink(), 'evidence directory absent/symlink: ' + str(root))
    result = []
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink() and (path.is_dir() or path.is_file()), 'link/special evidence file: ' + str(path))
        if path.is_file():
            need(not path.name.endswith(FORBIDDEN_SUFFIXES), 'unexpected payload in evidence tree; inspect, do not skip: ' + str(path))
            result.append(path)
    return result


def fixed_terminal(trees, args):
    owner, cold, execution = (trees['raw/' + key] for key in ('owner', 'cold-journal', 'execution'))
    plan, receipt = read(PREREG / 'owner-plan.json'), read(PREREG / 'receipt.json')
    need(info(PREREG / 'owner-plan.json')['sha256'] == info(owner / 'owner-plan.json')['sha256']
         == PLAN_SHA == receipt['owner_plan_sha256'], 'fixed registered plan')
    complete, attempt = read(owner / 'complete.json'), read(owner / 'attempt.json')
    need(complete['schema'] == 'E023-lithos-D512-formal-owner-result-v1'
         and complete['status'] in ('PASS', 'FAIL')
         and complete['owner_plan_sha256'] == attempt['owner_plan_sha256'] == PLAN_SHA
         and complete['owner_pid'] == attempt['owner_pid']
         and complete['planned_trials'] == list(PHASES) and complete['performance_qualified'] is False,
         'closed actual owner required; partial/live owner cannot be archived')
    journal, final = read(cold / 'journal.json'), read(execution / 'final.json')
    journal_sha, complete_sha = info(cold / 'journal.json')['sha256'], info(owner / 'complete.json')['sha256']
    need(journal['schema'] == 'E023-lithos-D512-formal-cold-journal-v1'
         and journal['state'] == final['journal_state'] == 'RESTORED'
         and type(final['recovery_returncode']) is int and final['recovery_returncode'] == 0
         and journal['owner_plan']['sha256'] == PLAN_SHA, 'completed restoration required')

    formal = read(args.formal_audit) if args.formal_audit.is_file() else None
    if formal is not None:
        need(formal['schema'] == 'E023-lithos-D512-formal-independent-audit-v1'
             and formal['status'] in ('PASS', 'FAIL', 'INCOMPLETE')
             and formal['audit_source_sha256'] == AUDIT_SHAS['audit_formal.py']
             and Path(formal['owner_directory']).resolve() == owner
             and formal['full_state_qualified'] is False and formal['output_quality_qualified'] is False,
             'independent formal receipt source/scope')
        if formal['status'] != 'PASS':
            need(formal['qualified'] is False and formal['performance_qualified'] is False, 'failed formal audit cannot qualify')
    if args.restoration_audit is not None:
        restore = read(args.restoration_audit)
        restore_origin = dict(path=str(args.restoration_audit), sha256=info(args.restoration_audit)['sha256'],
                              json_pointer='', sidecar='exact original bytes')
    else:
        need(formal is not None and isinstance(formal.get('restoration'), dict),
             'no nested restoration audit; supply a real --restoration-audit PASS receipt')
        restore = formal['restoration']
        restore_origin = dict(path=str(args.formal_audit), sha256=info(args.formal_audit)['sha256'],
                              json_pointer='/restoration', sidecar='verbatim JSON value, serialized as a separate sidecar')
    need(restore['schema'] == 'cold-service-restoration-audit-v1' and restore['status'] == 'PASS'
         and restore['qualified'] is True and restore['numerical_qualified'] is False
         and restore['performance_qualified'] is False
         and restore['audit_source_sha256'] == AUDIT_SHAS['audit_restore.py']
         and restore['journal_sha256'] == journal_sha and restore['owner_plan_sha256'] == PLAN_SHA
         and restore['controller_sha256'] == plan['cold_restart']['controller']['sha256']
         and restore['service_restore']['replacement'] == journal['replacement'], 'independent restoration receipt binding')

    qualified = formal is not None and formal['status'] == 'PASS'
    if qualified:
        resource = formal['resources']
        need(complete['status'] == 'PASS' and complete['all_children_reaped'] is True
             and complete['formal_trials_complete'] is True and formal['qualified'] is True
             and formal['performance_qualified'] is True and formal['formal_requests'] == 384
             and formal['warm_requests'] == 48 and formal['owner_plan_sha256'] == PLAN_SHA
             and formal['owner_complete_sha256'] == complete_sha and formal['journal_sha256'] == journal_sha
             and [p['phase'] for p in formal['points']] == list(PHASES)
             and [(p['concurrency'], p['repeat'], p['requests']) for p in formal['points']]
             == [(1, 1, 64), (8, 1, 64), (8, 2, 64), (1, 2, 64), (1, 3, 64), (8, 3, 64)],
             'complete independently qualified six-process formal scope')
        need(resource['schema'] == 'Lithos-D512-formal-resource-audit-v1' and resource['status'] == 'PASS'
             and resource['independent_resource_verified'] is True and resource['performance_qualified'] is False
             and resource['audit_source_sha256'] == AUDIT_SHAS['audit_resources.py']
             and resource['owner_plan_sha256'] == PLAN_SHA and resource['owner_complete_sha256'] == complete_sha
             and Path(resource['owner_directory']).resolve() == owner, 'nested independent resource audit binding')
        need(formal['restoration'] == restore, 'formal and separate restoration receipts differ')
    if complete['status'] == 'PASS':
        for phase in PHASES:
            trial = owner / 'phases' / phase / 'trial'
            for relative in ('warm-group.json', 'formal-group.json', 'warm/group.json', 'formal/group.json',
                             'warm-generation.log', 'formal-generation.log', 'server.stdout', 'server.stderr',
                             'warm-raw-check.json', 'formal-raw-check.json', 'warm-log-check.json',
                             'formal-log-check.json', 'log-final-check.json', 'complete.json', 'release.json',
                             'launch.json', 'spec.json', 'point.json'):
                need((trial / relative).is_file(), 'successful trial raw evidence missing: ' + str(trial / relative))
    pins = dict(plan['execution']['pins'])
    for path, digest in receipt['source_pins'].items():
        need(path not in pins or pins[path] == digest, 'conflicting registered source pin: ' + path)
        pins[path] = digest
    for name, digest in AUDIT_SHAS.items():
        path = PREREG.parent / name
        need(pins[str(path)] == digest == info(path)['sha256'], 'registered independent auditor source pin')
    children_path = owner / 'children.json'
    children = read(children_path)['children'] if children_path.is_file() else []
    summary = dict(schema='Lithos-D512-formal-archive-summary-v1', owner_status=complete['status'],
        owner_plan_sha256=PLAN_SHA, owner_pid=complete['owner_pid'], owner_complete_sha256=complete_sha,
        journal_sha256=journal_sha, restored_identity=journal['replacement'], restoration_status='PASS',
        restoration_provenance=restore_origin, formal_audit_status=formal['status'] if formal else 'NOT_RUN',
        observed_phases=[row['phase'] for row in children], planned_phases=list(PHASES),
        performance_qualified=qualified, full_state_qualified=False, output_quality_qualified=False,
        formal_requests_independently_verified=formal['formal_requests'] if qualified else None,
        warm_requests_independently_verified=formal['warm_requests'] if qualified else None,
        points=formal['points'] if qualified else [], medians=formal['medians'] if qualified else [],
        scope='Evidence preservation only. Performance eligibility is copied from the pinned independent formal audit, never inferred from packaging or recovery. C is HTTP outstanding callers, not GPU batch size; normal-EOS counts use the qualified official route. No full-state or output-quality equality claim.')
    return plan, pins, summary, restore


def archive(args):
    out = args.out.resolve()
    need(not out.exists() and out.parent.is_dir(), 'fresh destination in existing parent required')
    trees = {'raw/' + key: ART / (NAME + '-' + key) for key in TREE_KEYS}
    trees['preregistration'] = PREREG
    for path in (args.formal_audit, args.restoration_audit):
        if path is not None:
            need(path.resolve().parent == trees['raw/analysis'], 'audit must be a direct file in the fixed analysis directory')
    plan, pins, summary, restore = fixed_terminal(trees, args)
    inputs, snapshots = {}, {}
    for prefix, root in trees.items():
        paths = files(root)
        snapshots[prefix] = [p.relative_to(root).as_posix() for p in paths]
        inputs.update({prefix + '/' + p.relative_to(root).as_posix(): p for p in paths})
    for path, digest in pins.items():
        p = Path(path)
        base = ROOT if p.is_relative_to(ROOT) else LITHOS
        need(p.is_relative_to(base) and p.resolve() == p and not p.name.endswith(FORBIDDEN_SUFFIXES)
             and info(p)['sha256'] == digest, 'registered source changed/outside allowed repositories: ' + path)
        prefix = 'registered-source/minivllm/' if base == ROOT else 'registered-source/lithos-metal/'
        inputs[prefix + p.relative_to(base).as_posix()] = p
    excluded_payloads = []
    for path, row in plan['verification']['groups']['lithos_source'].items():
        p = Path(path)
        need(p.is_relative_to(LITHOS) and p.resolve() == p, 'official source outside repository: ' + path)
        if p.name.endswith(FORBIDDEN_SUFFIXES):
            expected = EXCLUDED_OFFICIAL_PAYLOADS.get(path)
            need(expected is not None and row['sha256'] == expected['sha256']
                 and type(row['bytes']) is int and row['bytes'] == expected['bytes'],
                 'unknown or changed excluded official payload: ' + path)
            excluded_payloads.append(dict(path=path, **expected, inherited_inventory=row,
                payload_copied=False, payload_rehashed_by_archive=False,
                verification_scope='Supplied exact inventory and retained owner pre/post verification receipts; no independent binary-content verification by this packer.'))
            continue
        need(info(p)['sha256'] == row['sha256'], 'official source changed: ' + path)
        inputs['official-source/' + p.relative_to(LITHOS).as_posix()] = p
    need({row['path'] for row in excluded_payloads} == set(EXCLUDED_OFFICIAL_PAYLOADS),
         'registered known native payload missing from inventory')
    for relative in ('LICENSE', 'third_party/MLX-LICENSE', 'third_party/NOTICE'):
        inputs['licenses/lithos-metal/' + relative] = LITHOS / relative
    pack = [(Path(p), r) for p, r in plan['verification']['groups']['pack'].items() if Path(p).name == 'manifest.json']
    need(len(pack) == 1 and info(pack[0][0])['sha256'] == pack[0][1]['sha256'], 'pack metadata pin')
    inputs['external-metadata/target-pack-manifest.json'] = pack[0][0]
    inputs['archive-source/archive_formal_v1.py'] = Path(__file__).resolve()
    summary['excluded_payloads'] = excluded_payloads
    summary['source_copy_counts'] = {prefix: sum(name.startswith(prefix + '/') for name in inputs)
        for prefix in ('registered-source/minivllm', 'registered-source/lithos-metal', 'official-source', 'archive-source')}
    out.mkdir()
    sidecars = {'restored-journal.json': trees['raw/cold-journal'] / 'journal.json'}
    if args.formal_audit.is_file():
        sidecars['independent-formal.json'] = args.formal_audit
    if args.restoration_audit is not None:
        sidecars['restoration-audit.json'] = args.restoration_audit
    else:
        write(out / 'restoration-audit.json', restore)
        inputs['restoration-audit.json'] = out / 'restoration-audit.json'
    for name, path in sidecars.items():
        with (out / name).open('xb') as stream:
            stream.write(path.read_bytes())
        need((out / name).read_bytes() == path.read_bytes(), 'exact sidecar copy')
        inputs[name] = out / name
    write(out / 'summary.json', summary)
    inputs['summary.json'] = out / 'summary.json'
    (out / 'ARCHIVE.md').write_text('''# Lithos D512 formal C1/C8 evidence

Read summary.json and independent-formal.json (if present) for the actual outcome.
Packaging PASS proves byte preservation, never inference or performance success.
A failed/unfinished formal audit remains unqualified even when recovery passed.
An independent restoration PASS is mandatory for this archive. If that receipt
was nested in the formal audit, restoration-audit.json is its exact JSON value
serialized separately; summary.json records the original file SHA and pointer.

All files in owner, cold-journal, execution, analysis, cpu-prepare and preregistration
are retained, including partial/failed requests. The planned six fresh observer-OFF
processes each issue eight warm and 64 formal requests: 48 warm and 384 formal.
Actual completed counts come only from the independent audit. No failed point is
replaced. Throughput is committed output tokens / first-POST-to-last-DONE window;
latency is each POST-to-DONE interval, including server queue but not client-slot
waiting. C8 is HTTP concurrency with official FIFO GPU execution, not GPU B8.
Process medians are not pooled request percentiles or confidence intervals.
The preserved qualification establishes official route/count/text behavior; it
does not establish full-state or output-quality equivalence across engines.

Registered old/new harness and auditor sources, pinned official Lithos sources,
their licenses/notices, and pack manifest metadata are retained. No model weights,
pack payload or runtime binaries are copied/rehashed. Those supplied identities
and pre/post checks remain in original verification and owner receipts.
The one known native extension classified under lithos_source is explicitly listed
in summary.json/manifest.json excluded_payloads with its fixed path, SHA and size;
all other unexpected forbidden payloads still reject the archive. Source-copy
counts are computed from the actual retained entries, not a preset total.

manifest.json maps every evidence.tar.xz member to original path, SHA256, size
and stat. archive-verification.json records complete extraction/member SHA checks,
original rehash and unchanged tree membership; both stay outside the tar to avoid
recursive hashes. The tar contains regular relative files only, with no absolute
paths, parent traversal or duplicate members. Verify SHA256 before extracting;
extract to a new directory. The retained packer also shows the safe-member checks.
No archive input or prior evidence is overwritten. Exact warm KV/runtime-state
restoration is not claimed; the independent restoration scope is recorded in its
receipt. Preserve the raw evidence even if future analysis changes a conclusion.
''')
    inputs['ARCHIVE.md'] = out / 'ARCHIVE.md'
    records = {name: dict(source=str(p), **info(p)) for name, p in sorted(inputs.items())}
    manifest = dict(schema='Lithos-D512-formal-evidence-manifest-v1', members=records,
        included_members=len(records), all_original_bytes=sum(r['bytes'] for r in records.values()),
        tree_members=snapshots, excluded_payloads=excluded_payloads,
        source_copy_counts=summary['source_copy_counts'],
        excluded_external_payloads='No model/pack weights or runtime binaries copied; original path/hash/size inventories remain in verification and owner pre/post receipts.')
    write(out / 'manifest.json', manifest)
    package = out / 'evidence.tar.xz'
    with package.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:xz', dereference=True, preset=6) as tar:
        for name, path in sorted(inputs.items()):
            tar.add(path, arcname=name, recursive=False)
    with tempfile.TemporaryDirectory(prefix='verify-', dir=out) as temp:
        seen = set()
        with tarfile.open(package, 'r:xz') as tar:
            for member in tar:
                name, relative = member.name, PurePosixPath(member.name)
                need(member.isfile() and not relative.is_absolute() and '..' not in relative.parts
                     and name not in seen and name in records, 'unsafe/duplicate/unexpected tar member')
                expected = records[name]
                need(member.size == expected['bytes'], 'tar member size')
                dest = Path(temp).joinpath(*relative.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, dest.open('xb') as output:
                    shutil.copyfileobj(source, output, 1 << 20)
                got = info(dest)
                need((got['sha256'], got['bytes']) == (expected['sha256'], expected['bytes']), 'extracted member SHA: ' + name)
                seen.add(name)
        need(seen == set(records), 'omitted tar member')
    for name, path in inputs.items():
        need(info(path) == {k: v for k, v in records[name].items() if k != 'source'}, 'original changed: ' + str(path))
    for prefix, root in trees.items():
        need([p.relative_to(root).as_posix() for p in files(root)] == snapshots[prefix], 'evidence tree changed')
    for name, path in sidecars.items():
        need((out / name).read_bytes() == path.read_bytes(), 'sidecar changed')
    need(read(out / 'restoration-audit.json') == restore, 'restoration sidecar value changed')
    package_info = info(package)
    result = dict(schema='Lithos-D512-formal-archive-verification-v1', status='PASS',
        archive_sha256=package_info['sha256'], archive_bytes=package_info['bytes'],
        manifest_sha256=info(out / 'manifest.json')['sha256'], included_members=len(records),
        all_archived_member_hashes_verified=True, all_original_files_rehashed=True,
        all_tree_members_rechecked=True, restoration_sidecar_value_verified=True,
        model_pack_payloads_rehashed=False, packaging_grants_performance_qualification=False,
        independent_formal_status=summary['formal_audit_status'],
        compressed_file_under_100_MiB=package_info['bytes'] < (100 << 20))
    write(out / 'archive-verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--formal-audit', type=Path, default=ART / (NAME + '-analysis') / 'independent-formal.json')
    parser.add_argument('--restoration-audit', type=Path,
                        help='Optional standalone independent restoration PASS, required if no usable nested restoration exists')
    args = parser.parse_args()
    args.formal_audit = args.formal_audit.resolve()
    if args.restoration_audit is not None:
        args.restoration_audit = args.restoration_audit.resolve()
    if not args.run:
        print(json.dumps(dict(action='PROPOSAL_ONLY_NO_INPUT_SCAN', trees={k: str(ART / (NAME + '-' + k)) for k in TREE_KEYS},
            preregistration=str(PREREG), out=str(args.out), formal_audit=str(args.formal_audit),
            restoration_audit=str(args.restoration_audit) if args.restoration_audit else 'nested /restoration in formal audit',
            prerequisite='All writers stopped; owner terminal, journal RESTORED, real independent restoration PASS. Run only after root closes measurement and recovery.'), indent=2))
        return 0
    print(json.dumps(archive(args), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
