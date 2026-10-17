"""CPU-only preparation of the fixed oMLX D512 qualification registration.

Default invocation describes the plan. --run writes only a fresh
prereg-qualification-v1 directory, never starts a process, imports a vendor
runtime, loads a model, contacts HTTP, or prepares/stops/restores a service.
Inherited payload SHA/stat inventories are checked against current metadata;
only missing explicit small official sources are newly hashed. The actual owner
must independently hash all four supplied groups before and after execution.
Root must retain the real controller PREPARED receipt and commit before stop.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREREG = HERE / 'prereg-qualification-v1'
ARTIFACTS = ROOT / '.artifacts/m5-ultra-256gb'
PATHS = {name:ARTIFACTS / ('omlx-d512-qualification-v1-' + name)
         for name in ('owner', 'cold-journal', 'execution', 'analysis', 'cpu-prepare')}
E020 = HERE.parent / 'external_spec_http/prereg-v1/plan.json'
E020_SHA = 'e3f0c3e24d4ea841d5e6c0b0094652e65e463c02367836e765ef6d9e3c6e2bdc'
NUMERIC = HERE.parent / 'pareto_realistic/prereg-qualification-v1/plan.json'
NUMERIC_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
OUTER_PLAN = HERE.parent / 'lithos_d512_formal/prereg-formal-v1/owner-plan.json'
OUTER_PLAN_SHA = 'ab8d035bf53f65855143ac9c5f2713d0a119bcd81a9504766a0aa8b376612523'
OWNER_SHA = '9c2df0b4e76bd2a6ef590c34bb7a6f9c89972c878f76f4b99f69d0b8e658019f'
CONTROLLER_SHA = 'a2fb20c043880faace5c6075190b115b073baad91acda9232f9468f3757790dc'
DEADLINES = dict(ready=180, request=600, group=720, overall=1200,
                 close=10, term=15, kill=15, resource_interval=5)
PORT = 18882
API_KEY = 'omlx-d512-local-benchmark-dummy'
SMALL_LIMIT = 2 << 20
INPUT_LIMIT = 32 << 20


def need(ok, message):
    if not ok: raise ValueError(message)


def stat(path):
    value = Path(path).stat()
    need(Path(path).is_file(), 'regular supplied file required: ' + str(path))
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def small_bytes(path, limit=SMALL_LIMIT):
    """Bounded stable read: this helper cannot accidentally hash a weight shard."""
    path = Path(path); before = path.stat()
    need(path.is_file() and before.st_size <= limit, 'small-file limit: ' + str(path))
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    final = path.stat()
    identity = lambda s:(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    need(len(raw) <= limit and all(identity(s) == identity(before) for s in (opened, after, final))
         and len(raw) == before.st_size, 'small source changed while reading: ' + str(path))
    return raw


def sha(path, limit=SMALL_LIMIT): return hashlib.sha256(small_bytes(path, limit)).hexdigest()
def ref(path, limit=SMALL_LIMIT): return dict(path=str(path), sha256=sha(path, limit))


def pinned(path, expected):
    path = Path(path)
    need(path.is_absolute(), 'absolute pinned input required')
    raw = small_bytes(path, INPUT_LIMIT)
    need(hashlib.sha256(raw).hexdigest() == expected, 'pinned input changed: ' + str(path))
    return json.loads(raw), raw


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def load(name, expected):
    path = HERE / name; need(sha(path) == expected, 'reviewed source changed: ' + name)
    spec = importlib.util.spec_from_file_location('_omlx_registration_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def inventory(owner, old, outer, model, specs):
    """Use real inherited hashes/stats, with no refreshed large-file hash claim."""
    groups = {name:{} for name in owner.GROUPS}; evidence = []

    def put(group, path, expected, inherited=None, origin=None):
        path = str(path); current = stat(path)
        if inherited is not None:
            need(inherited['sha256'] == expected and current == {k:inherited[k] for k in ('bytes','mtime_ns')},
                 'inherited SHA/stat no longer applies: ' + path)
            mode = 'inherited-payload-sha-current-stat-only'
        else:
            need(sha(path) == expected, 'new small official source differs: ' + path)
            need(stat(path) == current, 'new source metadata changed: ' + path)
            mode = 'new-small-source-payload-rehashed'
        row = dict(sha256=expected, **current)
        need(path not in groups[group] or groups[group][path] == row, 'conflicting inventory row')
        groups[group][path] = row
        evidence.append(dict(group=group, path=path, resolved=str(Path(path).resolve()),
            origin=origin, verification=mode, **row))

    def from_e020(group, path, expected):
        need(old['pins'].get(path) == expected and path in old['pin_stats'], 'missing E020 payload identity: ' + path)
        put(group, path, expected, dict(sha256=expected, **old['pin_stats'][path]), str(E020))

    prefix = str(owner.OMLX_VENV) + '/'
    for path, digest in old['pins'].items():
        if path.startswith(prefix): from_e020('omlx_runtime', path, digest)
    python_binary = str((owner.OMLX_VENV / 'bin/python').resolve())
    need(python_binary in old['pins'], 'resolved oMLX Python is not in E020')
    if python_binary not in groups['omlx_runtime']:
        from_e020('omlx_runtime', python_binary, old['pins'][python_binary])
    official = {path:digest for spec in specs for path,digest in spec['sources'].items()
                if Path(path).is_relative_to(owner.OMLX_SITE)}
    need(bool(official), 'actual observed official sources missing')
    for path, digest in official.items():
        inherited = None
        if path in old['pins']:
            need(old['pins'][path] == digest and path in old['pin_stats'], 'E020 official source changed')
            inherited = dict(sha256=digest, **old['pin_stats'][path])
        for group in ('omlx_source', 'omlx_runtime'):
            if path not in groups[group]:
                put(group, path, digest, inherited, str(E020) if inherited else 'actual trial.sources')
            else: need(groups[group][path]['sha256'] == digest, 'official/runtime SHA conflict')
    for relative, digest in model['files'].items():
        from_e020('target_model', str(Path(model['model']) / relative), digest)
    verification = outer['verification']
    need(Path(sys.executable) == Path(verification['python']) and
         Path(sys.executable).resolve() == Path(verification['python']).resolve(),
         'use the previously inventoried outer Python, not the oMLX server interpreter')
    for path, row in verification['groups']['outer_runtime'].items():
        put('outer_runtime', path, row['sha256'], row, str(OUTER_PLAN))
    result = dict(python=str(Path(sys.executable).absolute()), groups=groups)
    return result, evidence


def prepare(args):
    need(not PREREG.exists() and all(not p.exists() for p in PATHS.values()), 'all registration/output paths must be new')
    PREREG.mkdir()
    try:
        old, old_raw = pinned(E020, E020_SHA)
        numeric, numeric_raw = pinned(NUMERIC, NUMERIC_SHA)
        outer, outer_raw = pinned(OUTER_PLAN, OUTER_PLAN_SHA)
        model, model_raw = pinned(**dict(path=numeric['model_manifest']['path'], expected=numeric['model_manifest']['sha256']))
        predecessor = dict(path=str(args.predecessor), sha256=args.predecessor_sha256,
            audit=dict(path=str(args.restoration_audit), sha256=args.restoration_audit_sha256))
        prior, prior_raw = pinned(args.predecessor, args.predecessor_sha256)
        _, audit_raw = pinned(args.restoration_audit, args.restoration_audit_sha256)
        owner = load('owner.py', OWNER_SHA); controller = load('controller.py', CONTROLLER_SHA)
        accepted, _ = controller.predecessor(predecessor)
        need(accepted == prior, 'archived predecessor must satisfy the actual controller')
        trial = owner.load('trial'); probe = trial.load('serve_probe_v2.py')
        minimal = probe.specification(PATHS['owner'] / 'phases/omlx-c1/trial', 1, PORT, API_KEY, observed=True)
        # A clean explicit environment; no ambient MLX/model flags are inherited.
        environment = {k:os.environ[k] for k in ('PATH','HOME','TMPDIR','LANG','LC_ALL','USER','LOGNAME') if k in os.environ}
        environment.update(minimal['environment'])
        specs = [trial.prepare(PATHS['owner'] / 'phases' / phase / 'trial', C, PORT, API_KEY,
                    deadlines=DEADLINES, environment=environment) for phase,C in zip(owner.PHASES,(1,8))]
        verification, provenance = inventory(owner, old, outer, model, specs)
        owner.verification_contract(verification, numeric, specs)
        plan = owner.plan_template(qualification=dict(path=str(NUMERIC),sha256=NUMERIC_SHA),
            cold_restart=dict(journal_path=str(PATHS['cold-journal']),controller=ref(HERE/'controller.py'),
                approved_contract=controller.CONTRACT,proposal_plan_sha256=controller.PLAN_SHA,predecessor=predecessor),
            http=dict(port=PORT,api_key=API_KEY,deadlines=dict(DEADLINES),environment=environment), verification=verification)
        write(PREREG/'owner-plan.json', plan); plan_ref = ref(PREREG/'owner-plan.json', INPUT_LIMIT)
        _, checked, _, _, checked_specs = owner.validate_plan(SimpleNamespace(
            plan=PREREG/'owner-plan.json', plan_sha256=plan_ref['sha256'], out=PATHS['owner']))
        need(checked == plan and checked_specs == specs and [len(s['jobs']) for s in specs] == [8,16],
             'real owner validation and exact 8/16 request specs')
        python = verification['python']; common = ['--journal',str(PATHS['cold-journal']),'--plan-sha256',controller.PLAN_SHA]
        commands = dict(
            cpu_prepare=[python,'-I','-B',str(HERE/'owner.py'),'--plan',plan_ref['path'],'--plan-sha256',plan_ref['sha256'],
                '--out',str(PATHS['cpu-prepare']),'--prepare-only'],
            readonly_service_prepare=[python,'-I','-B',str(HERE/'controller.py'),'prepare',*common,
                '--owner-directory',str(PATHS['owner']),'--owner-plan',plan_ref['path'],'--owner-plan-sha256',plan_ref['sha256']],
            stop=[python,'-I','-B',str(HERE/'controller.py'),'stop',*common,'--approved-contract',controller.CONTRACT],
            owner=[python,'-I','-B',str(HERE/'owner.py'),'--plan',plan_ref['path'],'--plan-sha256',plan_ref['sha256'],'--out',str(PATHS['owner'])],
            execute_after_receipt_and_commit=[python,'-I','-B',str(HERE/'execute.py'),'--run'],
            recover=[python,'-I','-B',str(HERE/'controller.py'),'recover',*common,'--approved-contract',controller.CONTRACT],
            recover_no_experiment_only_if_owner_never_launched=[python,'-I','-B',str(HERE/'controller.py'),'recover',*common,
                '--approved-contract',controller.CONTRACT,'--no-experiment'],
            create_analysis_directory=[python,'-I','-B','-c','from pathlib import Path; Path('+repr(str(PATHS['analysis']))+').mkdir()'],
            restoration_audit=[python,'-I','-B',str(HERE/'audit_restore.py'),'--journal',str(PATHS['cold-journal']),
                '--out',str(PATHS['analysis']/'restoration-audit.json')],
            resource_audit=[python,'-I','-B',str(HERE/'audit_resources.py'),'--owner',str(PATHS['owner']),
                '--out',str(PATHS['analysis']/'resource-summary.json')],
            qualification_audit=[python,'-I','-B',str(HERE/'audit_qualification.py'),'--owner',str(PATHS['owner']),
                '--out',str(PATHS['analysis']/'independent-qualification.json')])
        write(PREREG/'commands.json', commands)
        write(PREREG/'trial-specs.json', specs)
        write(PREREG/'verification-provenance.json', dict(rows=provenance,
            inherited_payloads_rehashed=False, recursive_runtime_coverage_verified=False,
            scope='Actual current stat against inherited exact hashes; new small official sources explicitly identified. Full supplied payload rehash is mandatory in owner preflight/postflight.'))
        inputs = PREREG/'inputs'; inputs.mkdir()
        for name, raw in [('e020-plan.json',old_raw),('numeric-plan.json',numeric_raw),('outer-plan.json',outer_raw),
                          ('model-manifest.json',model_raw),('predecessor.json',prior_raw),('restoration-audit.json',audit_raw)]:
            (inputs/name).write_bytes(raw)
        pins = controller.source_pins(plan)
        for spec in specs:
            for path,digest in spec['sources'].items():
                need(path not in pins or pins[path] == digest, 'conflicting source pin: ' + path); pins[path] = digest
        # Additional offline audit support sources are metadata/source only.
        for path in [Path(__file__), HERE/'execute.py', *sorted(HERE.glob('test_*.py')),
            HERE.parent/'omlx_cold_restart/audit_restore_v4.py',
            HERE.parent/'pareto_realistic/audit_absent_qualification.py',
            HERE.parent/'pareto_realistic/audit_native_formal.py']:
            pins[str(path)] = sha(path)
        for path in (PREREG/'owner-plan.json',PREREG/'commands.json',PREREG/'trial-specs.json'):
            pins[str(path)] = sha(path, INPUT_LIMIT)
        for path,digest in pins.items():
            raw = small_bytes(path, INPUT_LIMIT if Path(path).parent == PREREG else SMALL_LIMIT)
            need(hashlib.sha256(raw).hexdigest() == digest, 'source changed during preparation: ' + path)
            dest = PREREG/'frozen-source'/path.lstrip('/'); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(raw)
        # Recheck every inherited file stat after the small-file work; still no payload-read claim.
        for group in verification['groups'].values():
            for path,row in group.items(): need(stat(path) == {k:row[k] for k in ('bytes','mtime_ns')}, 'inventory changed: ' + path)
        write(PREREG/'source-manifest.json', dict(schema='oMLX-D512-prereg-source-manifest-v1',source_pins=pins,
            owner_plan=plan_ref,predecessor=predecessor,source_count=len(pins),
            runtime_model_payloads_rehashed=False,recursive_runtime_coverage_verified=False))
        result = dict(schema='oMLX-D512-prereg-preparation-v1',status='CPU_PREPARED',owner_plan=plan_ref,
            source_manifest=ref(PREREG/'source-manifest.json'),paths={k:str(v) for k,v in PATHS.items()},
            predecessor=predecessor,original_identity=prior['replacement'],trials=plan['trials'],requests=[8,16],
            deadlines=DEADLINES,owner_work_seconds=plan['deadline_seconds'],postflight_seconds=owner.POSTFLIGHT_SECONDS,
            group_counts={k:len(v) for k,v in verification['groups'].items()},
            bytes_by_group={k:sum(r['bytes'] for r in v.values()) for k,v in verification['groups'].items()},
            inherited_payloads_rehashed=False,service_action=False,prepared_journal_created=False,
            performance_qualified=False,full_state_qualified=False,
            remaining='Root must run read-only controller prepare, save its real PREPARED journal SHA and execution receipt, review/commit, then use the separately reviewed execute.py --run. This builder only records its command/source; no driver or service evidence is executed or fabricated.')
        write(PREREG/'preparation.json', result)
        return result
    except BaseException as error:
        write(PREREG/'preparation-failure.json',dict(status='FAIL',type=type(error).__name__,message=str(error),service_action=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--predecessor', type=Path)
    parser.add_argument('--predecessor-sha256')
    parser.add_argument('--restoration-audit', type=Path)
    parser.add_argument('--restoration-audit-sha256')
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(dict(schema='oMLX-D512-prereg-description-v1',writes=False,prereg=str(PREREG),
            paths={k:str(v) for k,v in PATHS.items()},port=PORT,api_key=API_KEY,deadlines=DEADLINES,
            trials=['omlx-c1','omlx-c8'],requests=[8,16],outer_python=str(ROOT/'.venv/bin/python'),
            required_run_arguments=['--predecessor ARCHIVED_JSON','--predecessor-sha256 SHA',
                '--restoration-audit ARCHIVED_JSON','--restoration-audit-sha256 SHA']),indent=2)); return 0
    need(all((args.predecessor,args.predecessor_sha256,args.restoration_audit,args.restoration_audit_sha256)),
         '--run requires explicit immutable predecessor and restoration-audit paths plus both SHA256 values')
    print(json.dumps(prepare(args),sort_keys=True));return 0


if __name__ == '__main__': raise SystemExit(main())
