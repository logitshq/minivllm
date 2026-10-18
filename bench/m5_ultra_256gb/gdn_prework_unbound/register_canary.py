"""Prepare one fixed GDN real-activation canary after the archived route HTTP cycle.

Default: describe only. --prepare performs the original inherited CPU preflight
(including its real runtime/model reads), but no service or device operation.
Root must first seal cold_cycle_v3's four migration hashes and regenerate its
manifest from the actual archived RESTORED journal and independent restoration
PASS. This script never seals a guessed predecessor or edits lifecycle sources.
After root separately runs the emitted read-only controller prepare command,
--seal-receipt binds that actual PREPARED journal. All outputs are exclusive.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
import traceback
from types import SimpleNamespace

R = Path('/Users/ryanli/code/minivllm')
B = R / 'bench/m5_ultra_256gb'
HERE = B / 'gdn_prework_unbound'
P = HERE / 'prereg-canary-v1'
A = R / '.artifacts/m5-ultra-256gb'
NAME = 'gdn-prework-canary-v1'
PHASE = 'gdn-prework-real-layer'
L = B / 'cold_cycle_v3'
PRIOR = R / 'docs/experiments/m5-ultra-256gb/E024-route-http-v1'
PY = str(R / '.venv/bin/python')
FIXED = {
    'gdn_prework_unbound/qualify.py': 'f3088a98d2aca6b4f267e7d4c258efbbf9f83e1beccb6745c88ae42343f88e89',
    'gdn_prework_unbound/candidate.py': '1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260',
    'long_decode_isolation/qualify.py': '9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e',
    'long_decode_graph/qualify.py': 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
    'long_decode_graph/model-source-pins.json': '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
    'pareto_realistic/qualify_native.py': '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    'cold_cycle_v3/owner.py': '4c2cc6f305de3ecf0c6495e45838baae885efbf7a8f479ed2043dc0ffb67f31c',
    'cold_cycle_v3/audit_restore.py': '34d608b54ed5779b97e85a0463c17f8796876be731f6f0aafcbe804ffc7b15ca',
}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def bindings():
    for relative, digest in FIXED.items():
        need(sha(B / relative) == digest, 'fixed source changed: ' + relative)
    owner = load(L / 'owner.py', '_gdn_registered_single_child_owner')
    c = owner.controller()
    need(all(re.fullmatch('[0-9a-f]{64}', value) for value in
         (c.MIGRATION_CONTROLLER_SHA, c.MIGRATION_AUDITOR_SHA,
          c.MIGRATION_JOURNAL_SHA, c.MIGRATION_AUDIT_SHA)),
         'actual route HTTP migration must be sealed first')
    lifecycle = ref(L / 'lifecycle.json'); c.lifecycle(lifecycle)
    predecessor = dict(**ref(PRIOR / 'restored-journal.json'),
                       audit=ref(PRIOR / 'restoration-audit.json'))
    prior, _ = c.predecessor(predecessor, lifecycle)
    archive = read(PRIOR / 'archive-verification.json')
    need(archive['schema'] == 'E024-route-http-archive-verification-v1'
         and archive['status'] == 'PASS', 'actual predecessor archive must be complete')
    # c.predecessor requires restoration PASS, deliberately not numerical PASS.
    need(prior['state'] == 'RESTORED', 'actual predecessor must be restored')
    return owner, c, lifecycle, predecessor


def prepare():
    owner, c, lifecycle, predecessor = bindings()
    for suffix in ('owner', 'cold-journal', 'execution', 'cpu-prepare'):
        need(not (A / (NAME + '-' + suffix)).exists(), 'fresh artifact required: ' + suffix)
    for name in ('child-plan.json', 'owner-plan.json', 'preparation.json', 'commands.json', 'receipt.json'):
        need(not (P / name).exists(), 'registration already exists: ' + name)
    old_path = B / 'long_decode_graph/prereg-v1/owner-plan.json'
    need(sha(old_path) == owner.OWNER_PLAN_SHA, 'original 4dff preparation plan changed')
    old = read(old_path)
    qualifier = load(HERE / 'qualify.py', '_gdn_registered_real_activation_qualifier')
    child = dict(schema='GDN-prework-real-layer-plan-v1',
        qualifier_sha256=FIXED['gdn_prework_unbound/qualify.py'],
        candidate=ref(HERE / 'candidate.py'), e024_plan=old['qualification_plan'], deadline_s=1800)
    qualifier.validate(child)
    P.mkdir(exist_ok=True)
    write(P / 'child-plan.json', child)
    dependencies = {str(B / name): digest for name, digest in FIXED.items()
                    if not name.startswith('cold_cycle_v3/')}
    # The real candidate verifies seven current native dependencies. Keep the
    # entire unchanged core manifest explicit, as well as inherited preflight.
    core = read(B / 'long_decode_graph/model-source-pins.json')
    for relative, digest in core['files'].items():
        path = R / 'tutorial_code/miniengine' / relative
        need(sha(path) == digest, 'current native core changed: ' + relative)
        dependencies[str(path)] = digest
    plan = dict(schema=c.RUN_SCHEMA, lifecycle=lifecycle, predecessor=predecessor,
        experiment=dict(phase=PHASE, qualifier=ref(HERE / 'qualify.py'),
                        plan=ref(P / 'child-plan.json'), source_pins=dependencies),
        inherited_owner_plan=ref(old_path),
        outputs=dict(journal=str(A / (NAME + '-cold-journal')), out=str(A / (NAME + '-owner'))),
        budgets=dict(work_seconds=1830, child_seconds=1800, term_seconds=15, kill_seconds=15))
    write(P / 'owner-plan.json', plan); plan_sha = sha(P / 'owner-plan.json')
    owner.validate_plan(SimpleNamespace(plan=P / 'owner-plan.json', plan_sha256=plan_sha,
                                        out=Path(plan['outputs']['out'])))
    cpu = A / (NAME + '-cpu-prepare'); cpu.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(schema='GDN-prework-canary-cpu-prepare-v1', status='INCOMPLETE',
                  owner_plan_sha256=plan_sha, gpu_executed=False, service_changed=False)
    def check():
        if time.monotonic() - started > 120: raise TimeoutError('CPU preparation exceeded120 seconds')
    try:
        command, env = owner.prepare(plan, cpu / 'prepared', check); check()
        need(command[1:] == ['-I', '-B', str(HERE / 'qualify.py'), '--plan', str(P / 'child-plan.json'),
             '--plan-sha256', sha(P / 'child-plan.json'), '--out', str(cpu / 'prepared/qualification')],
             'actual child CLI must retain exactly the standard three arguments; no --run')
        result.update(status='PASS', command=command,
                      mlx_environment={k: v for k, v in env.items() if k.startswith('MLX_')})
    except BaseException as error:
        result.update(status='FAIL', error=dict(type=type(error).__name__, message=str(error),
                                               traceback=traceback.format_exc()))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        write(cpu / 'result.json', result)
    pins = owner.source_pins(plan)
    extra = [Path(__file__).resolve(), HERE / 'execute_canary.py', HERE / 'test_execute_canary_cpu.py',
             HERE / 'audit.py', HERE / 'test_audit_cpu.py', HERE / 'canary-plan.md',
             L / 'test_cpu.py', L / 'test_predecessor_cpu.py', P / 'owner-plan.json',
             PRIOR / 'restored-journal.json', PRIOR / 'restoration-audit.json',
             PRIOR / 'archive-verification.json']
    for path in extra: pins[str(path)] = sha(path)
    target = P / 'frozen-source'; target.mkdir(exist_ok=False)
    for name, digest in pins.items():
        raw = Path(name).read_bytes(); need(hashlib.sha256(raw).hexdigest() == digest, 'source changed while freezing')
        dest = target / name.lstrip('/'); dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream: stream.write(raw)
    write(P / 'preparation.json', dict(schema='GDN-prework-canary-prereg-preparation-v1',
        status='CPU_PREPARED', owner_plan_sha256=plan_sha, child_plan_sha256=sha(P / 'child-plan.json'),
        lifecycle=lifecycle, predecessor=predecessor, cpu_prepare=ref(cpu / 'result.json'), source_pins=pins,
        contract=qualifier.fixed_contract(), expected_pairs=54, expected_raw_files=60,
        service_prepared=False, numerical_qualified=False, full_model_qualified=False,
        http_qualified=False, performance_qualified=False))
    journal = plan['outputs']['journal']; output = plan['outputs']['out']
    write(P / 'commands.json', dict(readonly_service_prepare=[PY, '-I', '-B', str(L / 'controller.py'),
        'prepare', '--journal', journal, '--plan-sha256', c.PLAN_SHA,
        '--owner-directory', output, '--owner-plan', str(P / 'owner-plan.json'), '--owner-plan-sha256', plan_sha],
        seal_receipt=[PY, '-I', '-B', str(Path(__file__).resolve()), '--seal-receipt'],
        execute=[PY, '-I', '-B', str(HERE / 'execute_canary.py'), '--run'],
        independent_numeric_audit=[PY, '-I', '-B', str(HERE / 'audit.py'), '--child',
          str(Path(output) / 'phases' / PHASE / 'qualification'), '--plan-sha256', sha(P / 'child-plan.json'),
          '--out', str(A / (NAME + '-analysis') / 'independent-numerical.json')]))
    print(json.dumps(dict(status='CPU_PREPARED', owner_plan_sha256=plan_sha,
                         child_plan_sha256=sha(P / 'child-plan.json'), service_prepared=False)))


def seal_receipt():
    """Bind root's actual read-only PREPARED journal; never create it here."""
    owner, c, lifecycle, predecessor = bindings()
    preparation = read(P / 'preparation.json')
    need(preparation['status'] == 'CPU_PREPARED' and preparation['lifecycle'] == lifecycle
         and preparation['predecessor'] == predecessor, 'actual CPU preparation binding')
    pins = preparation['source_pins']
    for path, digest in pins.items(): need(sha(path) == digest, 'prepared source changed: ' + path)
    plan_sha = preparation['owner_plan_sha256']; plan = read(P / 'owner-plan.json')
    need(sha(P / 'owner-plan.json') == plan_sha, 'prepared owner plan changed')
    owner.validate_plan(SimpleNamespace(plan=P / 'owner-plan.json', plan_sha256=plan_sha,
                                        out=Path(plan['outputs']['out'])))
    journal_path = Path(plan['outputs']['journal']) / 'journal.json'; journal = read(journal_path)
    prior, _ = c.predecessor(predecessor, lifecycle)
    need(journal['schema'] == c.JOURNAL_SCHEMA and journal['state'] == 'PREPARED'
         and journal['owner_plan'] == dict(path=str(P / 'owner-plan.json'), sha256=plan_sha)
         and journal['owner_directory'] == plan['outputs']['out'] and journal['lifecycle'] == lifecycle
         and journal['predecessor'] == predecessor and journal['original'] == prior['replacement']
         and journal['controller_sha256'] == sha(L / 'controller.py') and journal['trials'] == [PHASE]
         and journal['plan_sha256'] == c.PLAN_SHA, 'actual read-only PREPARED journal binding')
    need(not Path(plan['outputs']['out']).exists() and not (A / (NAME + '-execution')).exists(),
         'fresh owner/execution required')
    pins = dict(pins, **{str(P / 'preparation.json'): sha(P / 'preparation.json'),
                         str(P / 'commands.json'): sha(P / 'commands.json')})
    write(P / 'receipt.json', dict(schema='GDN-prework-canary-prereg-receipt-v1', status='PREPARED',
        owner_plan_sha256=plan_sha, child_plan_sha256=preparation['child_plan_sha256'],
        prepared_journal_sha256=sha(journal_path), source_pins=pins,
        numerical_qualified=False, full_model_qualified=False, http_qualified=False, performance_qualified=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(); mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--seal-receipt', action='store_true'); args = p.parse_args()
    if args.prepare: prepare()
    elif args.seal_receipt: seal_receipt()
    else: print('DESCRIPTION_ONLY: actual archived route HTTP restoration and sealed v3 lifecycle required; no operations.')


if __name__ == '__main__': main()
