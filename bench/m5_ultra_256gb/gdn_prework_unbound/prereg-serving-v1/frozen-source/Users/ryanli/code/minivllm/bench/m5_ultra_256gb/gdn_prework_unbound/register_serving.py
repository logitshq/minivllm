"""Register the original 37-request GDN numerical workload after a real canary.

Default only describes scope. --prepare requires the immutable archived canary
RESTORED journal, independent restoration/resource/raw-numerical PASS, and the
reviewed serving auditor source. It performs the unchanged inherited CPU
preflight (including real runtime/model reads), never device/service actions.
Root separately runs the emitted read-only controller prepare command, commits
the reviewable registration, then invokes the explicit --run execution driver.
--seal-receipt binds that actual PREPARED journal; all outputs are exclusive.
The stable v3 lifecycle is reused unchanged, including unknown-launch refusal.
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
P = HERE / 'prereg-serving-v1'
A = R / '.artifacts/m5-ultra-256gb'
NAME = 'gdn-prework-serving-v1'
PHASE = 'gdn-prework-serving'
L = B / 'cold_cycle_v3'
PRIOR = R / 'docs/experiments/m5-ultra-256gb/E029-gdn-prework-canary-v1'
CANARY_OWNER = A / 'gdn-prework-canary-v1-owner'
CANARY_CHILD = CANARY_OWNER / 'phases/gdn-prework-real-layer/qualification'
PY = str(R / '.venv/bin/python')
CANARY_OWNER_PLAN_SHA = 'a54e3e74bc3d1c838894a60f4696f142d82156f31a7a18d501805e078562f4dd'
CANARY_CHILD_PLAN_SHA = '17e3e23e6d4fd1e2430a176bc0830e17984eca9ddea0c20c61778bc7f60acfe3'
FIXED = {
    'gdn_prework_unbound/serving.py': '3f493752d95996bd08cc1ea7389fc3159ab615e1ac162b36332eecc47dab862c',
    'gdn_prework_unbound/candidate.py': '1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260',
    'gdn_prework_unbound/qualify.py': 'f3088a98d2aca6b4f267e7d4c258efbbf9f83e1beccb6745c88ae42343f88e89',
    'gdn_prework_unbound/audit.py': 'a3e97dc368da47735d7c8e9fd2bcb40d7decaccb9c5926084d53cffb10495d3d',
    'gdn_prework_unbound/audit_serving.py': '268cc336417a3d31dffc30c749701d132da610fbb41d9b0f4c85ad6adaa2f2bc',
    'long_decode_graph/model-source-pins.json': '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
    'pareto_realistic/qualify_native.py': '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    'pareto_realistic/audit_qualification.py': 'b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1',
    'resource_reparse.py': '5948ccbf7baeb21d1e71d9e8f7169b1bd9f84e65eb492746aba86fe377806c23',
    'cold_cycle_v3/owner.py': '4c2cc6f305de3ecf0c6495e45838baae885efbf7a8f479ed2043dc0ffb67f31c',
    'cold_cycle_v3/controller.py': '674b21fbcafa0e79164dedb78875f513e9a24d478337c2aeeb67a439e69a4dc9',
    'cold_cycle_v3/audit_restore.py': '34d608b54ed5779b97e85a0463c17f8796876be731f6f0aafcbe804ffc7b15ca',
    'cold_cycle_v3/lifecycle.json': 'c7a6afdcc7807a2b1e5629df628c716ff77278fe3d0bf8562715b8794c4e8dca',
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
    need(all(re.fullmatch('[0-9a-f]{64}', value) for value in FIXED.values()),
         'independent serving auditor and its helpers must be reviewed and sealed first')
    for relative, digest in FIXED.items():
        need(sha(B / relative) == digest, 'fixed source changed: ' + relative)
    owner = load(L / 'owner.py', '_gdn_serving_single_child_owner')
    c = owner.controller()
    lifecycle = ref(L / 'lifecycle.json'); c.lifecycle(lifecycle)
    predecessor = dict(**ref(PRIOR / 'restored-journal.json'),
                       audit=ref(PRIOR / 'restoration-audit.json'))
    prior, _ = c.predecessor(predecessor, lifecycle)
    archive, summary = (read(PRIOR / name) for name in ('archive-verification.json', 'summary.json'))
    need(archive['schema'] == 'E029-gdn-canary-archive-verification-v1'
         and archive['status'] == 'PASS', 'actual canary archive must be complete')
    need(summary['schema'] == 'E029-gdn-canary-archive-summary-v1'
         and summary['owner_plan_sha256'] == CANARY_OWNER_PLAN_SHA
         and summary['child_plan_sha256'] == CANARY_CHILD_PLAN_SHA
         and summary['owner_status'] == summary['child_status'] == 'PASS'
         and all(summary[key] is True for key in ('canary_qualified', 'resources_verified', 'restoration_verified')),
         'canary archive retention alone is not numerical/resource qualification')
    need(prior['owner_plan']['sha256'] == CANARY_OWNER_PLAN_SHA
         and prior['owner_directory'] == str(CANARY_OWNER)
         and prior['trials'] == ['gdn-prework-real-layer'], 'same actual canary restoration')
    resource = read(PRIOR / 'resource-summary.json')
    need(resource['schema'] == 'cold-single-child-resource-reparse-v1' and resource['status'] == 'PASS'
         and resource['source_sha256'] == FIXED['resource_reparse.py']
         and resource['owner_directory'] == str(CANARY_OWNER), 'actual canary resource PASS required')
    qualifier = load(HERE / 'serving.py', '_gdn_registered_serving_qualifier')
    canary = dict(audit=ref(PRIOR / 'independent-numerical.json'),
                  plan=ref(CANARY_CHILD / 'plan.json'), complete=ref(CANARY_CHILD / 'complete.json'))
    need(canary['plan']['sha256'] == CANARY_CHILD_PLAN_SHA, 'same actual canary child plan')
    proof = qualifier.bind_canary(canary)
    return owner, c, lifecycle, predecessor, qualifier, canary, proof


def prepare():
    owner, c, lifecycle, predecessor, qualifier, canary, proof = bindings()
    for suffix in ('owner', 'cold-journal', 'execution', 'cpu-prepare', 'analysis'):
        need(not (A / (NAME + '-' + suffix)).exists(), 'fresh artifact required: ' + suffix)
    for name in ('child-plan.json', 'owner-plan.json', 'preparation.json', 'commands.json', 'receipt.json'):
        need(not (P / name).exists(), 'registration already exists: ' + name)
    old_path = B / 'long_decode_graph/prereg-v1/owner-plan.json'
    need(sha(old_path) == owner.OWNER_PLAN_SHA, 'original 4dff preparation plan changed')
    old = read(old_path)
    child = dict(schema='GDN-prework-serving-plan-v1',
        qualifier_sha256=FIXED['gdn_prework_unbound/serving.py'],
        qualification=old['inherited_plan'], canary=canary, sources=qualifier.source_pins(), deadline_s=1800)
    P.mkdir(exist_ok=True); write(P / 'child-plan.json', child)
    qualifier.validate(P / 'child-plan.json', sha(P / 'child-plan.json'))
    dependencies = {str(B / name): digest for name, digest in FIXED.items()
                    if not name.startswith('cold_cycle_v3/')}
    # Current core source closure is explicit; original4dff still performs its
    # unchanged actual runtime/model preflight. No model payload is copied here.
    core = read(B / 'long_decode_graph/model-source-pins.json')
    for relative, digest in core['files'].items():
        path = R / 'tutorial_code/miniengine' / relative
        need(sha(path) == digest, 'current native core changed: ' + relative)
        dependencies[str(path)] = digest
    plan = dict(schema=c.RUN_SCHEMA, lifecycle=lifecycle, predecessor=predecessor,
        experiment=dict(phase=PHASE, qualifier=ref(HERE / 'serving.py'),
                        plan=ref(P / 'child-plan.json'), source_pins=dependencies),
        inherited_owner_plan=ref(old_path),
        outputs=dict(journal=str(A / (NAME + '-cold-journal')), out=str(A / (NAME + '-owner'))),
        budgets=dict(work_seconds=1830, child_seconds=1800, term_seconds=15, kill_seconds=15))
    write(P / 'owner-plan.json', plan); plan_sha = sha(P / 'owner-plan.json')
    owner.validate_plan(SimpleNamespace(plan=P / 'owner-plan.json', plan_sha256=plan_sha,
                                        out=Path(plan['outputs']['out'])))
    cpu = A / (NAME + '-cpu-prepare'); cpu.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(schema='GDN-prework-serving-cpu-prepare-v1', status='INCOMPLETE',
                  owner_plan_sha256=plan_sha, gpu_executed=False, service_changed=False)
    def check():
        if time.monotonic() - started > 120: raise TimeoutError('CPU preparation exceeded 120 seconds')
    try:
        command, env = owner.prepare(plan, cpu / 'prepared', check); check()
        need(command[1:] == ['-I', '-B', str(HERE / 'serving.py'), '--plan', str(P / 'child-plan.json'),
             '--plan-sha256', sha(P / 'child-plan.json'), '--out', str(cpu / 'prepared/qualification')],
             'actual child CLI must retain exactly the standard three arguments; no --run')
        result.update(status='PASS', command=command,
                      mlx_environment={k: v for k, v in env.items() if k.startswith('MLX_')})
    except BaseException as error:
        result.update(status='FAIL', error=dict(type=type(error).__name__, message=str(error),
                                               traceback=traceback.format_exc()))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started; write(cpu / 'result.json', result)
    pins = owner.source_pins(plan)
    extra = [Path(__file__).resolve(), HERE / 'execute_serving.py', HERE / 'test_execute_serving_cpu.py',
             HERE / 'test_serving_cpu.py', HERE / 'test_audit_serving_cpu.py',
             L / 'test_cpu.py', L / 'test_predecessor_cpu.py', P / 'owner-plan.json',
             *(PRIOR / name for name in ('restored-journal.json', 'restoration-audit.json',
                 'archive-verification.json', 'summary.json', 'resource-summary.json', 'independent-numerical.json')),
             *(CANARY_CHILD / name for name in ('plan.json', 'complete.json', 'events.jsonl', 'captures.json'))]
    for path in extra: pins[str(path)] = sha(path)
    target = P / 'frozen-source'; target.mkdir(exist_ok=False)
    for name, digest in pins.items():
        raw = Path(name).read_bytes(); need(hashlib.sha256(raw).hexdigest() == digest, 'source changed while freezing')
        dest = target / name.lstrip('/'); dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream: stream.write(raw)
    write(P / 'preparation.json', dict(schema='GDN-prework-serving-prereg-preparation-v1',
        status='CPU_PREPARED', owner_plan_sha256=plan_sha, child_plan_sha256=sha(P / 'child-plan.json'),
        lifecycle=lifecycle, predecessor=predecessor, cpu_prepare=ref(cpu / 'result.json'), source_pins=pins,
        canary=proof, expected_requests=dict(D_C1=8, D_C8=16, lifecycle=13),
        expected_total_requests=37, scope=qualifier.SCOPE,
        service_prepared=False, numerical_qualified=False, full_model_qualified=False,
        http_qualified=False, performance_qualified=False))
    journal, output = plan['outputs']['journal'], plan['outputs']['out']
    approved = ['--journal', journal, '--approved-contract', c.CONTRACT, '--plan-sha256', c.PLAN_SHA]
    analysis = A / (NAME + '-analysis')
    write(P / 'commands.json', dict(readonly_service_prepare=[PY, '-I', '-B', str(L / 'controller.py'),
        'prepare', '--journal', journal, '--plan-sha256', c.PLAN_SHA,
        '--owner-directory', output, '--owner-plan', str(P / 'owner-plan.json'), '--owner-plan-sha256', plan_sha],
        seal_receipt=[PY, '-I', '-B', str(Path(__file__).resolve()), '--seal-receipt'],
        execute=[PY, '-I', '-B', str(HERE / 'execute_serving.py'), '--run'],
        stop=[PY, '-I', '-B', str(L / 'controller.py'), 'stop', *approved],
        owner=[PY, '-I', '-B', str(L / 'owner.py'), '--plan', str(P / 'owner-plan.json'),
               '--plan-sha256', plan_sha, '--out', output],
        recover=[PY, '-I', '-B', str(L / 'controller.py'), 'recover', *approved, '--watchdog-seconds', '3600'],
        independent_numeric_audit=[PY, '-I', '-B', str(HERE / 'audit_serving.py'), '--child',
          str(Path(output) / 'phases' / PHASE / 'qualification'), '--plan-sha256', sha(P / 'child-plan.json'),
          '--out', str(analysis / 'independent-numerical.json')],
        independent_resource_audit=[PY, '-I', '-B', str(B / 'resource_reparse.py'), '--owner', output,
          '--out', str(analysis / 'resource-summary.json')],
        independent_restoration_audit=[PY, '-I', '-B', str(L / 'audit_restore.py'), '--journal', journal,
          '--out', str(analysis / 'restoration-audit.json')],
        audit_precondition='Closed owner and actual RESTORED/recovery rc0; create NEW analysis directory first.'))
    print(json.dumps(dict(status='CPU_PREPARED', owner_plan_sha256=plan_sha,
                         child_plan_sha256=sha(P / 'child-plan.json'), service_prepared=False)))


def seal_receipt():
    """Bind root's actual read-only PREPARED journal; never create it here."""
    owner, c, lifecycle, predecessor, qualifier, canary, proof = bindings()
    preparation = read(P / 'preparation.json')
    need(preparation['status'] == 'CPU_PREPARED' and preparation['lifecycle'] == lifecycle
         and preparation['predecessor'] == predecessor and preparation['canary'] == proof,
         'actual CPU preparation binding')
    pins = preparation['source_pins']
    for path, digest in pins.items(): need(sha(path) == digest, 'prepared source changed: ' + path)
    plan_sha = preparation['owner_plan_sha256']; plan = read(P / 'owner-plan.json')
    need(sha(P / 'owner-plan.json') == plan_sha, 'prepared owner plan changed')
    qualifier.validate(P / 'child-plan.json', preparation['child_plan_sha256'])
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
    write(P / 'receipt.json', dict(schema='GDN-prework-serving-prereg-receipt-v1', status='PREPARED',
        owner_plan_sha256=plan_sha, child_plan_sha256=preparation['child_plan_sha256'],
        prepared_journal_sha256=sha(journal_path), source_pins=pins,
        numerical_qualified=False, full_model_qualified=False, http_qualified=False, performance_qualified=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(); mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--seal-receipt', action='store_true'); args = p.parse_args()
    if args.prepare: prepare()
    elif args.seal_receipt: seal_receipt()
    else: print('DESCRIPTION_ONLY: archived canary PASS/restoration and reviewed serving auditor required; no operations.')


if __name__ == '__main__': main()
