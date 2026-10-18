"""Independent E028 nested-owner/cold-recovery audit; stdlib and offline only.

The original resource/recovery auditor remains byte-pinned and unmodified.
Original preflight pins are read where actually written, without inventing an
E023 launch or rearranging evidence. Missing terminal evidence is INCOMPLETE;
active numerical ledgers are never opened. No model/HTTP/service operation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / 'pareto_realistic/audit_absent_qualification.py'
SUPPORT_SHA = '1b7b526dac98c43f5b77f915aec3173d755d1efb28ed49b989e7601293ddaf6d'
OWNER_SHA = 'da80f65af9cafcbc382f1c8444fe5f6ca54a012aabea7ce8943ad85dd04c53b4'
PREREG = HERE / 'prereg-model-v1'
WRAPPER_SHA = '02a2cc98d068d037c6ddd5cda24fe2c5be13152013b919b5a94ddcdfdd0290f2'
CHILD_PLAN_SHA = 'bc4890e6d302141597eae8e0e9347c093307a8492c09f958f42bdb8d1c7aa73c'
MODEL_AUDIT_SHA = 'd46c60266e70b7998e2ebeb17d1c126ef4eb896226857419ba6146ae1b642c67'
SCOPE = 'E028 fixed 37-request numerical qualification plus offline owner/recovery evidence; no HTTP or throughput-latency performance point.'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load(path, digest, name):
    need(sha(path) == digest, 'independent auditor source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registration(a):
    wrapper_path, child_path = PREREG / 'owner-plan.json', PREREG / 'qualification-plan.json'
    need(sha(wrapper_path) == WRAPPER_SHA and sha(child_path) == CHILD_PLAN_SHA, 'preregistered plans changed')
    wrapper, child = a.read(wrapper_path), a.read(child_path)
    need(wrapper['qualification'] == {'path': str(child_path), 'sha256': CHILD_PLAN_SHA}, 'preregistered child binding')
    return dict(wrapper=wrapper, wrapper_path=str(wrapper_path), wrapper_sha=WRAPPER_SHA,
                child=child, child_sha=CHILD_PLAN_SHA)


def owner_binding(owner, journal, data, a, registered):
    complete = a.read(owner / 'complete.json')
    need(complete['schema'] == 'E023-service-absent-owner-result-v1' and complete['status'] == 'PASS'
         and complete['errors'] == [] and complete['spawn_intent_written'] is True
         and complete['child_started'] is True, 'owner is not a successful terminated child run')
    need(not (owner / 'terminal.json').exists(), 'NO_CHILD terminal conflicts with numerical PASS')
    wrapper = a.read(owner / 'owner-plan.json')
    need(wrapper == registered['wrapper'] and sha(owner / 'owner-plan.json') == registered['wrapper_sha'], 'actual owner plan differs from preregistration')
    need(type(wrapper['deadline_seconds']) is int and wrapper['deadline_seconds'] == 1830
         and a.finite(complete['elapsed_wall_s']) and 0 < complete['elapsed_wall_s'] <= 1895, 'owner work/cleanup bound')
    expected_owner = {'path': str(HERE / 'model_owner.py'), 'sha256': OWNER_SHA}
    expected_ctl = {'path': str(HERE.parent / 'omlx_cold_restart/controller_v2.py'), 'sha256': a.CONTROLLER_SHA}
    need(wrapper['execution'] == {'owner': expected_owner} and wrapper['cold_restart'] == {
        'journal_path': str(journal), 'controller': expected_ctl,
        'approved_contract': a.CONTRACT, 'proposal_plan_sha256': a.PROPOSAL_SHA}, 'owner/controller/journal binding')
    for item in (expected_owner, expected_ctl):
        need(sha(owner / 'owner-source' / Path(item['path']).name) == item['sha256'], 'frozen owner/controller source')
    q = wrapper['qualification']
    need(complete['owner_plan_sha256'] == registered['wrapper_sha']
         and complete['numerical_plan_sha256'] == registered['child_sha'], 'owner final plan bindings')
    frozen = a.read(owner / 'stopped-journal.json')
    need(frozen['state'] == 'STOPPED' and data['state'] == 'RESTORED', 'STOPPED-to-RESTORED required')
    expected = {'schema': 'omlx-cold-restart-journal-v2', 'plan_sha256': a.PROPOSAL_SHA,
        'controller_sha256': a.CONTROLLER_SHA, 'support_sha256': a.SUPPORT_SHA,
        'approved_contract': a.CONTRACT, 'original': a.ORIGINAL,
        'owner_directory': str(owner), 'owner_source': expected_owner,
        'owner_plan': {'path': registered['wrapper_path'], 'sha256': registered['wrapper_sha']}}
    need(all(frozen.get(k) == data.get(k) == v for k, v in expected.items()), 'stable journal identity')
    for key in ('id', 'removed_at', 'global_settings_stat', 'model_metadata', 'original_status', 'log_start'):
        need(frozen[key] == data[key], 'journal stable field changed: ' + key)
    need(isinstance(data['id'], str) and data['id'] and a.finite(data['removed_at'])
         and data['removed_at'] > 0, 'runtime journal identity')
    a.status_receipt(data['original_status'])
    need(data['events'][:len(frozen['events'])] == frozen['events'], 'STOPPED event prefix changed')
    attempt, intent, child, release = [a.read(owner / n) for n in ('attempt.json', 'spawn-intent.json', 'child.json', 'release.json')]
    opid = complete['owner_pid']; a.identity(attempt['owner_identity'], pid=opid)
    need(attempt['schema'] == 'cold-service-owner-attempt-v1' and attempt['owner_pid'] == opid
         and attempt['owner_source'] == expected_owner and attempt['owner_plan_sha256'] == registered['wrapper_sha']
         and attempt['journal_id'] == data['id'] and attempt['journal_path'] == str(journal)
         and attempt['controller_sha256'] == a.CONTROLLER_SHA and attempt['removed_at'] == data['removed_at']
         and attempt['phase'] == 'PRESPAWN' and attempt['stopped_journal_sha256'] == sha(owner / 'stopped-journal.json'), 'durable attempt binding')
    need(intent['schema'] == 'cold-service-owner-spawn-intent-v1' and intent['owner_pid'] == opid
         and intent['journal_id'] == data['id'] and intent['owner_plan_sha256'] == registered['wrapper_sha']
         and intent['attempt_sha256'] == sha(owner / 'attempt.json'), 'durable spawn intent binding')
    pid = child['pid']
    need(type(pid) is int and pid > 1 and pid not in (opid, 74384) and child['owner_pid'] == opid, 'direct child identity')
    need(release['pid'] == pid and type(release['returncode']) is int and release['returncode'] == 0
         and release['waited_reaped'] is True and release['strict_absence'] is True, 'strict direct child exit/reap')
    times = [data['removed_at'], attempt['created_at'], intent['time'], child['spawned_at'], release['time']]
    need(all(a.finite(t) for t in times) and times == sorted(times), 'owner chronological boundaries')
    absent = a.read(owner / 'child-absence.json')
    need(type(absent['returncode']) is int and absent == {
        'command': ['/bin/ps', '-p', str(pid), '-o', 'pid='], 'returncode': 1, 'stdout': ''}, 'raw child absence')
    saved_child = a.read(owner / 'qualification-plan.json')
    need(saved_child == registered['child'] and sha(owner / 'qualification-plan.json') == registered['child_sha']
         and sha(owner / 'qualification/plan.json') == registered['child_sha'], 'saved E028 child plans')
    nested = owner / 'e023-input-preflight'
    numeric = a.read(nested / 'plan.json')
    need(sha(nested / 'plan.json') == sha(owner / 'qualification/base/plan.json') == a.PLAN_SHA
         and saved_child['qualification']['sha256'] == a.PLAN_SHA, 'actual original nested plans')
    launch = a.read(owner / 'launch.json')
    need(launch == {'command': [numeric['execution']['python'], '-I', '-B', str(HERE / 'model_qualification.py'),
        '--plan', q['path'], '--plan-sha256', registered['child_sha'], '--out', str(owner / 'qualification')],
        'environment': numeric['execution']['environment'], 'numerical_plan_sha256': registered['child_sha']}, 'actual E028 CLI/environment')
    paths = [owner / n for n in ('attempt.json', 'stopped-journal.json', 'spawn-intent.json', 'child.json', 'release.json')]
    need(data['reap'] == {'no_experiment': False, 'no_child_started': False, 'owner_pid': opid,
        'receipts': [{'path': str(p), 'sha256': sha(p)} for p in paths]}, 'controller bound owner reap')
    return complete, numeric, saved_child, opid, pid, child['spawned_at'], release['time'], attempt['created_at']


def preflight_pins(owner, numeric, child, a):
    """Audit real nested preparation; no invented old launch/attempt object."""
    nested = owner / 'e023-input-preflight'
    verified = a.read(nested / 'verified-pins.json')
    need(len(numeric['execution']['pins']) == 122, '122 original execution pins required')
    expected = {**numeric['execution']['pins'], child['qualification']['path']: a.PLAN_SHA}
    manifests = {}
    for name in ('fixtures', 'source_manifest', 'model_manifest'):
        item = numeric[name]; expected[item['path']] = item['sha256']
        path = a.within(nested / 'frozen-source' / item['path'].lstrip('/'), nested)
        need(sha(path) == item['sha256'], 'nested manifest SHA: ' + name)
        manifests[name] = a.read(path)
    root = Path(numeric['package_root']) / 'tutorial_code/miniengine'
    for relative, digest in manifests['source_manifest']['files'].items():
        need(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'relative core source path')
        expected[str(root / relative)] = digest
    need(len(expected) == len(verified) == 157 and set(expected) == set(verified), 'exact157 original preflight entries')
    binary_bytes = 0
    for original, digest in expected.items():
        row = verified[original]; copied = Path(original).suffix in ('.py', '.json', '.md')
        need(row['sha256'] == digest and row['source_copied'] is copied
             and type(row['bytes']) is int and row['bytes'] >= 0, 'verified pin metadata')
        target = a.within(nested / 'frozen-source' / original.lstrip('/'), nested) if copied else a.require(original)
        need(target.stat().st_size == row['bytes'] and sha(target) == digest, 'original pin content: ' + original)
        if not copied: binary_bytes += row['bytes']
    extra = {**child['sources'], str(HERE.parent / 'pareto_realistic/qualify_absent_owner.py'): a.OWNER_SHA}
    recorded = a.read(owner / 'adapter-pins.json')
    need(len(extra) == 5 and set(recorded) == set(extra), 'exact five adapter source copies')
    for original, digest in extra.items():
        path = a.within(owner / 'frozen-source' / original.lstrip('/'), owner)
        row = recorded[original]
        need(row == {'sha256': digest, 'bytes': path.stat().st_size, 'source_copied': True}
             and type(row['bytes']) is int and sha(path) == digest, 'adapter source copy: ' + original)
    receipt = a.read(owner / 'adapter.json')
    need(receipt == {'schema': 'E028-model-owner-adapter-v1', 'status': 'PREPARED',
        'full_model_qualified': False, 'performance_qualified': False, 'independent_e028_audit_required': True,
        'child_plan_sha256': CHILD_PLAN_SHA, 'inherited_plan_sha256': a.PLAN_SHA,
        'inherited_evidence': 'e023-input-preflight', 'frozen_absent_owner_sha256': a.OWNER_SHA,
        'pressure_required': 1, 'work_deadline_seconds': 1830, 'child_deadline_seconds': 1800,
        'owner_result_schema': 'E023-service-absent-owner-result-v1',
        'qualification_scope': 'E028 full-model numerical only; no HTTP/performance qualification'}, 'adapter preparation claim')
    model = manifests['model_manifest']
    need(model['schema'] == 'E023-model-v1' and model['model'] == numeric['model']
         and model['revision'] == Path(numeric['model']).name, 'model manifest identity')
    return {'original_verified_entries': 157, 'execution_pins': 122, 'adapter_source_copies': 5,
        'runtime_bytes_rehashed': binary_bytes, 'weight_shards_rehashed': False,
        'scope': 'Actual frozen source/manifest/fixture copies and runtime binaries independently hashed; model weight recheck remains the numerical child receipt.'}


def numerical_auditor():
    return load(HERE / 'audit_model.py', MODEL_AUDIT_SHA, '_e028_independent_numerical_auditor')


def audit(owner, journal):
    owner, journal = Path(owner).resolve(), Path(journal).resolve()
    result = {'schema': 'E028-model-owner-audit-v1', 'qualified': False, 'ledger_opened': False,
              'performance_qualified': False, 'scope': SCOPE}
    try:
        a = load(SUPPORT, SUPPORT_SHA, '_e028_frozen_resource_recovery_audit')
    except Exception as error:
        result.update(status='FAIL', first_failure={'type': type(error).__name__, 'message': str(error)})
        return result
    try:
        data = a.read(journal / 'journal.json')
        if data.get('state') != 'RESTORED': raise a.Incomplete('controller not RESTORED')
        for name in ('final_status', 'log_window', 'reap', 'load_worker', 'post_result', 'replacement', 'resource_samples'):
            if name not in data: raise a.Incomplete('missing final controller field: ' + name)
        for path in [owner / 'complete.json', owner / 'qualification/complete.json', owner / 'qualification/base/complete.json',
                     owner / 'release.json', journal / 'load-worker/release.json', journal / 'load-worker/result.json']:
            a.require(path)
        for path in (owner / 'complete.json', owner / 'qualification/complete.json', owner / 'qualification/base/complete.json'):
            if a.read(path).get('status') == 'INCOMPLETE': raise a.Incomplete('unfinished result: ' + str(path))
        complete, numeric, child, opid, cpid, spawned, released, created = owner_binding(owner, journal, data, a, registration(a))
        result['owner_resources'] = a.owner_resources(owner, complete, opid, cpid, spawned, released, created)
        result['service_restore'] = a.restore_evidence(data, journal, max(released, result['owner_resources']['last_recorded_at']))
        result['pins'] = preflight_pins(owner, numeric, child, a)
        result['numerical'] = numerical_auditor().audit(owner / 'qualification')
        result['ledger_opened'] = result['numerical'].get('ledger_opened', False)
        status = result['numerical']['status']
        need(status in ('PASS', 'FAIL', 'INCOMPLETE'), 'numerical auditor status')
        if status == 'INCOMPLETE': raise a.Incomplete('numerical audit incomplete')
        need(status == 'PASS' and result['numerical']['qualified'] is True, 'numerical audit rejected qualification')
        result.update(status='PASS', qualified=True, owner_plan_sha256=WRAPPER_SHA,
                      child_plan_sha256=CHILD_PLAN_SHA, resource_auditor_sha256=SUPPORT_SHA,
                      numerical_auditor_sha256=MODEL_AUDIT_SHA)
    except (a.Incomplete, FileNotFoundError) as error:
        result.update(status='INCOMPLETE', reason=str(error))
    except Exception as error:
        result.update(status='FAIL', first_failure={'type': type(error).__name__, 'message': str(error)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True, type=Path)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    need(not args.out.exists() and args.out.parent.is_dir(), 'new audit output required')
    result = audit(args.owner, args.journal); result['audit_source_sha256'] = sha(__file__)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return {'PASS': 0, 'FAIL': 1, 'INCOMPLETE': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
