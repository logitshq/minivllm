"""Six CPU corruption challenges against E020's independent raw-evidence audit.

Mutates deep copies in memory. Never changes raw evidence or its verified tar.
No model/server/GPU import. The tokenizer is CPU-only, from the pinned target.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-script', type=Path, default=Path(__file__).with_name('external_spec_http_audit.py'))
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Use a new challenge receipt; preserve earlier attempts'
    spec = importlib.util.spec_from_file_location('independent_e020_audit', args.audit_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tokenizers import Tokenizer
    plan = module.read(args.inputs / 'plan.json')
    tok = Tokenizer.from_file(str(args.tokenizer or Path(plan['workload']['model_dir']) / 'tokenizer.json'))
    source = args.run / 'pair1-native-none/warm-0.json'
    original = module.read(source)
    attempt = module.read(args.run / 'attempt.json')
    lease = module.read(args.lease / 'lease.json')
    resume = module.read(args.lease / 'resume.json')
    tests, baselines = [], []

    def request_check(row):
        audit = module.Audit()
        result = module.request_audit(audit, source, row, plan, tok, 1, 'native-none', 'warm', 0)
        return result, audit

    result, baseline = request_check(original)
    assert result['valid'] and not baseline.integrity_errors
    baselines.append('original raw SSE reconstructed successfully')

    changed = copy.deepcopy(original)
    changed['valid'] = True
    for event in changed['sse_events']:
        if event['data'] == '[DONE]':
            continue
        payload = json.loads(event['data'])
        if payload.get('choices') and payload['choices'][0].get('text'):
            payload['choices'][0]['text'] = 'UNQUALIFIED ' + payload['choices'][0]['text']
            event['data'] = json.dumps(payload)
            break
    result, checker = request_check(changed)
    assert not result['valid'] and checker.integrity_errors
    tests.append({'case': 'visible_ID_corruption', 'rejected': True, 'mutation': 'raw SSE text changed; cached text/IDs/counts and valid=true left untouched',
                  'contract_failures': result['failures'], 'independent_integrity_errors': len(checker.integrity_errors)})

    changed = copy.deepcopy(original)
    changed['valid'] = True
    changed['sse_events'][1]['client_perf_counter_s'] = changed['sse_events'][0]['client_perf_counter_s'] - 1
    result, checker = request_check(changed)
    assert not result['valid'] and any('clock' in f or 'outside' in f for f in result['failures'])
    tests.append({'case': 'reversed_SSE_timestamp', 'rejected': True, 'mutation': 'second event precedes first; stored timing metrics retained',
                  'contract_failures': result['failures'], 'independent_integrity_errors': len(checker.integrity_errors)})

    log_path = args.run / 'pair1-omlx-mtp/server.log'
    log = log_path.read_text()
    assert module.route_audit('omlx-mtp', log, plan)['valid']
    baselines.append('original official MTP logs establish six valid routes')
    changed, n = re.subn(r'accepted=\d+/\d+', 'accepted=9999/1', log, count=1)
    assert n == 1
    route = module.route_audit('omlx-mtp', changed, plan)
    assert not route['valid']
    tests.append({'case': 'forged_MTP_acceptance', 'rejected': True, 'mutation': 'ordinary log accepts more than its capacity; route.json valid flag is not consumed',
                  'contract_failures': route['failures']})

    folder = args.run / 'pair1-native-none'
    class MemoryAudit(module.Audit):
        def __init__(self, replacements):
            super().__init__()
            self.replacements = replacements
        def load(self, path, required=False):
            if Path(path) in self.replacements:
                return copy.deepcopy(self.replacements[Path(path)])
            return super().load(path, required)
    owner = module.lifecycle_audit(module.Audit(), folder, attempt['pid'], lease, resume)
    assert owner['valid']
    baselines.append('original listener/PPID/reap/absence proof valid')
    changed = copy.deepcopy(module.read(folder / 'ready.json'))
    fake_pid = owner['pid'] + 100000
    changed['listeners']['pids'] = [fake_pid]
    changed['listeners']['stdout'] = f'p{fake_pid}\ncfake-server\n'
    checker = MemoryAudit({folder / 'ready.json': changed})
    result = module.lifecycle_audit(checker, folder, attempt['pid'], lease, resume)
    assert not result['valid']
    tests.append({'case': 'forged_listener_PID', 'rejected': True, 'mutation': 'listener inventory points to a different PID while direct_child/valid booleans remain true',
                  'contract_failures': result['issues']})

    changed = copy.deepcopy(module.read(folder / 'owned-pid.json'))
    fields = changed['ps_at_launch']['stdout'].split(None, 2)
    changed['ps_at_launch']['stdout'] = f'{fields[0]} {attempt["pid"] + 1} {fields[2]}'
    checker = MemoryAudit({folder / 'owned-pid.json': changed})
    result = module.lifecycle_audit(checker, folder, attempt['pid'], lease, resume)
    assert not result['valid']
    tests.append({'case': 'forged_parent_PID', 'rejected': True, 'mutation': 'actual PS PPID differs from controller, declaration remains direct_child=true',
                  'contract_failures': result['issues']})

    original_service = module.lease_audit(module.Audit(), args, attempt, module.sha(args.inputs / 'plan.json'))
    assert original_service['valid']
    baselines.append('original authorized service restore identity/transition valid')
    changed = copy.deepcopy(resume)
    changed['identity_match'] = False
    checker = MemoryAudit({args.lease / 'resume.json': changed})
    result = module.lease_audit(checker, args, attempt, module.sha(args.inputs / 'plan.json'))
    assert not result['valid']
    tests.append({'case': 'wrong_restore_identity', 'rejected': True, 'mutation': 'resume identity_match=false despite stored T->S transition',
                  'contract_failures': result['issues']})

    assert len(tests) == 6 and all(t['rejected'] for t in tests)
    receipt = {'status': 'PASS_6_NEGATIVE_CHALLENGES', 'baselines': baselines, 'challenges': tests,
        'source_sha256': {'audit': module.sha(args.audit_script), 'challenge_script': module.sha(Path(__file__))},
        'original_evidence_sha256': {str(p): module.sha(p) for p in [source, log_path, folder / 'owned-pid.json',
            folder / 'ready.json', folder / 'release.json', args.lease / 'lease.json', args.lease / 'resume.json']},
        'scope': 'In-memory corruptions only; no original raw file, verified archive, server or GPU was modified'}
    module.write(args.out, receipt)
    print(json.dumps({'status': receipt['status'], 'challenges': len(tests), 'positive_baselines': len(baselines)}))


if __name__ == '__main__':
    main()
