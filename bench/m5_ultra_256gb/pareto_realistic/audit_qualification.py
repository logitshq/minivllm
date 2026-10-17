"""Independent stdlib E023 v1 audit. Never imports measurement/GPU code.

--owner DIR --lease DIR --out NEW_JSON. Missing terminal receipts yield
INCOMPLETE without opening the active ledger. A completed ledger is streamed
one line at a time and SHA256'd in that same pass. No model shard is rehashed.
Closed resource-aborted runs are audited as RESOURCE_ABORT, never qualified.
PASS means the fixed numerical experiment and its owner/restore evidence only;
it grants no HTTP qualification or performance result. --self-test is CPU only.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import traceback


PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
FIXTURE_SHA = '6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db'
CASES = ('D-C1', 'D-C8', 'lifecycle')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def same(left, right, label):
    need(json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False), label)


def read_json(path):
    path = Path(path)
    need(path.suffix != '.jsonl', 'JSONL must be streamed, never read wholesale')
    with path.open() as stream:
        return json.load(stream)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 << 20):
            result.update(chunk)
    return result.hexdigest()


def array_refs(value):
    if isinstance(value, dict):
        if set(value) == {'array'}:
            return {value['array']}
        return set().union(*(array_refs(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(array_refs(v) for v in value)) if value else set()
    return set()


def expected_jobs(fixtures, case):
    def job(i, cap, prefix):
        return {'rid': prefix + str(i), 'fixture': fixtures[i % 8], 'cap': cap}
    if case == 'D-C1':
        return [job(i, 512, 'c1-') for i in range(8)]
    if case == 'D-C8':
        return [job(i, 512, 'c8-') for i in range(16)]
    return ([job(i, cap, 'probe-') for i, cap in enumerate((1, 2, 3, 5, 8, 13, 21, 34))]
        + [job(i + 8, cap, 'probe-') for i, cap in enumerate((2, 5, 9, 17))]
        + [{'rid': 'probe-0', 'fixture': fixtures[1], 'cap': 3}])


class CaseAudit:
    """Small independent simulation of the fixed FCFS/budget/retirement contract."""
    def __init__(self, name, fixtures, config, eos, pending=()):
        self.name, self.jobs, self.eos, self.config = name, expected_jobs(fixtures, name), set(eos), config
        self.C = 1 if name == 'D-C1' else 8
        self.requests, self.waiting, self.running = {}, [], []
        self.free, self.finished = list(range(self.C - 1, -1, -1)), list(pending)
        self.admissions, self.completed, self.hist = [], [], Counter()
        self.next, self.step, self.previous_ahead = 0, 0, None
        self.previous_arrays, self.staging = set(), set()
        self.previous_batch = 0
        self.array_counts, self.ahead_counts = Counter(), Counter()
        self.layout = config['layer_types']

    def metadata(self, rid):
        req = self.requests[rid]
        return {'rid': rid, 'prompt_ids': req['job']['fixture']['prompt_ids'], 'output_ids': req['ids'],
            'computed': req['computed'], 'slot': req['slot'], 'status': req['status'],
            'sampling': {'max_tokens': req['job']['cap'], 'temperature': 0, 'top_p': 1.0,
                         'top_k': 0, 'ignore_eos': False, 'stop_token_ids': []}}

    def admit(self):
        if self.name != 'lifecycle':
            count = min(len(self.jobs) - self.next, self.C - len(self.requests))
        else:
            count = 8 if self.step == 1 else 4 if self.step == 2 else 1 if not self.requests and self.next == 12 else 0
        for _ in range(count):
            job = self.jobs[self.next]
            rid = job['rid']
            need(rid not in self.requests, 'active rid reused')
            entry = {'ordinal': self.next, 'rid': rid, 'fixture_id': job['fixture']['fixture_id'],
                     'cap': job['cap'], 'before_step': self.step}
            self.admissions.append(entry)
            self.requests[rid] = {'job': job, 'entry': entry, 'ids': [], 'computed': 0, 'slot': None, 'status': 'WAITING'}
            self.waiting.append(rid)
            self.next += 1

    def schedule(self):
        budget, prefill = 2048, []
        decode = [rid for rid in self.running if self.requests[rid]['computed'] >= len(self.requests[rid]['job']['fixture']['prompt_ids'])]
        budget -= len(decode)
        for rid in self.running:
            req = self.requests[rid]
            remaining = len(req['job']['fixture']['prompt_ids']) - req['computed']
            if remaining > 0 and budget > 0:
                n = min(remaining, budget)
                prefill.append([rid, n]); budget -= n
        while self.waiting and self.free and budget > 0:
            rid = self.waiting.pop(0)
            req = self.requests[rid]
            req.update(slot=self.free.pop(), status='RUNNING')
            self.running.append(rid)
            n = min(len(req['job']['fixture']['prompt_ids']), budget)
            prefill.append([rid, n]); budget -= n
        return {'decode': decode, 'prefill': prefill, 'finished_ids': self.finished,
                'requests': [self.metadata(rid) for rid in self.requests]}

    def cache_names(self, prefix='root', single=False):
        names = set()
        for i, kind in enumerate(self.layout):
            fields = ('keys', 'values') if single and kind == 'full_attention' else ('K', 'V') if kind == 'full_attention' else ('conv_state', 'S') if single else ('conv', 'S')
            names.update(f'{prefix}/{i}/{field}' for field in fields)
        return names

    def check_arrays(self, row, calls, pre_staging, has_future, B):
        groups = defaultdict(dict)
        for entry in row['arrays']:
            label, name = entry['check'], entry['name']
            need(name not in groups[label], 'duplicate array comparison')
            need(all(entry[k] is True for k in ('bit_equal', 'reference_finite', 'candidate_finite')), 'false/missing bit/finite flag')
            shape, dtype = entry['shape'], entry['dtype']
            need(isinstance(shape, list) and all(type(x) is int and x > 0 for x in shape), 'invalid array shape')
            if label.startswith('logits '):
                need(name == 'root' and len(shape) == 2 and shape[-1] == self.config['vocab_size'], 'full-vocabulary logits missing')
                need(dtype == 'mlx.core.bfloat16', 'logits original dtype changed')
                self.array_counts['full_vocabulary_logits'] += 1
            elif name.endswith('/tokens'):
                need(dtype in ('mlx.core.uint32', 'mlx.core.int32', 'mlx.core.int64', 'mlx.core.uint64') and len(shape) == 1, 'invalid ahead token array')
                self.array_counts['ahead_token_arrays'] += 1
            else:
                field = name.rsplit('/', 1)[-1]
                if field in ('K', 'V', 'keys', 'values'):
                    need(len(shape) == 4 and shape[1] == 4 and shape[3] == 256 and 0 < shape[2] <= 2048 and shape[2] % 256 == 0, 'invalid full KV shape')
                elif field in ('conv', 'conv_state'):
                    need(shape[-2:] == [3, 10240] and len(shape) == (3 if field == 'conv' else 2), 'invalid convolution state')
                elif field == 'S':
                    need(shape[-3:] == [48, 128, 128] and len(shape) in (3, 4), 'invalid recurrent state')
                else:
                    raise ValueError('unrecognized state array ' + name)
                need(dtype == ('mlx.core.float32' if field == 'S' else 'mlx.core.bfloat16'), 'state original dtype changed')
                self.array_counts['state_arrays'] += 1
            groups[label][name] = entry
            self.array_counts['all_arrays'] += 1
        expected = {}
        for key in calls:
            single = key[0] == 'prefill'
            literal = tuple(key) if single else ('decode', tuple(key[1]), tuple(key[2]), tuple(key[3]))
            label = repr(literal)
            expected['logits ' + label] = {'root'}
            expected['call-state ' + label] = self.cache_names(single=single)
            logits = groups['logits ' + label].get('root')
            need(logits is not None and logits['shape'][0] == (1 if single else len(key[1])), 'logits batch dimension')
        if B:
            for arm in ('reference', 'candidate'):
                expected[arm + '/committed-not-rewritten-by-ahead'] = self.cache_names()
        postrun = self.cache_names('root/committed') if B else set()
        for rid in pre_staging:
            postrun |= self.cache_names('root/staging/' + rid, single=True)
        if has_future:
            postrun |= self.cache_names('root/ahead/layers') | {'root/ahead/tokens'}
        expected.update({'before-step-state': self.previous_arrays, 'before-retirement-state': postrun,
                         'after-retirement-state': array_refs(row['after']['state'])})
        expected = {label: names for label, names in expected.items() if names}
        need(set(groups) == set(expected), 'array check group omitted/added')
        for label, names in expected.items():
            need(set(groups[label]) == names, 'incomplete all-layer comparison: ' + label)
            if label.startswith('logits '):
                continue
            group_B = (self.previous_batch if label == 'before-step-state' else
                       len(row['after']['engine']['batch_rids']) if label == 'after-retirement-state' else B)
            for name, entry in groups[label].items():
                field, shape = name.rsplit('/', 1)[-1], entry['shape']
                single = '/staging/' in name or label.startswith("call-state ('prefill',")
                width = 1 if single else group_B
                if field in ('K', 'V', 'keys', 'values'):
                    need(shape[0] == width, 'KV batch dimension mismatch')
                elif field == 'S':
                    need(shape == ([48, 128, 128] if single else [width, 48, 128, 128]), 'S batch/rank mismatch')
                elif field in ('conv', 'conv_state'):
                    need(shape == ([3, 10240] if single else [width, 3, 10240]), 'conv batch/rank mismatch')
                elif field == 'tokens':
                    need(shape == [width], 'ahead token batch mismatch')

    def accept(self, row):
        self.step += 1
        need(row['case'] == self.name and row['step'] == self.step and row['status'] == 'PASS', 'case/continuous step/PASS mismatch')
        self.admit()
        schedule = self.schedule()
        same(row['schedule'], schedule, 'independent FCFS/metadata schedule mismatch')
        decode, prefill = schedule['decode'], schedule['prefill']
        B = len(decode); self.hist[B] += 1
        calls = []
        if decode:
            current = ['decode', decode, [self.requests[r]['computed'] for r in decode], [self.requests[r]['ids'][-1] for r in decode]]
            calls.append(current)
        has_future = bool(decode and not prefill and all(len(self.requests[r]['ids']) + 1 < self.requests[r]['job']['cap']
            and len(self.requests[r]['job']['fixture']['prompt_ids']) + len(self.requests[r]['ids']) + 1 < 2048 for r in decode))
        same(row['reference_delivered'], row['candidate_delivered'], 'two arm output tuples differ')
        delivered = row['reference_delivered']
        expected_delivered = decode + [rid for rid, n in prefill if self.requests[rid]['computed'] + n == len(self.requests[rid]['job']['fixture']['prompt_ids'])]
        same([x[0] for x in delivered], expected_delivered, 'missing/duplicate/reordered delivered request')
        future = ['decode', decode, [self.requests[r]['computed'] + 1 for r in decode], [x[1] for x in delivered]] if has_future else None
        if has_future:
            calls.append(future)
        calls += [['prefill', rid, self.requests[rid]['computed'], n] for rid, n in prefill]
        same(row['call_keys'], calls, 'semantic call coverage mismatch')
        consumed = current if decode and current == self.previous_ahead else None
        same(row['reference_discarded_ahead'], self.previous_ahead, 'reference ahead discard mismatch')
        for rid, n in prefill:
            self.requests[rid]['computed'] += n
        for rid in decode:
            self.requests[rid]['computed'] += 1
        for rid, token, done in delivered:
            need(type(token) is int and 0 <= token < self.config['vocab_size'] and type(done) is bool, 'token/done types/range')
            req = self.requests[rid]
            need(not req['ids'] or req['ids'][-1] not in self.eos, 'output after EOS')
            req['ids'].append(token)
            cap = req['job']['cap']
            need(len(req['ids']) <= cap, 'over-cap raw output')
            expected_done = token in self.eos or len(req['ids']) == cap
            need(done == expected_done, 'normal EOS/cap retirement mismatch')
            if done:
                self.completed.append({**req['entry'], 'raw_ids': list(req['ids']),
                    'committed_ids': req['ids'][:-1] if token in self.eos else list(req['ids'])})
        finished = [rid for rid in self.running if any(x[0] == rid and x[2] for x in delivered)]
        for rid in finished:
            self.free.append(self.requests[rid]['slot'])
            del self.requests[rid]
        self.running = [rid for rid in self.running if rid not in finished]
        self.finished = finished
        pre_staging = (self.staging - set(decode)) | {r for r, _ in prefill}
        self.staging = pre_staging - set(finished)
        carried = future if not finished else None
        same(row['ahead']['reference'], {'carried_future': carried, 'consumed_prior_future': None,
             'discarded_prior_future': self.previous_ahead, 'discarded_current_calls': [future] if future and not carried else []}, 'reference ahead accounting')
        same(row['ahead']['candidate'], {'carried_future': carried, 'consumed_prior_future': consumed,
             'discarded_prior_future': self.previous_ahead if consumed is None else None,
             'discarded_current_calls': [future] if future and not carried else []}, 'candidate ahead accounting')
        batch = [rid for rid in decode if rid not in finished]
        after = row['after']['engine']
        expected_after = {'requests': [self.metadata(rid) for rid in self.requests], 'waiting': self.waiting,
            'running': self.running, 'free_slots': self.free, 'finished_pending': finished,
            'batch_rids': batch, 'batch_lens': [self.requests[r]['computed'] for r in batch],
            'has_work': bool(self.waiting or self.running), 'ahead_key': None}
        if carried:
            expected_after['ahead_key'] = [carried[1], carried[2],
                [[0, 1.0, 0, self.requests[r]['job']['cap'], False, []] for r in decode], carried[3]]
        same(after, expected_after, 'after-retirement request/slot/batch/ahead metadata')
        expected_after_arrays = self.cache_names('root/committed') if batch else set()
        for rid in self.staging:
            expected_after_arrays |= self.cache_names('root/staging/' + rid, single=True)
        if carried:
            expected_after_arrays |= self.cache_names('root/ahead/layers') | {'root/ahead/tokens'}
        need(array_refs(row['after']['state']) == expected_after_arrays, 'after-state full tree coverage')
        self.check_arrays(row, calls, pre_staging, has_future, B)
        self.previous_arrays = expected_after_arrays
        self.previous_batch = len(batch)
        self.previous_ahead = carried
        self.ahead_counts.update(produced=int(has_future), carried=int(carried is not None), consumed=int(consumed is not None), invalidated=int(has_future and not carried))

    def finish(self, claimed):
        need(self.next == len(self.jobs) and not self.requests and not self.waiting and not self.running and not self.staging and self.previous_ahead is None, 'case not fully drained')
        if self.name == 'D-C8':
            need(max(self.hist, default=0) >= 2, 'C8 never exercised multirow decode')
        expected = {'name': self.name, 'steps': self.step,
            'step_limit': sum(len(j['fixture']['prompt_ids']) + j['cap'] for j in self.jobs) + 1,
            'admissions': self.admissions, 'completed': self.completed,
            'decode_batch_histogram': {str(k): v for k, v in self.hist.items()},
            'raw_output_count': sum(len(x['raw_ids']) for x in self.completed)}
        same(claimed, expected, 'complete.json case differs from streamed reconstruction')
        return {**expected, 'array_counts': dict(self.array_counts), 'ahead_counts': dict(self.ahead_counts),
                'committed_output_count': sum(len(x['committed_ids']) for x in self.completed)}


def partial_summary(state):
    return {'name': state.name, 'status': 'INCOMPLETE_NUMERICAL_NOT_QUALIFIED',
        'last_complete_step': state.step, 'admissions': state.admissions, 'completed': state.completed,
        'active': [{'ordinal': req['entry']['ordinal'], 'rid': rid, 'raw_ids': req['ids'],
                    'computed': req['computed'], 'slot': req['slot'], 'status': req['status']}
                   for rid, req in state.requests.items()],
        'waiting': state.waiting, 'running': state.running, 'next_admission_ordinal': state.next,
        'decode_batch_histogram': {str(k): v for k, v in state.hist.items()},
        'array_counts': dict(state.array_counts), 'ahead_counts': dict(state.ahead_counts),
        'raw_output_count': sum(len(x['raw_ids']) for x in state.completed)
                            + sum(len(r['ids']) for r in state.requests.values()),
        'committed_output_count': sum(len(x['committed_ids']) for x in state.completed)
                                  + sum(len(r['ids']) for r in state.requests.values())}


def stream_ledger(path, fixtures, config, eos, claimed, partial=False):
    digest, total_bytes, lines, maximum_line = hashlib.sha256(), 0, 0, 0
    states, events, current, truncated = [], [], None, None
    with Path(path).open('rb') as stream:
        for raw in stream:
            lines += 1; total_bytes += len(raw); maximum_line = max(maximum_line, len(raw)); digest.update(raw)
            if not raw.endswith(b'\n'):
                need(partial, 'truncated final ledger line')
                truncated = {'line': lines, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                continue
            row = json.loads(raw)
            if 'event' in row:
                events.append(row)
                continue
            if current is None or row['case'] != current.name:
                index = len(states)
                need(index < 3 and row['case'] == CASES[index], 'case order/duplicate case block')
                if current is not None:
                    need(index <= len(claimed), 'started new case without preceding completed receipt')
                    current.finish(claimed[index - 1])
                pending = current.finished if current is not None and row['case'] == 'lifecycle' else ()
                current = CaseAudit(row['case'], fixtures, config, eos, pending)
                states.append(current)
            try:
                current.accept(row)
            except Exception as error:
                raise ValueError(f'ledger line {lines}, {row["case"]} step {row["step"]}: {error}') from error
    need([c['name'] for c in claimed] == list(CASES[:len(claimed)]), 'completed case receipt order')
    if partial:
        need(len(claimed) <= len(states) <= min(3, len(claimed) + 1), 'partial case prefix coverage')
    else:
        need(len(states) == len(claimed) == 3, 'missing case')
    summaries = [state.finish(c) for state, c in zip(states, claimed)]
    full_events = ['load-pair-start', 'load-pair-complete', 'release-pair-start', 'release-pair-complete'] * 2
    same([e['event'] for e in events], full_events[:len(events)] if partial else full_events, 'load/release lifecycle events')
    loads = [e['max_seqs'] for e in events if e['event'] == 'load-pair-start']
    need(loads == [1, 8][:len(loads)] and (partial or loads == [1, 8]), 'C1/C8 model reload domains')
    need(all(e['independent_weight_arrays'] == 1349 for e in events if e['event'] == 'load-pair-complete'), 'independent model weight count')
    if states:
        need(len(events) >= (2 if len(states) == 1 else 6), 'steps without completed model pair load')
    counts = sum((s.array_counts for s in states), Counter())
    return {'sha256': digest.hexdigest(), 'bytes': total_bytes, 'lines': lines, 'max_line_bytes': maximum_line,
            'complete_lines': lines - int(truncated is not None), 'truncated_final_line': truncated,
            'completed_step_rows': sum(s.step for s in states), 'array_counts': dict(counts),
            'bit_equal_checks': counts['all_arrays'], 'finite_checks': 2 * counts['all_arrays'],
            'two_arm_delivered_tokens': sum(sum(len(x['raw_ids']) for x in s.completed)
                + sum(len(r['ids']) for r in s.requests.values()) for s in states),
            'cases': summaries, 'partial_cases': [partial_summary(s) for s in states[len(claimed):]],
            'events': events, 'scope': 'Recorded bit/finite checks independently accounted; original GPU arithmetic is not rerun'}


def audit_pins(owner, plan, attempt):
    need(len(plan['execution']['pins']) == 122, 'expected122 execution pins')
    verified = read_json(owner / 'verified-pins.json')
    command = attempt['command']
    source_plan = command[command.index('--plan') + 1]
    need(command[command.index('--plan-sha256') + 1] == PLAN_SHA, 'child command plan SHA')
    need(command[0] == plan['execution']['python'] and command[1:3] == ['-I', '-B'] and command[3] == plan['execution']['qualifier']['path'], 'child command interpreter/qualifier')
    same(attempt['environment'], plan['execution']['environment'], 'four MLX environment values')
    expected = {**plan['execution']['pins'], source_plan: PLAN_SHA}
    manifests = {}
    for field in ('fixtures', 'source_manifest', 'model_manifest'):
        item = plan[field]
        expected[item['path']] = item['sha256']
        frozen = owner / 'frozen-source' / item['path'].lstrip('/')
        need(sha(frozen) == item['sha256'], 'frozen manifest SHA')
        manifests[field] = read_json(frozen)
    root = Path(plan['package_root']) / 'tutorial_code/miniengine'
    expected.update({str(root / p): v for p, v in manifests['source_manifest']['files'].items()})
    need(len(expected) == 157 and set(expected) == set(verified), 'exact157 provenance entries (122 execution + source/plan/manifests)')
    for original, expected_sha in expected.items():
        row = verified[original]
        need(row['sha256'] == expected_sha, 'owner recorded wrong SHA')
        copied = Path(original).suffix in ('.py', '.json', '.md')
        need(row['source_copied'] is copied, 'frozen-source copy contract')
        actual = owner / 'frozen-source' / original.lstrip('/') if copied else Path(original)
        need(actual.stat().st_size == row['bytes'] and sha(actual) == expected_sha, 'independent pinned file hash mismatch: ' + original)
    model = manifests['model_manifest']
    need(model['schema'] == 'E023-model-v1' and model['model'] == plan['model'] and model['revision'] == Path(plan['model']).name, 'model manifest identity')
    config_path = Path(plan['model']) / 'config.json'
    need(sha(config_path) == model['files']['config.json'], 'model config SHA')
    config = read_json(config_path); config = config.get('text_config', config)
    need(len(config['layer_types']) == 64 and config['vocab_size'] == 248320, 'fixed model layer/vocabulary contract')
    fixtures = manifests['fixtures']
    need(plan['fixtures']['sha256'] == FIXTURE_SHA and fixtures['model'] == plan['model'] and len(fixtures['fixtures']) == 8, 'fixture/model identity')
    return fixtures['fixtures'], config, {'execution_pins': 122, 'all_verified_entries': 157,
        'weight_shards_rehashed_by_auditor': False, 'weight_scope': 'Frozen manifest and owner source/runtime receipts independently hashed; qualifier final model receipt checked separately when present'}


def audit_resources(owner, child, owner_pid, owner_complete, resource_abort=False):
    paths = sorted(owner.glob('resource-[0-9][0-9][0-9].json'))
    need(len(paths) == owner_complete['samples'] >= 3, 'resource sample count')
    need([p.name for p in paths] == [f'resource-{i:03d}.json' for i in range(len(paths))], 'resource sample sequence')
    samples, deltas, known, violations = [], [], 0, []
    for p in paths:
        x = read_json(p); samples.append(x)
        need(x['process_inventory_ok'] is True and x['native_process_inventory_ok'] is True, 'inventory failure')
        need(x['pressure_level_value'] in (1, 2, 4), 'unrecognized pressure value')
        if x['pressure_level_value'] >= 2:
            violations.append({'index': len(samples) - 1, 'label': x['label'], 'guard': 'pressure', 'value': x['pressure_level_value']})
        need(int(x['pressure_level']['text']) == x['pressure_level_value'], 'raw pressure mismatch')
        free = int(re.search(r'free percentage:\s*(\d+)', x['memory_pressure']['text'])[1])
        swap = float(re.search(r'used = ([0-9.]+)M', x['swap']['text'])[1])
        need(free == x['free_percent'] and swap == x['swap_used_mb'], 'raw memory/swap mismatch')
        need(x['power_metadata']['source'] == 'AC Power' and x['processinfo_thermal_metadata']['code'] in (0, 1), 'power/thermal gate')
        need('AC Power' in x['power_metadata']['raw']['stdout'] and int(x['processinfo_thermal_metadata']['raw']['stdout']) == x['processinfo_thermal_metadata']['code'], 'raw power/thermal mismatch')
        for key, value in x['vm_counters'].items():
            need(int(re.search(r'^' + key + r':\s*(\d+)', x['vm_stat']['text'], re.M)[1]) == value, 'raw VM counter mismatch')
        need({'Swapouts', 'Pageouts'} <= set(x['vm_counters']), 'paging counters absent')
        native = x['native_inference_processes']
        service = [entry for entry in native if entry['pid'] == 74384]
        need(len(service) == 1 and service[0]['authorized_suspended'] is True and 'T' in service[0]['state'], 'service not identically suspended')
        need(all(e['pid'] in (child, 74384) for e in native) and all(e['owned'] or e['pid'] == 74384 for e in x['known_inference_processes']), 'foreign native process')
        need(x['no_active_foreign_native_processes'] is True and x['authorized_suspended_pids'] == [74384], 'native guard receipt')
        if x['label'] == 'child-running':
            need(x['owned_model_pid'] == child, 'wrong resource target')
            exact = x['exact_owned_pid_metadata']
            if exact['status'] == 'known':
                need(exact['pid'] == child and exact['ppid'] == owner_pid, 'child ownership mismatch')
                raw = exact['raw']['stdout'].split(None, 3)
                need([int(raw[0]), int(raw[1])] == [child, owner_pid], 'raw PID/PPID mismatch')
                known += 1
            else:
                need(p == paths[-2] and exact['raw']['returncode'] == 1, 'unexplained missing owned PID sample')
        if len(samples) > 1:
            prev = samples[-2]
            need(x['monotonic_s'] > prev['monotonic_s'], 'resource time not monotonic')
            delta = {'swap': swap - prev['swap_used_mb'], 'Swapouts': x['vm_counters']['Swapouts'] - prev['vm_counters']['Swapouts']}
            deltas.append(delta)
            if len(deltas) >= 2:
                if all(d['swap'] > 64 for d in deltas[-2:]) or all(d['Swapouts'] > 0 for d in deltas[-2:]):
                    violations.append({'index': len(samples) - 1, 'label': x['label'], 'guard': 'sustained_swap'})
    need(samples[0]['label'] == 'preflight' and samples[-1]['label'] == 'postflight' and known > 0, 'resource pre/post/owned coverage')
    if resource_abort:
        need(violations and violations[0] == {'index': len(samples) - 2, 'label': 'child-running',
             'guard': 'pressure', 'value': samples[-2]['pressure_level_value']}, 'resource abort not immediate first pressure violation')
        need(all(v['index'] >= len(samples) - 2 for v in violations), 'ignored resource guard earlier in run')
        need(samples[-1]['pressure_level_value'] >= 2, 'postflight pressure error not supported')
    else:
        need(not violations, 'resource guard failure')
    aggregate_path = owner / 'resources.json'
    if aggregate_path.exists():
        aggregate = read_json(aggregate_path)
        need(len(aggregate) == len(samples), 'aggregate resources count')
        for i, (raw, full) in enumerate(zip(samples, aggregate)):
            same({k: v for k, v in full.items() if k not in ('swap_used_mb_delta', 'vm_counter_delta')},
                 {k: v for k, v in raw.items() if k not in ('swap_used_mb_delta', 'vm_counter_delta')}, 'aggregate/raw sample mismatch')
            if i:
                need(full['swap_used_mb_delta'] == raw['swap_used_mb'] - samples[i - 1]['swap_used_mb'], 'aggregate swap delta')
                same(full['vm_counter_delta'], {k: v - samples[i - 1]['vm_counters'][k] for k, v in raw['vm_counters'].items()}, 'aggregate VM deltas')
    else:
        need(resource_abort, 'resources aggregate missing on successful run')
    return {'status': 'RESOURCE_ABORT' if violations else 'PASS', 'violations': violations,
            'samples': len(samples), 'known_owned_samples': known,
            'aggregate_present': aggregate_path.exists(),
            'aggregate_note': 'Postflight guard raises before aggregate write on this frozen owner' if not aggregate_path.exists() else None,
            'max_owned_rss_kib': max(x.get('owned_pid_rss_kib') or 0 for x in samples),
            'swap_delta_mb': samples[-1]['swap_used_mb'] - samples[0]['swap_used_mb'],
            'max_swap_used_mb': max(x['swap_used_mb'] for x in samples),
            'swapouts_delta': samples[-1]['vm_counters']['Swapouts'] - samples[0]['vm_counters']['Swapouts'],
            'pageouts_delta': samples[-1]['vm_counters']['Pageouts'] - samples[0]['vm_counters']['Pageouts'],
            'scope': 'Recorded probes; unrecognized runtimes/GUI GPU use remain outside the inventory'}


def audit(owner, lease):
    required = [owner / 'complete.json', owner / 'release.json', owner / 'qualification/complete.json',
                lease / 'lease.json', lease / 'resume.json', lease / 'job-0.json']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return {'status': 'INCOMPLETE', 'qualified': False, 'missing_terminal_evidence': missing, 'ledger_opened': False}
    owner_complete, qualified = read_json(required[0]), read_json(required[2])
    lease_data = read_json(lease / 'lease.json')
    if not lease_data['closed'] or owner_complete['status'] == 'INCOMPLETE':
        return {'status': 'INCOMPLETE', 'qualified': False, 'ledger_opened': False, 'reason': 'owner/lease not terminal'}
    resource_abort = owner_complete['status'] == 'FAIL' and qualified['status'] == 'INCOMPLETE'
    if resource_abort:
        errors = owner_complete['errors']
        need([e['phase'] for e in errors] == ['execution', 'postflight'] and all(
             e['traceback'].endswith('RuntimeError: Resource pressure warning: unload owned model and pause\n')
             for e in errors), 'owner failure not solely expected resource pressure abort')
        same(read_json(owner / 'first-error.json'), errors[0], 'first-error owner mismatch')
        same(read_json(owner / 'errors.json'), errors, 'owner error receipts mismatch')
    else:
        need(owner_complete['status'] == qualified['status'] == 'PASS' and owner_complete['errors'] == [], 'owner/qualifier failed')
    plan = read_json(owner / 'plan.json')
    need(sha(owner / 'plan.json') == PLAN_SHA and sha(owner / 'qualification/plan.json') == PLAN_SHA, 'fixed plan SHA')
    same(plan['configuration'], {'max_len': 2048, 'max_seqs_by_case': {'D-C1': 1, 'D-C8': 8, 'lifecycle': 8}, 'max_batched_tokens': 2048, 'cache_limit_gib': 2}, 'fixed capacities')
    same(plan['candidate'], {'compile_decode': True, 'fuse_decode': True, 'gates_compile': True, 'norm_reuse': False}, 'fixed candidate switches')
    need(qualified['plan_sha256'] == PLAN_SHA and qualified['numeric_only_not_http'] is True, 'qualifier plan/scope receipt')
    if not resource_abort:
        need(qualified['source_model_pins_rechecked'] is True and qualified['total_requests'] == 37, 'qualifier final pin/request receipt')
    else:
        need('source_model_pins_rechecked' not in qualified and 'total_requests' not in qualified, 'aborted run unexpectedly claims final pin/request receipt')
    need(qualified['qualifier_sha256'] == plan['execution']['qualifier']['sha256'], 'executed qualifier SHA')
    attempt, child = read_json(owner / 'attempt.json'), read_json(owner / 'child.json')
    owner_pid = owner_complete['owner_pid']
    need(attempt['owner_pid'] == child['owner_pid'] == lease_data['child_pid'] == owner_pid, 'lease/owner/child chain')
    release = read_json(owner / 'release.json')
    need(release == {'pid': child['pid'], 'returncode': -15 if resource_abort else 0, 'waited_reaped': True, 'strict_absence': True}, 'child release/reap failed')
    job, resume = read_json(lease / 'job-0.json'), read_json(lease / 'resume.json')
    need(job['pid'] == owner_pid and job['returncode'] == int(resource_abort) and job['waited'] is True and job['command'] == lease_data['jobs'][0] and len(lease_data['jobs']) == 1, 'outer job wait/exit proof')
    need(lease_data['service_pid'] == 74384 and lease_data['paused_at'] < resume['time'] <= lease_data['deadline'] and resume['identity_match'] is True
         and 'T' in resume['state_before'] and resume['state_after'] and 'T' not in resume['state_after'], 'authorized service restoration failed')
    fixtures, config, pins = audit_pins(owner, plan, attempt)
    pins['qualifier_final_model_source_recheck_receipt'] = qualified.get('source_model_pins_rechecked', False)
    eos = fixtures[0]['eos_token_ids']
    need(all(f['eos_token_ids'] == eos and f['normal_eos'] is True and f['max_output_tokens'] == 512 for f in fixtures), 'fixed fixture EOS/cap')
    resources = audit_resources(owner, child['pid'], owner_pid, owner_complete, resource_abort)
    ledger = stream_ledger(owner / 'qualification/steps.jsonl', fixtures, config, eos, qualified['cases'], partial=resource_abort)
    expected_counts = [8, 16, 13][:len(ledger['cases'])] if resource_abort else [8, 16, 13]
    need([len(c['completed']) for c in ledger['cases']] == expected_counts, '8+16+13 completed counts/prefix')
    return {'status': 'RESOURCE_ABORT' if resource_abort else 'PASS',
            'numerical_status': 'INCOMPLETE_NUMERICAL_NOT_QUALIFIED' if resource_abort else 'PASS',
            'qualified': not resource_abort, 'audit_evidence_consistent': True,
            'plan_sha256': PLAN_SHA, 'pins': pins, 'resources': resources, 'ledger': ledger,
            'owner_status': owner_complete['status'], 'owner_elapsed_wall_s': owner_complete['elapsed_wall_s'],
            'release': release, 'service_restore': resume, 'ledger_opened': True,
            'scope': 'Fixed E023 numerical traces only; completed subcases and checked prefixes are recorded. Not HTTP, performance or E024 long graph qualification.'}


def self_test():
    """Synthetic parser/ordinal/EOS checks; no existing ledger or model files."""
    fixtures = [{'fixture_id': 'f' + str(i), 'prompt_ids': [1, 2]} for i in range(8)]
    config = {'vocab_size': 100, 'layer_types': ['linear_attention'] * 64}
    c = CaseAudit('lifecycle', fixtures, config, [99])
    c.step = 1; c.admit()
    need(len(c.admissions) == 8, 'initial admission self-test')
    c.requests.clear(); c.waiting.clear(); c.step = 2; c.admit()
    need(len(c.admissions) == 12, 'step2 admission self-test')
    c.requests.clear(); c.waiting.clear(); c.step = 3; c.admit()
    need(c.admissions[-1]['rid'] == 'probe-0' and c.admissions[-1]['ordinal'] == 12
         and c.requests['probe-0']['job']['fixture']['fixture_id'] == 'f1', 'rid reuse must use new ordinal/fixture')
    c = CaseAudit('D-C1', fixtures, config, [99]); c.step = 1; c.admit()
    schedule = c.schedule()
    need(schedule['prefill'] == [['c1-0', 2]], 'independent scheduler')
    need(len(c.cache_names()) == len(c.cache_names(single=True)) == 128, 'full64layer/128array accounting')
    call = ['prefill', 'c1-0', 0, 2]
    label = repr(tuple(call))
    def array(check, name, shape, dtype='mlx.core.bfloat16'):
        return {'check': check, 'name': name, 'shape': shape, 'dtype': dtype,
                'reference_finite': True, 'candidate_finite': True, 'bit_equal': True}
    arrays = [array('logits ' + label, 'root', [1, 100])]
    for check, prefix in (('call-state ' + label, 'root'), ('before-retirement-state', 'root/staging/c1-0')):
        for i in range(64):
            arrays.append(array(check, f'{prefix}/{i}/conv_state', [3, 10240]))
            arrays.append(array(check, f'{prefix}/{i}/S', [48, 128, 128], 'mlx.core.float32'))
    no_ahead = {'carried_future': None, 'consumed_prior_future': None, 'discarded_prior_future': None, 'discarded_current_calls': []}
    row = {'case': 'D-C1', 'step': 1, 'status': 'PASS', 'schedule': schedule,
        'reference_discarded_ahead': None, 'reference_delivered': [['c1-0', 99, True]],
        'candidate_delivered': [['c1-0', 99, True]], 'call_keys': [call], 'arrays': arrays,
        'ahead': {'reference': no_ahead, 'candidate': no_ahead},
        'after': {'engine': {'requests': [], 'waiting': [], 'running': [], 'free_slots': [0],
            'finished_pending': ['c1-0'], 'batch_rids': [], 'batch_lens': [], 'has_work': False, 'ahead_key': None},
            'state': {'committed': [None] * 64, 'staging': {}, 'feats': {}, 'ahead': None}}}
    valid = CaseAudit('D-C1', fixtures, config, [99]); valid.accept(row)
    need(valid.completed[0]['raw_ids'] == [99] and valid.completed[0]['committed_ids'] == [], 'terminal EOS counted as committed')
    rejected = 0
    for mutation in ('bit', 'finite', 'missing-layer', 'vocabulary', 'cross-rid', 'skip-step', 'early-done', 'retirement', 'single-S-batch', 'single-conv-batch'):
        bad = copy.deepcopy(row)
        if mutation == 'bit': bad['arrays'][0]['bit_equal'] = False
        if mutation == 'finite': bad['arrays'][0]['candidate_finite'] = False
        if mutation == 'missing-layer': bad['arrays'].pop()
        if mutation == 'vocabulary': bad['arrays'][0]['shape'][-1] = 99
        if mutation == 'cross-rid': bad['candidate_delivered'][0][0] = 'c1-1'
        if mutation == 'skip-step': bad['step'] = 2
        if mutation == 'early-done':
            bad['candidate_delivered'][0][1] = bad['reference_delivered'][0][1] = 7
        if mutation == 'retirement': bad['after']['engine']['batch_rids'] = ['c1-0']
        if mutation == 'single-S-batch': bad['arrays'][2]['shape'].insert(0, 1)
        if mutation == 'single-conv-batch': bad['arrays'][1]['shape'].insert(0, 1)
        try:
            CaseAudit('D-C1', fixtures, config, [99]).accept(bad)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('corruption accepted: ' + mutation)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        need(audit(root, root)['status'] == 'INCOMPLETE', 'unfinished must never grant PASS')
        path = root / 'test.jsonl'; path.write_bytes(b'{"case":"D-C1","step":1}')
        try:
            read_json(path)
        except ValueError:
            pass
        else:
            raise AssertionError('wholesale JSONL read accepted')
        try:
            stream_ledger(path, fixtures, config, [99], [])
        except ValueError as error:
            need('truncated' in str(error), 'partial line must fail')
        else:
            raise AssertionError('partial stream passed')
        prefix = [{'event': 'load-pair-start', 'max_seqs': 1},
                  {'event': 'load-pair-complete', 'independent_weight_arrays': 1349}, row]
        path.write_bytes(b''.join((json.dumps(x) + '\n').encode() for x in prefix) + b'{"case":')
        partial = stream_ledger(path, fixtures, config, [99], [], partial=True)
        need(partial['completed_step_rows'] == 1 and partial['complete_lines'] == 3
             and partial['truncated_final_line']['bytes'] == 8 and not partial['cases']
             and partial['partial_cases'][0]['status'] == 'INCOMPLETE_NUMERICAL_NOT_QUALIFIED', 'partial prefix/truncation accounting')
        path.write_bytes(b''.join((json.dumps(x) + '\n').encode() for x in prefix) + b'{"case":\n')
        try:
            stream_ledger(path, fixtures, config, [99], [], partial=True)
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError('newline-delimited malformed JSON accepted as truncation')
    print(f'PASS: full synthetic step + {rejected} corruptions rejected; ordinal/rid reuse, scheduler, EOS, INCOMPLETE and streaming checks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', type=Path)
    parser.add_argument('--lease', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if not all((args.owner, args.lease, args.out)):
        parser.error('--owner, --lease and --out required')
    with args.out.open('x') as output:
        try:
            result = audit(args.owner.resolve(), args.lease.resolve())
        except Exception as error:
            result = {'status': 'FAIL', 'error': str(error), 'traceback': traceback.format_exc()}
        result['audit_source_sha256'] = sha(__file__)
        json.dump(result, output, indent=2); output.write('\n')
    print(json.dumps({'status': result['status'], 'out': str(args.out)}))
    return {'PASS': 0, 'INCOMPLETE': 2, 'RESOURCE_ABORT': 2, 'FAIL': 1}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
