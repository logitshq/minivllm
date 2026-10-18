"""E023 fixed-schedule native numerical qualification; NOT a performance harness.

Only stdlib is imported until main() has validated CLI/plan/pins. CLI:
  python qualify_native.py --plan PLAN.json --plan-sha256 SHA --out NEW_DIRECTORY

Plan schema E023-native-qualification-plan-v1 requires package_root, model,
fixtures/source_manifest/model_manifest (each {path, sha256}), configuration
{max_len, max_seqs_by_case:{'D-C1':1,'D-C8':8,'lifecycle':8},
 max_batched_tokens, cache_limit_gib}, candidate
{compile_decode, fuse_decode, gates_compile, norm_reuse} (bools), deadline_s.
Source manifest files are relative to package_root/tutorial_code/miniengine;
model manifest files are relative to model, and must cover all safetensors and
config.json/generation_config.json. Manifest payloads are {files:{path:sha256}}.
model manifest additionally requires schema='E023-model-v1', model (absolute
path), and revision. The same model pins are checked before and after, never
once per request/step. Two normal engines load for C1, are fully released, then
two normal engines load for C8 and lifecycle. Load/release work is inside the
30-minute absolute budget; at most two target models are retained at once.

Outputs: plan.json; steps.jsonl (schedule, delivered IDs, state metadata, every
array shape/dtype/finite/bit result and semantic call pairing); complete.json;
on failure, failure.json and first differing original-dtype tensors in
failure-arrays.safetensors with SHA. Partial steps survive failure/timeout.
PASS only covers the three fixed discrete traces, not live HTTP or higher C.
The owner must enforce a hard process deadline: Python SIGALRM is also used,
but cannot preempt a native GPU call that never returns to the interpreter.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback


class QualificationFailure(RuntimeError):
    pass


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path, deadline=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 << 20):
            if deadline:
                deadline()
            digest.update(chunk)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, (tuple, list)):
        return [plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if value is None or type(value) in (int, float, str, bool):
        return value
    if hasattr(value, 'name'):
        return value.name
    raise TypeError(f'unsupported metadata: {type(value).__name__}')


def snapshot(value):
    """Independent array handles, retaining original dtype and all padded storage."""
    arrays = {}
    def visit(item, path):
        if hasattr(item, 'dtype') and hasattr(item, 'shape') and hasattr(item, 'view'):
            arrays[path] = item[:] if item.ndim else item.reshape(())
            return {'array': path}
        if isinstance(item, (list, tuple)):
            return [visit(x, f'{path}/{i}') for i, x in enumerate(item)]
        if isinstance(item, dict):
            return {str(k): visit(v, f'{path}/{k}') for k, v in item.items()}
        if item is None or type(item) in (int, float, str, bool):
            return item
        return {'class': type(item).__name__, 'fields': visit(vars(item), path)}
    return {'metadata': visit(value, 'root'), 'arrays': arrays}


def request_metadata(req):
    return {'rid': req.req_id, 'prompt_ids': list(req.prompt_token_ids),
            'output_ids': list(req.output_token_ids), 'computed': req.num_computed_tokens,
            'slot': req.slot, 'status': plain(req.status), 'sampling': plain(vars(req.sampling))}


def state(engine):
    r, s = engine.runner, engine.scheduler
    metadata = {'requests': [request_metadata(x) for x in engine.requests.values()],
        'waiting': [x.req_id for x in s.waiting], 'running': [x.req_id for x in s.running],
        'free_slots': list(s.free_slots), 'finished_pending': list(s._finished_since_last_step),
        'batch_rids': list(r.batch.req_ids), 'batch_lens': list(r.batch.lens),
        'has_work': engine.has_work(), 'ahead_key': plain(r._ahead[0]) if r._ahead is not None else None}
    tensors = snapshot({'committed': r.batch.layers, 'staging': r.staging, 'feats': r.feats,
                        'ahead': None if r._ahead is None else {'tokens': r._ahead[1], 'layers': r._ahead[2]}})
    tensors['metadata'] = {'engine': metadata, 'state': tensors['metadata']}
    return tensors


def schedule_metadata(engine, out):
    return {'decode': [x.req_id for x in out.decode],
            'prefill': [[x.req_id, n] for x, n in out.prefill],
            'finished_ids': list(out.finished_ids),
            'requests': [request_metadata(x) for x in engine.requests.values()]}


def decode_key(rids, positions, tokens):
    return ('decode', tuple(rids), tuple(positions), tuple(tokens))


def ahead_semantic_key(ahead):
    if ahead is None:
        return None
    key = ahead[0]
    if key[3] is None:
        raise QualificationFailure('ahead input IDs were not finalized')
    return decode_key(key[0], key[1], key[3])


class Comparator:
    def __init__(self, mx, out):
        self.mx, self.out = mx, Path(out)
        self.rows = []
        self.first_failure = None

    def fail(self, label, details, arrays=None):
        if self.first_failure is None:
            self.first_failure = {'label': label, 'details': plain(details)}
            if arrays:
                path = self.out / 'failure-arrays.safetensors'
                self.mx.save_safetensors(str(path), arrays)
                self.first_failure['array_artifact'] = {'path': path.name, 'sha256': sha(path)}
            write(self.out / 'failure.json', self.first_failure)
        raise QualificationFailure(label)

    def meta(self, label, left, right):
        # JSON scalars retain bool/int/float distinctions that Python == erases.
        if json.dumps(plain(left), sort_keys=True, allow_nan=False) != json.dumps(plain(right), sort_keys=True, allow_nan=False):
            self.fail(label, {'reference': left, 'candidate': right})

    def compare(self, label, left, right):
        self.meta(label + '/metadata', left['metadata'], right['metadata'])
        self.meta(label + '/array_names', sorted(left['arrays']), sorted(right['arrays']))
        pairs, flags, rows = [], [], []
        mx = self.mx
        for name, a in left['arrays'].items():
            b = right['arrays'][name]
            spec = {'name': name, 'shape': list(a.shape), 'dtype': str(a.dtype)}
            if a.shape != b.shape or a.dtype != b.dtype:
                self.fail(label + '/' + name, {'reference': spec,
                    'candidate': {'shape': list(b.shape), 'dtype': str(b.dtype)}}, {'reference': a, 'candidate': b})
            if a.dtype in (mx.bfloat16, mx.float16, mx.int16, mx.uint16):
                unsigned = mx.uint16
            elif a.dtype in (mx.float32, mx.int32, mx.uint32):
                unsigned = mx.uint32
            elif a.dtype in (mx.int64, mx.uint64):
                unsigned = mx.uint64
            elif a.dtype in (mx.int8, mx.uint8, mx.bool_):
                unsigned = mx.uint8
            else:
                self.fail(label, {'unsupported original dtype': str(a.dtype)})
            flags.extend((mx.all(mx.isfinite(a)), mx.all(mx.isfinite(b)), mx.array_equal(a.view(unsigned), b.view(unsigned))))
            pairs.append((a, b))
            rows.append(spec)
        if flags:
            # One small boolean vector crosses to CPU, never cache tensor bytes.
            packed = mx.stack(flags)
            mx.eval(packed)
            values = packed.tolist()
            for index, row in enumerate(rows):
                row.update(reference_finite=bool(values[3 * index]), candidate_finite=bool(values[3 * index + 1]),
                           bit_equal=bool(values[3 * index + 2]))
                self.rows.append({'check': label, **row})
                if not all(values[3 * index:3 * index + 3]):
                    a, b = pairs[index]
                    self.fail(label + '/' + row['name'], row, {'reference': a, 'candidate': b})


class CallProxy:
    def __init__(self, target, observer):
        self.target, self.observer = target, observer

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __call__(self, tokens, ctx, **kwargs):
        return self.observer.call(self.target, tokens, ctx, kwargs)


class Observer:
    """A step has current callbacks and at most one carried actual future call."""
    def __init__(self, engine, comparator, label):
        self.engine, self.cmp, self.label = engine, comparator, label
        self.calls, self.previous_future = {}, None
        self.schedule, self.postrun, self.expected_current, self.consumed = None, None, None, None
        self.original_model, self.original_decode = engine.runner.model, engine.runner.decode_model
        engine.runner.model = CallProxy(self.original_model, self)
        engine.runner.decode_model = CallProxy(self.original_decode, self)
        self.original_run, self.original_schedule = engine.runner.run, engine.scheduler.schedule
        self.run_was_instance_attribute = 'run' in vars(engine.runner)
        self.schedule_was_instance_attribute = 'schedule' in vars(engine.scheduler)
        engine.runner.run, engine.scheduler.schedule = self.run, self.schedule_call
        self.expected_schedule = None

    def begin(self, expected_schedule=None):
        self.calls, self.schedule, self.postrun = {}, None, None
        self.expected_current, self.consumed = None, None
        self.expected_schedule = expected_schedule

    def schedule_call(self):
        out = self.original_schedule()
        self.schedule = schedule_metadata(self.engine, out)
        if self.expected_schedule is not None:
            self.cmp.meta('schedule-before-candidate-model', self.expected_schedule, self.schedule)
        return out

    def run(self, out):
        r = self.engine.runner
        if out.decode:
            self.expected_current = decode_key([x.req_id for x in out.decode],
                [x.num_computed_tokens for x in out.decode], [x.output_token_ids[-1] for x in out.decode])
            actual_ahead = ahead_semantic_key(r._ahead)
            sampling = tuple((x.sampling.temperature, x.sampling.top_p, x.sampling.top_k,
                              x.sampling.max_tokens, x.sampling.ignore_eos, tuple(x.sampling.stop_token_ids)) for x in out.decode)
            native_key = ([x.req_id for x in out.decode], [x.num_computed_tokens for x in out.decode],
                          sampling, tuple(x.output_token_ids[-1] for x in out.decode))
            if actual_ahead == self.expected_current and r._ahead[0] == native_key and r.batch.req_ids == native_key[0]:
                self.consumed = actual_ahead
        result = self.original_run(out)
        self.postrun = state(self.engine)  # Before scheduler.update and retirement.
        if self.expected_current is not None:
            record = self.calls.get(self.expected_current)
            if record is not None and self.consumed is not None:
                self.cmp.fail('reported ahead reuse also recomputed the same call', {'arm': self.label, 'key': self.expected_current})
            if record is None:
                if self.consumed != self.expected_current or self.previous_future is None or self.previous_future['key'] != self.consumed:
                    self.cmp.fail('missing actual decode or matching carried ahead', {'arm': self.label, 'key': self.expected_current})
                record = self.previous_future
            # The subsequent speculative call must not rewrite committed storage.
            self.cmp.compare(self.label + '/committed-not-rewritten-by-ahead', record['state'], snapshot(r.batch.layers))
        if r._ahead is not None:
            for current, future in zip(r.batch.layers, r._ahead[2]):
                for name in vars(current):
                    if getattr(current, name) is getattr(future, name):
                        self.cmp.fail('ahead aliases committed array handle', {'arm': self.label, 'field': name})
        return result

    def call(self, target, tokens, ctx, kwargs):
        r = self.engine.runner
        if ctx.mode == 'decode':
            key = decode_key(r.batch.req_ids, ctx.positions.tolist(), tokens.tolist())
        elif ctx.mode == 'prefill':
            owners = [rid for rid, cache in r.staging.items() if cache is ctx.caches]
            if len(owners) != 1:
                self.cmp.fail('prefill cache must have exactly one owner', {'owners': owners})
            req = self.engine.requests[owners[0]]
            key = ('prefill', req.req_id, req.num_computed_tokens, len(tokens))
            self.cmp.meta('prefill exact input IDs', req.all_token_ids[key[2]:key[2] + key[3]], tokens.tolist())
        else:
            self.cmp.fail('unqualified model route', {'mode': ctx.mode})
        if key in self.calls:
            self.cmp.fail('duplicate semantic model call', {'key': key})
        logits = target(tokens, ctx, **kwargs)
        record = {'key': key, 'logits': snapshot(logits), 'state': snapshot(ctx.caches),
                  'context': {'mode': ctx.mode, 'max_len': ctx.max_len, 'last_only': kwargs.get('last_only', False),
                              'lens': ctx.lens.tolist() if ctx.lens is not None else None}}
        self.calls[key] = record
        return logits

    def finish(self):
        key = ahead_semantic_key(self.engine.runner._ahead)
        if key is not None and key not in self.calls:
            self.cmp.fail('ahead state lacks actual future model call', {'arm': self.label, 'key': key})
        future = self.calls.get(key)
        result = {'carried_future': plain(key), 'consumed_prior_future': plain(self.consumed),
                  'discarded_prior_future': plain(self.previous_future['key']) if self.previous_future and self.consumed is None else None,
                  'discarded_current_calls': [plain(k) for k in self.calls if k != key and k != self.expected_current and k[0] == 'decode']}
        self.previous_future = future
        return result

    def restore(self):
        r = self.engine.runner
        r.model, r.decode_model = self.original_model, self.original_decode
        if self.run_was_instance_attribute:
            r.run = self.original_run
        else:
            vars(r).pop('run', None)
        if self.schedule_was_instance_attribute:
            self.engine.scheduler.schedule = self.original_schedule
        else:
            vars(self.engine.scheduler).pop('schedule', None)


def pair_calls(reference, candidate, cmp):
    right = dict(candidate.calls)
    if candidate.consumed is not None and candidate.consumed not in right:
        carried = candidate.previous_future
        if carried is None or carried['key'] != candidate.consumed:
            cmp.fail('consumed ahead has no semantic-key evidence', {'key': candidate.consumed})
        right[candidate.consumed] = carried
    cmp.meta('semantic call key set', sorted(reference.calls), sorted(right))
    for key, left in reference.calls.items():
        other = right[key]
        cmp.meta('call context ' + repr(key), left['context'], other['context'])
        cmp.compare('logits ' + repr(key), left['logits'], other['logits'])
        cmp.compare('call-state ' + repr(key), left['state'], other['state'])
    return [plain(key) for key in reference.calls]


@contextmanager
def grouping_disabled(runner_module):
    original = runner_module.prefill_batch_backend_supported
    runner_module.prefill_batch_backend_supported = lambda *args, **kwargs: False
    try:
        yield
    finally:
        runner_module.prefill_batch_backend_supported = original


def workloads(fixtures):
    if len(fixtures) != 8 or len({x['fixture_id'] for x in fixtures}) != 8:
        raise ValueError('exactly eight unique D fixtures required')
    def job(index, cap, prefix):
        return {'rid': prefix + str(index), 'fixture': fixtures[index % 8], 'cap': cap}
    return [
        {'name': 'D-C1', 'concurrency': 1, 'jobs': [job(i, 512, 'c1-') for i in range(8)], 'arrival': 'refill'},
        {'name': 'D-C8', 'concurrency': 8, 'jobs': [job(i, 512, 'c8-') for i in range(16)], 'arrival': 'refill'},
        {'name': 'lifecycle', 'concurrency': 8, 'jobs': [job(i, cap, 'probe-') for i, cap in enumerate((1, 2, 3, 5, 8, 13, 21, 34))]
         + [job(i + 8, cap, 'probe-') for i, cap in enumerate((2, 5, 9, 17))]
         + [{'rid': 'probe-0', 'fixture': fixtures[1], 'cap': 3}], 'arrival': 'step2_then_reuse'},
    ]


def assert_empty(engine, cmp):
    r = engine.runner
    if engine.has_work() or engine.requests or r.staging or r.feats or r.batch.size or r._ahead is not None:
        cmp.fail('engine failed to drain', state(engine)['metadata'])
    if any(x is not None for x in r.batch.layers):
        cmp.fail('empty batch retained cache layers', {})
    # _finished_since_last_step legitimately retains the last retired IDs.


def run_case(case, engines, observers, sampling, runner_module, cmp, ledger, deadline):
    for engine in engines:
        assert_empty(engine, cmp)
    next_job, step = 0, 0
    active, completed, delivered, admissions = set(), [], {}, []
    decode_histogram = {}
    limit = sum(len(j['fixture']['prompt_ids']) + j['cap'] for j in case['jobs']) + 1
    def add(job):
        nonlocal next_job
        rid = job['rid']
        if rid in active:
            cmp.fail('duplicate active fixture request', {'rid': rid})
        entry = {'ordinal': next_job, 'rid': rid, 'fixture_id': job['fixture']['fixture_id'], 'cap': job['cap']}
        for engine in engines:
            actual = engine.add_request(job['fixture']['prompt_ids'], sampling(max_tokens=job['cap'], temperature=0, ignore_eos=False), req_id=rid)
            cmp.meta('add_request preserved rid', rid, actual)
        admissions.append({**entry, 'before_step': step + 1})
        delivered[next_job] = []
        active.add(rid)
        next_job += 1
    while next_job < len(case['jobs']) or active:
        deadline()
        if step >= limit:
            cmp.fail('finite step limit exceeded', {'case': case['name'], 'limit': limit})
        if case['arrival'] == 'refill':
            while next_job < len(case['jobs']) and len(active) < case['concurrency']:
                add(case['jobs'][next_job])
        elif step == 0:
            while next_job < 8:
                add(case['jobs'][next_job])
        elif step == 1:
            while next_job < 12:
                add(case['jobs'][next_job])
        elif not active and next_job == 12:
            add(case['jobs'][next_job])
        step += 1
        row = {'case': case['name'], 'step': step, 'status': 'RUNNING'}
        cmp.rows = []
        try:
            cmp.compare('before-step-state', state(engines[0]), state(engines[1]))
            before_progress = [(r.req_id, r.num_computed_tokens, len(r.output_token_ids)) for r in engines[0].requests.values()]
            reference, candidate = observers
            reference.begin()
            # Disable only reuse; compare the previous ahead before dropping its handles.
            discarded = ahead_semantic_key(engines[0].runner._ahead)
            engines[0].runner._ahead = None
            with grouping_disabled(runner_module):
                left = engines[0].step()
            candidate.begin(reference.schedule)
            right = engines[1].step()
            row.update(schedule=reference.schedule, reference_discarded_ahead=plain(discarded),
                       reference_delivered=plain(left), candidate_delivered=plain(right))
            B = len(reference.schedule['decode'])
            decode_histogram[B] = decode_histogram.get(B, 0) + 1
            cmp.meta('delivered tuples', left, right)
            row['call_keys'] = pair_calls(reference, candidate, cmp)
            if reference.postrun is None or candidate.postrun is None:
                cmp.fail('step with work lacked runner observation', {})
            cmp.compare('before-retirement-state', reference.postrun, candidate.postrun)
            cmp.compare('after-retirement-state', state(engines[0]), state(engines[1]))
            row['after'] = state(engines[0])['metadata']
            for rid, token, done in left:
                if rid not in active:
                    cmp.fail('output for inactive/retired rid', {'rid': rid})
                entry = next(x for x in reversed(admissions) if x['rid'] == rid)
                ids = delivered[entry['ordinal']]
                ids.append(token)
                if len(ids) > entry['cap'] or any(x in engines[0].eos_ids for x in ids[:-1]):
                    cmp.fail('output after cap or EOS', {'rid': rid, 'ids': ids})
                if done:
                    if token not in engines[0].eos_ids and len(ids) != entry['cap']:
                        cmp.fail('finished without normal EOS or exact cap', {'rid': rid, 'ids': ids})
                    active.remove(rid)
                    completed.append({**entry, 'raw_ids': list(ids), 'committed_ids': ids[:-1] if token in engines[0].eos_ids else list(ids)})
            after_progress = [(r.req_id, r.num_computed_tokens, len(r.output_token_ids)) for r in engines[0].requests.values()]
            if engines[0].has_work() and before_progress == after_progress and not left:
                cmp.fail('work made no computed/output progress', {'case': case['name'], 'step': step})
            row['ahead'] = {'reference': reference.finish(), 'candidate': candidate.finish()}
            row['status'] = 'PASS'
        finally:
            row['arrays'] = cmp.rows
            if row['status'] != 'PASS':
                row['failure'] = cmp.first_failure
            ledger.write(json.dumps(row, ensure_ascii=False) + '\n')
            ledger.flush()
    for engine in engines:
        assert_empty(engine, cmp)
    cmp.meta('final pending retire buffer', engines[0].scheduler._finished_since_last_step, engines[1].scheduler._finished_since_last_step)
    if len(completed) != len(case['jobs']):
        cmp.fail('not every fixed job completed', {'completed': len(completed), 'planned': len(case['jobs'])})
    if case['name'] == 'D-C8' and max(decode_histogram, default=0) < 2:
        cmp.fail('C8 numerical trace did not exercise multi-row decode', decode_histogram)
    return {'name': case['name'], 'steps': step, 'step_limit': limit, 'admissions': admissions,
            'completed': completed, 'decode_batch_histogram': decode_histogram,
            'raw_output_count': sum(len(x['raw_ids']) for x in completed)}


def pinned_json(item):
    if sha(item['path']) != item['sha256']:
        raise ValueError('pinned JSON SHA mismatch: ' + item['path'])
    return json.loads(Path(item['path']).read_text())


def verify_files(root, files, deadline):
    root = Path(root).resolve()
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('manifest path must be relative without traversal')
        if sha(root / path, deadline) != expected:
            raise ValueError('source/model pin mismatch: ' + str(root / path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    plan = pinned_json({'path': str(args.plan), 'sha256': args.plan_sha256})
    if plan.get('schema') != 'E023-native-qualification-plan-v1':
        parser.error('unsupported qualification plan schema')
    budget = plan['deadline_s']
    if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 1800:
        parser.error('deadline_s must be positive and at most 1800 seconds')
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'plan.json', plan)
    started = time.monotonic()
    def deadline(*unused):
        if unused or time.monotonic() - started >= budget:
            raise TimeoutError('absolute numerical qualification deadline')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('an outer Python alarm is already active')
    handler = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, budget)
    result = {'status': 'INCOMPLETE', 'cases': [], 'numeric_only_not_http': True}
    observers, engines = [], []
    ledger = None
    mx = None
    def phase(name, **fields):
        if ledger is not None:
            ledger.write(json.dumps({'event': name, 'elapsed_wall_s': time.monotonic() - started, **fields}) + '\n')
            ledger.flush()
    def release_pair():
        phase('release-pair-start', model_instances=len(engines))
        for observed in observers:
            observed.restore()
            observed.calls.clear()
            observed.previous_future = observed.postrun = None
        if observers:
            del observed
        observers.clear()
        engines.clear()
        gc.collect()
        if mx is not None:
            mx.synchronize()
            mx.clear_cache()
        phase('release-pair-complete', qualification_only=True)
    try:
        source = pinned_json(plan['source_manifest'])['files']
        model_manifest = pinned_json(plan['model_manifest'])
        if (model_manifest.get('schema') != 'E023-model-v1'
                or Path(model_manifest['model']).resolve() != Path(plan['model']).resolve()
                or model_manifest['revision'] != Path(plan['model']).name):
            raise ValueError('model manifest schema/path/revision does not bind the requested model')
        model = model_manifest['files']
        fixtures = pinned_json(plan['fixtures'])
        if Path(fixtures['model']).resolve() != Path(plan['model']).resolve():
            raise ValueError('fixtures and model identity differ')
        root = Path(plan['package_root']).resolve() / 'tutorial_code/miniengine'
        if set(source) != {str(p.relative_to(root)) for p in root.rglob('*.py')}:
            raise ValueError('source manifest does not cover exactly all package Python files')
        required = {'config.json', 'generation_config.json'} | {str(p.relative_to(plan['model'])) for p in Path(plan['model']).rglob('*.safetensors')}
        if not required <= set(model) or not any(x.endswith('.safetensors') for x in required):
            raise ValueError('model manifest must cover config, generation config, and every weight shard')
        verify_files(root, source, deadline)
        verify_files(plan['model'], model, deadline)
        config, candidate = plan['configuration'], plan['candidate']
        if set(config) != {'max_len', 'max_seqs_by_case', 'max_batched_tokens', 'cache_limit_gib'}:
            raise ValueError('explicit capacity/allocator configuration required')
        if set(candidate) != {'compile_decode', 'fuse_decode', 'gates_compile', 'norm_reuse'} or any(type(x) is not bool for x in candidate.values()):
            raise ValueError('explicit candidate boolean flags required')
        if (candidate != {'compile_decode': True, 'fuse_decode': True, 'gates_compile': True, 'norm_reuse': False}
                or config['max_seqs_by_case'] != {'D-C1': 1, 'D-C8': 8, 'lifecycle': 8}
                or config['max_len'] != 2048 or config['cache_limit_gib'] != 2
                or type(config['max_batched_tokens']) is not int or config['max_batched_tokens'] < 8):
            raise ValueError('fixed current candidate, C1/C8 capacities, max_len2048 and cache2GiB required')
        if len(fixtures['fixtures']) != 8 or any(x['max_output_tokens'] != 512 or not x['normal_eos'] or len(x['prompt_ids']) <= 512 or len(x['prompt_ids']) + 512 > 2048 for x in fixtures['fixtures']):
            raise ValueError('expected the frozen eight long D fixtures with normal EOS and cap512')
        sys.path.insert(0, str(Path(plan['package_root']).resolve()))
        import mlx.core as mx
        import tutorial_code.miniengine as package
        from tutorial_code.miniengine.engine.llm_engine import LLMEngine
        from tutorial_code.miniengine.engine import model_runner as runner_module
        from tutorial_code.miniengine.sampling_params import SamplingParams
        from mlx.utils import tree_flatten
        if Path(package.__file__).resolve().parent != root:
            raise ValueError('loaded package root differs from pinned source root')
        cmp = Comparator(mx, args.out)
        def load_pair(max_seqs):
            phase('load-pair-start', max_seqs=max_seqs, model_instances=2)
            for flags in ({'compile_decode': False, 'fuse_decode': False, 'gates_compile': False, 'norm_reuse': False}, candidate):
                deadline()
                engine = LLMEngine(plan['model'], max_len=config['max_len'], max_seqs=max_seqs,
                    max_batched_tokens=config['max_batched_tokens'], cache_limit_gb=config['cache_limit_gib'],
                    dtype=mx.bfloat16, compile_decode=flags['compile_decode'], fuse_decode=flags['fuse_decode'])
                engines.append(engine)
                for layer in engine.model.layers:
                    if not layer.is_attn:
                        layer.linear_attn.set_gates_compile(flags['gates_compile'])
                        layer.linear_attn.set_norm_reuse(False)
                if engine.runner.spec is not None or len(engine.model.layers) != 64:
                    raise ValueError('expected target-only model with 64 layers')
            weights = [dict(tree_flatten(e.model.parameters())) for e in engines]
            cmp.meta('independent weights have same names', sorted(weights[0]), sorted(weights[1]))
            if any(weights[0][key] is weights[1][key] for key in weights[0]):
                raise ValueError('independent engine weights share array handles')
            cmp.meta('EOS set', sorted(engines[0].eos_ids), sorted(engines[1].eos_ids))
            for fixture in fixtures['fixtures']:
                cmp.meta('fixture EOS set', sorted(fixture['eos_token_ids']), sorted(engines[0].eos_ids))
            observers.extend(Observer(e, cmp, arm) for e, arm in zip(engines, ('reference', 'candidate')))
            mx.synchronize()
            phase('load-pair-complete', max_seqs=max_seqs, independent_weight_arrays=len(weights[0]))
        ledger = (args.out / 'steps.jsonl').open('x')
        result['qualifier_sha256'] = sha(__file__)
        result['plan_sha256'] = args.plan_sha256
        for case in workloads(fixtures['fixtures']):
            if case['name'] in {'D-C1', 'D-C8'}:
                if engines:
                    release_pair()
                load_pair(config['max_seqs_by_case'][case['name']])
            result['cases'].append(run_case(case, engines, observers, SamplingParams, runner_module, cmp, ledger, deadline))
            write(args.out / 'complete.json', result)
        result['peak_bytes'] = mx.get_peak_memory()
        release_pair()
        verify_files(root, source, deadline)
        verify_files(plan['model'], model, deadline)
        result.update(status='PASS', total_requests=sum(len(x['completed']) for x in result['cases']),
            source_model_pins_rechecked=True,
            reference='normal engine eager; gates/grouping OFF; previous ahead discarded before each step',
            candidate=candidate, scope='Same discrete schedule only. D decode graph and grouped prefill remain outside their old short-context guards.')
    except BaseException as exc:
        result.update(status='FAIL', error={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
    finally:
        try:
            if engines or observers:
                release_pair()
        except BaseException as exc:
            result.update(status='FAIL', cleanup_error={'type': type(exc).__name__, 'message': str(exc)})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)
            if ledger is not None:
                ledger.close()
        result['elapsed_wall_s'] = time.monotonic() - started
        write(args.out / 'complete.json', result)
    print(json.dumps({'status': result['status'], 'out': str(args.out)}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
