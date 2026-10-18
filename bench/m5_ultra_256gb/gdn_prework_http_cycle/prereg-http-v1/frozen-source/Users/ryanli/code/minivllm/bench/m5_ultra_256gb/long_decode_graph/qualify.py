"""E024 diagnostic numerical qualification, never a performance measurement.

CLI: --plan FILE --plan-sha256 SHA --out NEW_DIRECTORY. Plan schema
E024-diagnostic-qualification-v1 has exactly: schema, qualifier_sha256,
e023_plan {path,sha256}, e023_qualifier {path,sha256}, candidate {path,sha256},
deadline_s (0 < seconds <= 1800). The inherited E023 plan binds model shards,
source tree and eight D fixtures. Only main imports MLX, after validation.

Two independent engines, gates/norm/packed fusion OFF. Thirty-two direct
points: B1..8 x used257/1024/1280/1536. Eight diagnostic prompts repeat frozen
D IDs and are incrementally prefetched to 1535 (12280 tokens per arm).
Twenty-one additional decode calls test changed/restored norm, packed weight,
and encoded scales over B1->B8->B1, retaining the same compiled functions.
Eight length1533/cap4 diagnostic requests reuse E023 run_case/Observer for
full actual committed/ahead/retirement state (12264 more prefill tokens and
at most32 output tokens per arm). Normal EOS may make coverage insufficient.
Actual full D512 HTTP/gates-ON qualification remains mandatory separately.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback


SUPPORT_SHA = '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'
CANDIDATE_SHA = 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84'
MODEL_SOURCE_SHA = '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5'
USED = (257, 1024, 1280, 1536)
PARAMETERS = ('norm.weight', 'layers.0.linear_attn.in_proj_a.weight',
              'layers.0.linear_attn.in_proj_a.scales')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pinned(item, name, expected):
    if set(item) != {'path', 'sha256'} or item['sha256'] != expected or digest(item['path']) != expected:
        raise ValueError('fixed executable source SHA mismatch: ' + name)
    spec = importlib.util.spec_from_file_location(name, Path(item['path']).resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_plan(plan):
    fields = {'schema', 'qualifier_sha256', 'e023_plan', 'e023_qualifier', 'candidate', 'deadline_s'}
    if set(plan) != fields or plan['schema'] != 'E024-diagnostic-qualification-v1':
        raise ValueError('exact E024 diagnostic plan fields required')
    budget = plan['deadline_s']
    if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 1800:
        raise ValueError('absolute deadline must be positive and <=1800 seconds')
    if plan['qualifier_sha256'] != digest(__file__):
        raise ValueError('qualifier source SHA differs from plan')
    return budget


def diagnostic_fixtures(fixtures, length):
    if len(fixtures) != 8 or len({f['fixture_id'] for f in fixtures}) != 8:
        raise ValueError('eight distinct frozen D fixtures required')
    output = []
    for f in fixtures:
        ids = f['prompt_ids']
        if not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError('nonempty integer prompt IDs required')
        repeated = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
        output.append({**f, 'fixture_id': 'diagnostic-repeat-' + f['fixture_id'],
            'prompt_ids': repeated, 'source_fixture_id': f['fixture_id'],
            'diagnostic_only': True, 'normal_eos': True, 'max_output_tokens': 4})
    return output


class WitnessFunctions(dict):
    """Wrap the actual object returned by candidate's real mx.compile call."""
    def __init__(self):
        super().__init__()
        self.invocations = 0
        self.compiled_objects = []

    def __setitem__(self, key, compiled):
        if not callable(compiled):
            raise TypeError('compiler result is not callable')
        self.compiled_objects.append({'key': repr(key), 'type': type(compiled).__name__, 'id': id(compiled)})
        def actual_call(*args):
            self.invocations += 1
            return compiled(*args)
        super().__setitem__(key, actual_call)


class GraphWitness:
    """Qualification-only: ineligible calls are failures, never counted as graph."""
    def __init__(self, graph, emit):
        if graph._functions:
            raise ValueError('witness must attach before first compilation')
        self.graph, self.emit, self.phase = graph, emit, 'unassigned'
        self.functions = WitnessFunctions()
        graph._functions = self.functions
        self.rows = []

    def __call__(self, tokens, ctx):
        positions, lens = ctx.positions.tolist(), ctx.lens.tolist()
        backing = [(c.K.shape[2], c.V.shape[2]) for c in ctx.caches if hasattr(c, 'K')]
        before = self.functions.invocations
        row = {'event': 'graph-call', 'phase': self.phase, 'batch': len(positions),
            'used': ctx.max_len, 'positions': positions, 'lens': lens,
            'tokens': tokens.tolist(), 'KV_backing_lengths': backing, 'status': 'RUNNING'}
        try:
            if (not positions or lens != [p + 1 for p in positions]
                    or ctx.max_len != max(lens) or not backing or min(min(x) for x in backing) < 1536
                    or not self.graph._eligible(tokens, ctx)):
                raise RuntimeError('E024 expected actual eligible graph; fallback is not qualification')
            result = self.graph(tokens, ctx)
            if self.functions.invocations != before + 1:
                raise RuntimeError('eligible call did not invoke exactly one actual compiled callable')
            row.update(status='SUBMITTED', invocation=self.functions.invocations,
                       compiled_function_count=len(self.functions), actual_compile_objects=self.functions.compiled_objects)
            self.rows.append(row)
            return result
        finally:
            # SUBMITTED is routing evidence, not numerical PASS or timing.
            self.emit(row)


def parameter_slot(model, path):
    parts = path.split('.')
    for part in parts[:-1]:
        model = model[int(part)] if part.isdigit() else getattr(model, part)
    return model, parts[-1]


def changed_parameter(mx, index, original):
    if index == 0:
        return (original + mx.array(0.125, dtype=original.dtype)).astype(original.dtype)
    if index == 1:
        if original.dtype != mx.uint32:
            raise ValueError('packed NVFP4 weight must be uint32')
        # Flip only each FP4 sign bit, retaining valid encoded magnitudes.
        return original ^ mx.array(0x88888888, dtype=mx.uint32)
    if original.dtype != mx.uint8:
        raise ValueError('encoded NVFP4 scales must be uint8')
    # Permute existing scale bytes rather than inventing invalid FP8 encodings.
    return mx.concatenate([original[1:], original[:1]], axis=0)


@contextmanager
def parameter_replacement(mx, models, path, index, cmp, snapshot, emit):
    slots = [parameter_slot(model, path) for model in models]
    original = [getattr(obj, name) for obj, name in slots]
    if original[0] is original[1]:
        raise RuntimeError('parameter handles must be independent')
    cmp.compare(path + '/original-values', snapshot(original[0]), snapshot(original[1]))
    try:
        changed = [changed_parameter(mx, index, x) for x in original]
        if changed[0] is changed[1] or any(bool(mx.array_equal(a, b).item()) for a, b in zip(original, changed)):
            raise RuntimeError('parameter change needs independent handles and changed values')
        cmp.compare(path + '/independent-new-values', snapshot(changed[0]), snapshot(changed[1]))
        for (obj, name), value in zip(slots, changed):
            setattr(obj, name, value)
        yield
    finally:
        for (obj, name), value in zip(slots, original):
            setattr(obj, name, value)
        emit({'event': 'parameter-original-handles-restored', 'parameter': path,
              'both': all(getattr(obj, name) is value for (obj, name), value in zip(slots, original))})


def retain_prefill_caches(caches, models, make_cache):
    """Snapshot handles, not cache objects mutated by later incremental prefill."""
    retained = [[make_cache(model) for _ in arm] for model, arm in zip(models, caches)]
    for dst_arm, src_arm in zip(retained, caches):
        for dst, src in zip(dst_arm, src_arm):
            for d, s in zip(dst, src):
                vars(d).update({k: v[:] if hasattr(v, 'dtype') else v for k, v in vars(s).items()})
    return retained


def check_lifecycle_coverage(rows, step_rows):
    calls = [r for r in rows if r['phase'] == 'lifecycle']
    steps = [r for r in step_rows if r.get('case') == 'E024-diagnostic-ahead']
    if (not any(r['batch'] == 8 for r in calls)
            or not {1534, 1535, 1536} <= {r['used'] for r in calls}
            or not any(r.get('ahead', {}).get('candidate', {}).get('consumed_prior_future') is not None for r in steps)
            or not any(r.get('ahead', {}).get('candidate', {}).get('carried_future') is not None for r in steps)):
        raise RuntimeError('normal-EOS diagnostic did not cover required actual B8/ahead produce/consume/boundary')


def qualify_device(plan, inherited, support, candidate_module, out, deadline, emit, result):
    """Imported only by main after plan validation; all tensors remain on device."""
    source = support.pinned_json(inherited['source_manifest'])['files']
    model_manifest = support.pinned_json(inherited['model_manifest'])
    fixtures = support.pinned_json(inherited['fixtures'])
    root = Path(inherited['package_root']).resolve() / 'tutorial_code/miniengine'
    model_path = Path(inherited['model']).resolve()
    if (model_manifest.get('schema') != 'E023-model-v1'
            or Path(model_manifest['model']).resolve() != model_path
            or model_manifest['revision'] != model_path.name
            or Path(fixtures['model']).resolve() != model_path):
        raise ValueError('inherited source/model/fixture identity mismatch')
    if set(source) != {str(p.relative_to(root)) for p in root.rglob('*.py')}:
        raise ValueError('source manifest must cover the exact Python tree')
    expected_source = json.loads(candidate_module.PINS.read_text())['files']
    if source != expected_source or digest(candidate_module.PINS) != MODEL_SOURCE_SHA:
        raise ValueError('candidate and inherited model Python source pins differ')
    required = {'config.json', 'generation_config.json'} | {str(p.relative_to(model_path)) for p in model_path.rglob('*.safetensors')}
    if not required <= set(model_manifest['files']) or not any(p.endswith('.safetensors') for p in required):
        raise ValueError('model manifest lacks all weight shards/configuration')
    support.verify_files(root, source, deadline)
    support.verify_files(model_path, model_manifest['files'], deadline)
    sys.path.insert(0, str(root.parent.parent))
    import mlx.core as mx
    import tutorial_code.miniengine as package
    from tutorial_code.miniengine.cache import make_cache, DecodeBatch
    from tutorial_code.miniengine.context import ForwardContext
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from tutorial_code.miniengine.engine import model_runner as runner_module
    from tutorial_code.miniengine.sampling_params import SamplingParams
    from mlx.utils import tree_flatten
    if Path(package.__file__).resolve().parent != root:
        raise ValueError('mixed actual package root')
    cmp = support.Comparator(mx, out)
    engines, observers, caches, last_logits, retained = [], [], [], [], None
    witness = None
    def flush(label):
        emit({'event': label, 'arrays': cmp.rows})
        cmp.rows = []
    def direct(prefix, B, used, source_caches, tokens):
        deadline()
        witness.phase = prefix
        batches = [DecodeBatch(e.model) for e in engines]
        contexts = []
        for batch, src in zip(batches, source_caches):
            batch.rebuild([], [(f'diagnostic-row-{i}', src[i], 1536) for i in range(B)])
            contexts.append(ForwardContext('decode', batch.layers, positions=mx.array([used - 1] * B),
                lens=mx.array([used] * B), max_len=used))
        cmp.compare(prefix + '/input', support.snapshot(contexts[0]), support.snapshot(contexts[1]))
        left = engines[0].model(mx.array(tokens[:B]), contexts[0])
        right = witness(mx.array(tokens[:B]), contexts[1])
        if len(contexts[0].caches) != 64 or len(support.snapshot(contexts[0].caches)['arrays']) != 128:
            raise RuntimeError('expected all64layers/128cachearrays')
        cmp.compare(prefix + '/logits', support.snapshot(left), support.snapshot(right))
        cmp.compare(prefix + '/all-cache', support.snapshot(contexts[0].caches), support.snapshot(contexts[1].caches))
        cmp.meta(prefix + '/greedy-ids', mx.argmax(left, axis=-1).tolist(), mx.argmax(right, axis=-1).tolist())
        flush(prefix)
        return support.snapshot(left)
    try:
        for arm in ('reference', 'candidate'):
            deadline()
            emit({'event': 'load-model', 'arm': arm})
            e = LLMEngine(str(model_path), max_len=2048, max_seqs=8, max_batched_tokens=16384,
                          cache_limit_gb=2, dtype=mx.bfloat16, compile_decode=False, fuse_decode=False)
            engines.append(e)
            for layer in e.model.layers:
                if not layer.is_attn:
                    layer.linear_attn.set_gates_compile(False)
                    layer.linear_attn.set_norm_reuse(False)
            if e.runner.spec is not None or len(e.model.layers) != 64:
                raise ValueError('expected64layer target-only model')
        weights = [dict(tree_flatten(e.model.parameters())) for e in engines]
        cmp.meta('parameter names', sorted(weights[0]), sorted(weights[1]))
        if any(weights[0][k] is weights[1][k] for k in weights[0]):
            raise RuntimeError('independent models share parameter handles')
        cmp.compare('independent-loaded-weights', support.snapshot(weights[0]), support.snapshot(weights[1]))
        del weights
        flush('loaded-weights')
        cmp.meta('EOS', sorted(engines[0].eos_ids), sorted(engines[1].eos_ids))
        for f in fixtures['fixtures']:
            cmp.meta('fixture EOS', sorted(f['eos_token_ids']), sorted(engines[0].eos_ids))
        graph = candidate_module.LongContextDecode(engines[1].model, enabled=True)
        witness = GraphWitness(graph, emit)
        engines[1].runner.decode_model = witness
        result['model_source_receipt'] = graph.source_receipt
        inputs = diagnostic_fixtures(fixtures['fixtures'], 1535)
        support.write(out / 'diagnostic-inputs.json', inputs)
        caches = [[make_cache(e.model) for _ in range(8)] for e in engines]
        previous, tokens = 0, []
        for used in USED:
            end = used - 1
            tokens = []
            for row, f in enumerate(inputs):
                deadline()
                last_logits = [e.model(mx.array(f['prompt_ids'][previous:end]), ForwardContext('prefill', arm[row]), last_only=True)
                               for e, arm in zip(engines, caches)]
                cmp.compare(f'prefill-{end}-{row}/logits', support.snapshot(last_logits[0]), support.snapshot(last_logits[1]))
                cmp.compare(f'prefill-{end}-{row}/state', support.snapshot(caches[0][row]), support.snapshot(caches[1][row]))
                tokens.append(int(mx.argmax(last_logits[0], axis=-1).item()))
                flush(f'prefill-{end}-{row}')
            old = [support.snapshot(arm) for arm in caches]
            for B in range(1, 9):
                direct(f'matrix-B{B}-used{used}', B, used, caches, tokens)
                result['matrix'].append({'B': B, 'used': used, 'status': 'PASS'})
            for arm, saved in zip(caches, old):
                cmp.compare('prefill-input-not-rewritten-by-matrix', saved, support.snapshot(arm))
            flush('matrix-input-ownership')
            del old
            if used == 1024:
                # Retain independent wrapper snapshots of original checkpoint;
                # later prefill updates must not replace these saved handles.
                retained = retain_prefill_caches(caches, [e.model for e in engines], make_cache)
                retained_tokens = list(tokens)
            previous = end
        caches.clear()
        last_logits.clear()
        baseline = {}
        for B in (1, 8, 1):
            baseline[B] = direct(f'parameters-baseline-B{B}', B, 1024, retained, retained_tokens)
        for index, path in enumerate(PARAMETERS):
            with parameter_replacement(mx, [e.model for e in engines], path, index, cmp, support.snapshot, emit):
                for B in (1, 8, 1):
                    output = direct(f'parameters-changed-{path}-B{B}', B, 1024, retained, retained_tokens)
                    a, b = output['arrays']['root'], baseline[B]['arrays']['root']
                    if bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()):
                        raise RuntimeError('eager logits did not change after parameter value replacement')
            for B in (1, 8, 1):
                output = direct(f'parameters-restored-{path}-B{B}', B, 1024, retained, retained_tokens)
                cmp.compare(path + '/restored-original-output', baseline[B], output)
                flush('parameter-restore-exact')
            result['parameter_replacements'].append({'path': path, 'changed_eager_logits': True, 'restored_exact': True})
        retained, baseline, output = None, None, None
        diagnostic = diagnostic_fixtures(fixtures['fixtures'], 1533)
        case = {'name': 'E024-diagnostic-ahead', 'concurrency': 8, 'arrival': 'refill',
            'jobs': [{'rid': f'ahead-{i}', 'fixture': f, 'cap': 4} for i, f in enumerate(diagnostic)]}
        witness.phase = 'lifecycle'
        observers = [support.Observer(e, cmp, arm) for e, arm in zip(engines, ('reference', 'candidate'))]
        with (out / 'steps.jsonl').open('x') as ledger:
            result['lifecycle'] = support.run_case(case, engines, observers, SamplingParams, runner_module, cmp, ledger, deadline)
        steps = [json.loads(line) for line in (out / 'steps.jsonl').read_text().splitlines()]
        check_lifecycle_coverage(witness.rows, steps)
        if {(r['B'], r['used']) for r in result['matrix']} != {(B, used) for B in range(1, 9) for used in USED}:
            raise RuntimeError('incomplete fixed matrix')
        result.update(actual_graph_calls=len(witness.rows), compiled_objects=witness.functions.compiled_objects,
                      peak_bytes=mx.get_peak_memory(), actual_ahead_coverage=True)
    finally:
        for observer in observers:
            observer.restore()
            observer.calls.clear()
            observer.previous_future = observer.postrun = None
        observers.clear()
        if witness is not None:
            witness.graph._functions.clear()
        caches.clear()
        last_logits.clear()
        retained = None
        engines.clear()
        # Local scope also owns temporary model/tensor references until return.
        # Final synchronize/cache cleanup happens after this frame is released.
    support.verify_files(root, source, deadline)
    support.verify_files(model_path, model_manifest['files'], deadline)
    for field in ('source_manifest', 'model_manifest', 'fixtures'):
        support.pinned_json(inherited[field])
    if digest(candidate_module.PINS) != MODEL_SOURCE_SHA:
        raise RuntimeError('candidate model source manifest changed during run')
    result['source_model_pins_rechecked'] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if digest(args.plan) != args.plan_sha256:
        parser.error('plan SHA mismatch')
    plan = json.loads(args.plan.read_text())
    budget = validate_plan(plan)
    support = load_pinned(plan['e023_qualifier'], 'e024_pinned_e023_qualifier', SUPPORT_SHA)
    candidate = load_pinned(plan['candidate'], 'e024_pinned_candidate', CANDIDATE_SHA)
    inherited = support.pinned_json(plan['e023_plan'])
    if inherited['schema'] != 'E023-native-qualification-plan-v1':
        parser.error('expected pinned E023 numerical plan')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('outer Python alarm already active')
    args.out.mkdir(parents=True, exist_ok=False)
    support.write(args.out / 'plan.json', plan)
    support.write(args.out / 'inherited-plan.json', inherited)
    started = time.monotonic()
    def deadline(*unused):
        if unused or time.monotonic() - started >= budget:
            raise TimeoutError('E024 absolute numerical qualification deadline')
    old = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, budget)
    result = {'status': 'INCOMPLETE', 'diagnostic_only_not_full_D512': True,
        'numeric_only_not_http_or_performance': True, 'matrix': [], 'parameter_replacements': [],
        'gates_compile': False, 'norm_reuse': False, 'fuse_decode': False,
        'max_prompt_tokens_per_arm': 24544, 'max_lifecycle_output_tokens_per_arm': 32}
    with (args.out / 'events.jsonl').open('x') as ledger:
        def emit(row):
            ledger.write(json.dumps({**row, 'elapsed_wall_s': time.monotonic() - started}) + '\n')
            ledger.flush()
        try:
            emit({'event': 'fixed-diagnostic-contract', 'matrix': [[B, used] for used in USED for B in range(1, 9)],
                  'parameter_paths': PARAMETERS, 'parameter_batch_sequence': [1, 8, 1],
                  'configuration': {'max_len': 2048, 'max_seqs': 8, 'max_batched_tokens': 16384,
                                    'cache_limit_gib': 2, 'gates_compile': False, 'norm_reuse': False,
                                    'compile_decode_core': False, 'fuse_decode': False},
                  'direct_graph_backing': 1536, 'lifecycle_prompt_length': 1533, 'lifecycle_cap': 4,
                  'normal_eos': True, 'source_sha256': {'qualify.py': digest(__file__),
                  'candidate.py': CANDIDATE_SHA, 'E023_qualify_native.py': SUPPORT_SHA}})
            qualify_device(plan, inherited, support, candidate, args.out, deadline, emit, result)
            for item, expected in ((plan['candidate'], CANDIDATE_SHA), (plan['e023_qualifier'], SUPPORT_SHA)):
                if digest(item['path']) != expected:
                    raise RuntimeError('qualification executable changed during run')
            if digest(__file__) != plan['qualifier_sha256']:
                raise RuntimeError('qualifier changed during run')
            support.pinned_json(plan['e023_plan'])
            if digest(args.plan) != args.plan_sha256:
                raise RuntimeError('plan changed during run')
            result['status'] = 'PASS'
        except BaseException as exc:
            result.update(status='FAIL', error={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        finally:
            try:
                gc.collect()
                mx = sys.modules.get('mlx.core')
                if mx is not None:
                    mx.synchronize()
                    mx.clear_cache()
                    result['active_bytes_after_release'] = mx.get_active_memory()
            except BaseException as exc:
                result.update(status='FAIL', cleanup_error={'type': type(exc).__name__, 'message': str(exc)})
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)
                result['elapsed_wall_s'] = time.monotonic() - started
                support.write(args.out / 'complete.json', result)
    print(json.dumps({'status': result['status'], 'out': str(args.out)}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
