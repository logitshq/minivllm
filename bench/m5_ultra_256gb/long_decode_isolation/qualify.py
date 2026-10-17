"""E024 four-arm dispatch isolation, numerical diagnostic only.

CPU-safe help; device CLI: --plan FILE --plan-sha256 SHA --out NEW.
Exact plan: schema E024-dispatch-isolation-plan-v1, qualifier_sha256,
e024_plan {path,sha256}, deadline_s (positive <=1800). The original E024/E023
plans and sources remain unchanged. A=eager257, B=eager1536, C=compiled257,
D=original compiled1536. Only B1/used257; no scheduler, HTTP or timing claim.
PASS means complete finite diagnostic and unchanged sources/inputs/parameters,
NOT numerical equivalence: numerical_equal and comparisons report differences.
The external owner must enforce original full-model resource/reap constraints.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'long_decode_graph'
OLD_SHA = 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7'
OLD_PLAN_SHA = '5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
ARMS = (('A', 'eager', 257), ('B', 'eager', 1536), ('C', 'compiled', 257), ('D', 'compiled', 1536))
PAIRS = (('A', 'B'), ('A', 'C'), ('B', 'D'), ('A', 'D'))


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path, deadline=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 << 20): deadline(); h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load(path, name, expected):
    need(sha(path) == expected, 'fixed support source SHA: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixed_plan():
    return dict(schema='E024-dispatch-isolation-contract-v1', batch=1, used=257,
        prefill_tokens=256, KV_backing=1536, states=128, layers=64,
        arms=[dict(name=n, mode=m, bound=b) for n,m,b in ARMS],
        comparisons=['-'.join(p) for p in PAIRS],
        configuration=dict(max_len=2048,max_seqs=8,max_batched_tokens=16384,
            cache_limit_gib=2,compile_decode=False,fuse_decode=False,gates_compile=False,norm_reuse=False),
        shared_parameter_model=True, independent_evaluated_cache_storage=True,
        numerical_differences_continue=True, execution_or_invariant_failures_stop=True,
        SDPA_route='source-predicted only; no runtime kernel trace',
        no_fuse_used=False, full_model_qualified=False, performance_qualified=False)


def validate(plan):
    need(set(plan) == {'schema','qualifier_sha256','e024_plan','deadline_s'}
         and plan['schema'] == 'E024-dispatch-isolation-plan-v1', 'exact isolation plan required')
    need(plan['qualifier_sha256'] == sha(__file__), 'isolation source SHA')
    need(type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s'])
         and 0 < plan['deadline_s'] <= 1800, 'bounded deadline')
    item = plan['e024_plan']
    need(set(item) == {'path','sha256'} and item['sha256'] == OLD_PLAN_SHA
         and Path(item['path']).is_absolute() and sha(item['path']) == OLD_PLAN_SHA, 'original E024 plan binding')
    old = load(OLD/'qualify.py', 'e024_isolation_original', OLD_SHA)
    original = json.loads(Path(item['path']).read_text()); old.validate_plan(original)
    support = old.load_pinned(original['e023_qualifier'], 'e024_isolation_support', old.SUPPORT_SHA)
    candidate = old.load_pinned(original['candidate'], 'e024_isolation_candidate', old.CANDIDATE_SHA)
    inherited = support.pinned_json(original['e023_plan'])
    return old, support, candidate, original, inherited


def compact_manifest(path, arrays, finite, deadline):
    """Hash exact saved original-dtype payloads, without loading a whole file."""
    sizes = {'BF16':2,'F32':4,'I32':4}
    tensors = {}; cursor = 0
    with Path(path).open('rb') as stream:
        header_size, = struct.unpack('<Q', stream.read(8))
        need(0 < header_size <= 8 << 20, 'safetensors header bounds')
        header = json.loads(stream.read(header_size))
        need(header.pop('__metadata__', None) in (None, {}), 'unexpected tensor file metadata')
        need(set(header) == set(arrays), 'saved exact array keys')
        for name, desc in sorted(header.items(), key=lambda x:x[1]['data_offsets'][0]):
            deadline(); lo, hi = desc['data_offsets']; shape = desc['shape']; encoding = desc['dtype']
            need(encoding in sizes and shape == list(arrays[name].shape)
                 and lo == cursor and hi-lo == math.prod(shape)*sizes[encoding], 'saved shape/offset/dtype')
            h = hashlib.sha256(); remaining = hi-lo
            while remaining:
                deadline(); block = stream.read(min(4 << 20, remaining))
                need(bool(block), 'short tensor payload'); h.update(block); remaining -= len(block)
            tensors[name] = dict(shape=shape,dtype=str(arrays[name].dtype),encoding=encoding,
                bytes=hi-lo,sha256=h.hexdigest(),finite=finite[name])
            cursor = hi
        need(stream.read(1) == b'', 'trailing tensor bytes')
    return tensors


def dump(mx, out, relative, snapshot, deadline):
    arrays = snapshot['arrays']; deadline(); mx.eval(*arrays.values())
    finite = dict(zip(arrays, mx.stack([mx.all(mx.isfinite(a)) for a in arrays.values()]).tolist()))
    path = out/relative; path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), arrays)
    descriptor = dict(path=relative,sha256=sha(path,deadline),finite=all(finite.values()),
        tensors=compact_manifest(path,arrays,finite,deadline),metadata=snapshot['metadata'])
    write(path.with_suffix('.json'), descriptor)
    need(descriptor['finite'], 'nonfinite saved tensors: ' + relative)
    return descriptor


def parameter_hashes(mx, model, flatten, deadline):
    """Full original bytes of every parameter, no numeric cast or weight archive."""
    result = {}
    for name, value in flatten(model.parameters()):
        deadline(); mx.eval(value)
        unsigned = {1:mx.uint8,2:mx.uint16,4:mx.uint32,8:mx.uint64}[value.itemsize]
        bits = mx.contiguous(value).view(unsigned); mx.eval(bits)
        result[name] = dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,
            sha256=hashlib.sha256(memoryview(bits).cast('B')).hexdigest(),
            finite=bool(mx.all(mx.isfinite(value)).item()),handle_id=id(value))
        need(result[name]['finite'], 'nonfinite parameter: ' + name)
        del bits
    return result


def parameter_values(rows):
    # Compilation may restore equivalent array descriptors. Record identities,
    # but the contract is exact values/shape/dtype/finite, not Python object IDs.
    return {name:{k:v for k,v in row.items() if k != 'handle_id'} for name,row in rows.items()}


def make_bound257(model, runtime):
    """Original explicit-state forward, only the context bound is 257."""
    r = runtime
    layout = tuple(layer.is_attn for layer in model.layers)
    bound = 257
    def forward(ids, positions, lens, states):
        caches = [r.BatchKV(s[0][:], s[1][:]) if attn else r.BatchState(*s)
                  for attn, s in zip(layout, states)]
        logits = model(ids, r.ForwardContext('decode', caches,
            positions=positions, lens=lens, max_len=bound))
        return logits, r.states(caches)
    return r.mx.compile(forward, inputs=model)


def context_from_input(arrays, layout, runtime, bound):
    r = runtime; caches = []
    for i, attn in enumerate(layout):
        prefix = f'root/state/{i}/'
        # These are independent descriptors over this arm's own evaluated storage.
        # BatchKV.__setitem__ may replace its descriptor; source handles stay fixed.
        caches.append(r.BatchKV(arrays[prefix+'K'][:], arrays[prefix+'V'][:]) if attn else
                      r.BatchState(arrays[prefix+'conv'][:], arrays[prefix+'S'][:]))
    return r.ForwardContext('decode', caches,positions=arrays['root/positions'][:],
        lens=arrays['root/lens'][:],max_len=bound)


def compare_finite_outputs(support, mx, out, label, left, right):
    """Keep the frozen Comparator's checks; catch ONLY its finite bit differences."""
    class DifferenceRecorder(support.Comparator):
        def fail(self, name, details, arrays=None):
            numeric = (type(details) is dict and details.get('reference_finite') is True
                and details.get('candidate_finite') is True and details.get('bit_equal') is False)
            try: super().fail(name, details, arrays)
            except support.QualificationFailure:
                if not numeric: raise
    directory = out/'comparisons'/label; directory.mkdir(parents=True)
    cmp = DifferenceRecorder(mx, directory)
    cmp.compare(label,left,right)
    return dict(label=label,status='PASS' if cmp.first_failure is None else 'DIFFERENT',
        numerical_equal=cmp.first_failure is None,arrays=cmp.rows,first_difference=cmp.first_failure)


def device(plan, old, support, candidate, original, inherited, out, deadline, emit, result):
    source = support.pinned_json(inherited['source_manifest'])['files']
    manifest = support.pinned_json(inherited['model_manifest'])
    fixtures = support.pinned_json(inherited['fixtures'])
    root = Path(inherited['package_root']).resolve()/'tutorial_code/miniengine'
    model_path = Path(inherited['model']).resolve()
    need(manifest['schema'] == 'E023-model-v1' and Path(manifest['model']).resolve() == model_path
         and manifest['revision'] == model_path.name and Path(fixtures['model']).resolve() == model_path,
         'model/fixture manifest identity')
    need(set(source) == {str(p.relative_to(root)) for p in root.rglob('*.py')}
         and source == json.loads(candidate.PINS.read_text())['files']
         and sha(candidate.PINS) == old.MODEL_SOURCE_SHA, 'exact frozen core tree')
    required = {'config.json','generation_config.json'} | {str(p.relative_to(model_path)) for p in model_path.rglob('*.safetensors')}
    need(required <= set(manifest['files']) and any(x.endswith('.safetensors') for x in required), 'all model shard pins')
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    environment = {k:v for k,v in os.environ.items() if k.startswith('MLX_')}
    need(environment == inherited['execution']['environment'], 'exact inherited clean MLX environment')
    result['mlx_environment'] = environment
    sys.path.insert(0,str(root.parent.parent))
    import mlx.core as mx
    import tutorial_code.miniengine as package
    from tutorial_code.miniengine.cache import make_cache, DecodeBatch
    from tutorial_code.miniengine.context import ForwardContext
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from mlx.utils import tree_flatten
    need(Path(package.__file__).resolve().parent == root, 'actual package path')
    result['device_run'] = True
    result['device'] = dict(name=mx.device_info().get('device_name'),architecture=mx.metal.device_info().get('architecture'),
                            mlx_version=importlib.metadata.version('mlx'))
    need(result['device'] == dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3')
         and mx.default_device() == mx.gpu, 'fixed actual M5/runtime')
    def phase(name):
        deadline(); mx.synchronize()
        emit(dict(event='memory',phase=name,active_bytes=mx.get_active_memory(),cache_bytes=mx.get_cache_memory(),
                  peak_bytes=mx.get_peak_memory(),observation_not_perf=True))
    engines = []; seeds = []; prefills = []; graph = witness = compiled = None
    inputs = {}; outputs = {}; contexts = {}; cmp = support.Comparator(mx,out)
    try:
        for arm in ('reference','independent-prefill'):
            phase('before-load-'+arm)
            engine = LLMEngine(str(model_path),max_len=2048,max_seqs=8,max_batched_tokens=16384,
                cache_limit_gb=2,dtype=mx.bfloat16,compile_decode=False,fuse_decode=False)
            engines.append(engine)
            for layer in engine.model.layers:
                if not layer.is_attn: layer.linear_attn.set_gates_compile(False); layer.linear_attn.set_norm_reuse(False)
            need(engine.runner.spec is None and len(engine.model.layers) == 64, 'normal full64 model')
        weights = [dict(tree_flatten(e.model.parameters())) for e in engines]
        need(not any(weights[0][k] is weights[1][k] for k in weights[0]), 'independent initial models')
        cmp.compare('independent-loaded-weights',support.snapshot(weights[0]),support.snapshot(weights[1]))
        emit(dict(event='loaded-weights',checks=cmp.rows)); cmp.rows = []; del weights
        before = parameter_hashes(mx,engines[0].model,tree_flatten,deadline)
        write(out/'parameters-before.json',before)
        result['parameters_before'] = dict(path='parameters-before.json',sha256=sha(out/'parameters-before.json'))
        for f in fixtures['fixtures']: cmp.meta('fixture EOS',sorted(f['eos_token_ids']),sorted(engines[0].eos_ids))
        cmp.meta('model EOS',sorted(engines[0].eos_ids),sorted(engines[1].eos_ids))
        fixture = old.diagnostic_fixtures(fixtures['fixtures'],256)[0]
        write(out/'diagnostic-input.json',fixture)
        for e in engines:
            caches = make_cache(e.model); seeds.append(caches)
            prefills.append(e.model(mx.array(fixture['prompt_ids'],dtype=mx.int32),ForwardContext('prefill',caches),last_only=True))
        cmp.compare('verified-prefill-logits',support.snapshot(prefills[0]),support.snapshot(prefills[1]))
        cmp.compare('verified-prefill-state',support.snapshot(seeds[0]),support.snapshot(seeds[1]))
        result['prefill_checks'] = cmp.rows; cmp.rows = []
        token = int(mx.argmax(prefills[0],axis=-1).item())
        model = engines[0].model; batch = DecodeBatch(model)
        batch.rebuild([], [('diagnostic-row-0',seeds[0],1536)])
        canonical = support.snapshot(dict(state=batch.layers,tokens=mx.array([token],dtype=mx.int32),
            positions=mx.array([256],dtype=mx.int32),lens=mx.array([257],dtype=mx.int32)))
        need(len(canonical['arrays']) == 131, 'canonical full128 state +3 input arrays')
        result['canonical'] = dump(mx,out,'canonical.safetensors',canonical,deadline)
        # All four arms use the same immutable model. Release independent loader/prefill first.
        seeds.clear(); prefills.clear(); engines.pop(); del e, engine, layer, caches, batch, canonical
        gc.collect(); mx.synchronize(); mx.clear_cache(); phase('after-verified-prefill-release')
        after_prefill = parameter_hashes(mx,model,tree_flatten,deadline)
        write(out/'parameters-after-prefill.json',after_prefill)
        result['parameters_after_prefill'] = dict(path='parameters-after-prefill.json',sha256=sha(out/'parameters-after-prefill.json'))
        cmp.meta('prefill/unchanged-model-parameters',parameter_values(before),parameter_values(after_prefill)); del after_prefill
        graph = candidate.LongContextDecode(model,enabled=True); runtime = graph._runtime
        result['model_source_receipt'] = graph.source_receipt
        layout = tuple(layer.is_attn for layer in model.layers)
        for name,mode,bound in ARMS:
            arrays = mx.load(str(out/'canonical.safetensors')); mx.eval(*arrays.values())
            inputs[name] = dict(metadata=result['canonical']['metadata'],arrays=arrays)
            prior = sum(mx.get_array_buffer_size(x['arrays']) for x in inputs.values())
            union = mx.get_array_buffer_size(*[x['arrays'] for x in inputs.values()])
            need(union == prior, 'arms must own independent evaluated input buffers')
            contexts[name] = context_from_input(arrays,layout,runtime,bound if mode == 'eager' else 257)
            result['arms'][name] = dict(mode=mode,bound=bound,status='PREPARED',
                input_before=dump(mx,out,f'arms/{name}/input-before.safetensors',inputs[name],deadline),
                input_buffer_bytes=mx.get_array_buffer_size(arrays),input_union_bytes=union)
            cmp.compare(name+'/same-canonical',dict(metadata=inputs[name]['metadata'],arrays=mx.load(str(out/'canonical.safetensors'))),inputs[name])
            cmp.rows = []
        for name,mode,bound in ARMS:
            phase('before-'+name); row = result['arms'][name]; ctx = contexts[name]
            ids = inputs[name]['arrays']['root/tokens'][:]
            need(graph._eligible(ids,ctx), 'original fixed model/cache eligibility')
            if name in ('A','B'):
                logits = model(ids,ctx); route = dict(kind='eager',max_len=bound,compiled_invocations=0)
            elif name == 'C':
                compiled = make_bound257(model,runtime)
                logits, states = compiled(ids,ctx.positions,ctx.lens,runtime.states(ctx.caches))
                need(len(states) == 64 and all(len(s) == 2 for s in states), 'compiled257 full state return')
                for cache,state in zip(ctx.caches,states):
                    if type(cache) is runtime.BatchKV: cache.K,cache.V = state
                    else: cache.conv,cache.S = state
                route = dict(kind='explicit-state-mx.compile',max_len=257,compiled_invocations=1,callable_type=type(compiled).__name__)
                del states, state, cache
            else:
                witness = old.GraphWitness(graph,emit); witness.phase = 'D-compiled1536'
                logits = witness(ids,ctx)
                route = dict(kind='original-E024-candidate',max_len=1536,compiled_invocations=witness.functions.invocations,
                    actual_compile_objects=witness.functions.compiled_objects)
            outputs[name] = support.snapshot(dict(logits=logits,state=ctx.caches))
            row['output'] = dump(mx,out,f'arms/{name}/output.safetensors',outputs[name],deadline)
            need(len(outputs[name]['arrays']) == 129, 'output full128 state +logits')
            row['input_after'] = dump(mx,out,f'arms/{name}/input-after.safetensors',inputs[name],deadline)
            cmp.meta(name+'/unchanged-input',row['input_before']['tensors'],row['input_after']['tensors'])
            after = parameter_hashes(mx,model,tree_flatten,deadline)
            write(out/f'parameters-after-{name}.json',after)
            row['parameters_after'] = dict(path=f'parameters-after-{name}.json',sha256=sha(out/f'parameters-after-{name}.json'))
            cmp.meta(name+'/unchanged-model-parameters',parameter_values(before),parameter_values(after))
            row.update(status='COMPLETE',route=route,greedy_ids=mx.argmax(logits,axis=-1).tolist())
            phase('after-'+name); write(out/'progress.json',result); del after,logits,ids,ctx
        for left,right in PAIRS:
            deadline(); label = left+'-'+right
            row = compare_finite_outputs(support,mx,out,label,outputs[left],outputs[right])
            result['comparisons'].append(row); write(out/'comparisons.json',result['comparisons'])
        result['numerical_equal'] = all(r['numerical_equal'] for r in result['comparisons'])
        result['diagnostic_complete'] = True
    finally:
        if graph is not None: graph._functions.clear()
        contexts.clear(); inputs.clear(); outputs.clear(); seeds.clear(); prefills.clear(); engines.clear()
        # The outer cleanup runs after this frame/traceback tensor locals are gone.
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    for field in ('source_manifest','model_manifest','fixtures'): support.pinned_json(inherited[field])
    validate(plan)
    result['source_model_pins_rechecked'] = True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True); ap.add_argument('--plan-sha256',required=True)
    ap.add_argument('--out',type=Path,required=True); args = ap.parse_args(argv)
    need(sha(args.plan) == args.plan_sha256, 'plan SHA')
    plan = json.loads(args.plan.read_text()); old,support,candidate,original,inherited = validate(plan)
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0,0.0), 'existing alarm')
    args.out.mkdir(parents=True,exist_ok=False); started = time.monotonic()
    result = dict(schema='E024-dispatch-isolation-result-v1',status='INCOMPLETE',contract=fixed_plan(),
        device_run=False,pid=os.getpid(),ppid=os.getppid(),arms={},comparisons=[],numerical_equal=None,
        diagnostic_complete=False,source_model_pins_rechecked=False,full_model_qualified=False,performance_qualified=False,
        sources={str(Path(__file__).resolve()):sha(__file__),str(OLD/'qualify.py'):OLD_SHA,
            original['candidate']['path']:old.CANDIDATE_SHA,original['e023_qualifier']['path']:old.SUPPORT_SHA,
            str(candidate.PINS):old.MODEL_SOURCE_SHA},
        plan_sha256=args.plan_sha256,qualifier_sha256=plan['qualifier_sha256'])
    write(args.out/'plan.json',plan); write(args.out/'inherited-e024-plan.json',original); write(args.out/'inherited-e023-plan.json',inherited)
    def deadline(*unused):
        if unused or time.monotonic()-started >= plan['deadline_s']: raise TimeoutError('isolation absolute deadline')
    def stop(signum,frame):
        if signum == signal.SIGALRM: deadline(signum)
        raise InterruptedError('isolation cancellation '+signal.Signals(signum).name)
    previous = {s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
    with (args.out/'events.jsonl').open('x') as stream:
        def emit(row): stream.write(json.dumps(dict(**row,elapsed_wall_s=time.monotonic()-started))+'\n'); stream.flush()
        try:
            for s in previous: signal.signal(s,stop)
            signal.setitimer(signal.ITIMER_REAL,plan['deadline_s'])
            device(plan,old,support,candidate,original,inherited,args.out,deadline,emit,result)
            need(sha(args.plan) == args.plan_sha256, 'plan changed')
            result['status'] = 'PASS'
        except BaseException as error:
            result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        finally:
            try:
                gc.collect(); mx = sys.modules.get('mlx.core')
                if mx is not None:
                    mx.synchronize(); mx.clear_cache()
                    result['final_active_bytes'] = mx.get_active_memory(); result['peak_bytes'] = mx.get_peak_memory()
            except BaseException as error:
                result.update(status='FAIL',cleanup_error=dict(type=type(error).__name__,message=str(error)))
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                for s,h in previous.items(): signal.signal(s,h)
                result['elapsed_wall_s'] = time.monotonic()-started
                write(args.out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],out=str(args.out))),flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__': raise SystemExit(main())
