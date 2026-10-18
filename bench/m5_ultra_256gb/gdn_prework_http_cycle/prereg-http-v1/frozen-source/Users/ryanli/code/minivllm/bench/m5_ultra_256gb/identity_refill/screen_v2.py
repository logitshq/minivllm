"""E028 bounded state-only identity-refill diagnostic; default/help are CPU-safe.

Explicit --run --out NEW: ten cases, 88 checks, 53 observed resource phases.
One real-shape attention layer plus one GDN layer; no weights, forward pass,
64-layer model, HTTP or benchmark timing. OFF/ON sources are independently made
and evaluated; a separate unmodified native oracle covers delayed evaluation.
Full original BF16/F32 sources and outputs are saved as safetensors at every
boundary. Deliberate source/output writes are distinct from read-only checks.
120s child alarm requires an external hard owner; MLX1GiB/physical2GiB are
observations, not OS caps. No device/module import occurs without --run.
"""
import argparse
import gc
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / 'tutorial_code/miniengine/cache.py'
LEDGER = HERE.parent / 'shared_w_m1/screen.py'
PHYSICAL = HERE.parent / 'conv_tail_owned_copy/layer_screen.py'
FIXED_PINS = {
    CACHE: '056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791',
    LEDGER: 'd01f28e36b07176e73830b424f3ab65113f1263cc1f810678f2c88b50c11dacf',
    PHYSICAL: 'a0423d0648b19236f972598ce5acb5e8e9cdd617fadb2646f1c7fef01d9ec302',
}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def cases():
    result = []
    for name, B, kind in [('identity-2-to-4',2,'identity'),('identity-4-to-6',4,'identity'),
            ('identity-6-to-8',6,'identity'),('identity-offset-noncontiguous',4,'views'),
            ('control-reorder',4,'reorder'),('control-removal',4,'removal'),
            ('control-no-new',4,'no-new'),('control-length-growth',4,'growth'),
            ('control-empty-old',0,'empty'),('identity-delayed-eval',4,'delayed')]:
        keep = list(range(B))
        if kind == 'reorder': keep = list(reversed(keep))
        if kind == 'removal': keep = [0,2]
        new = 0 if kind == 'no-new' else 2
        result.append(dict(id=name, old_B=B, keep=keep, new_B=new, old_L=1536,
            new_need=1793 if kind == 'growth' else 1536,
            output_L=2048 if kind == 'growth' else 1536,
            output_B=len(keep)+new, new_offsets=[935,954][:new],
            views=kind == 'views', delayed=kind == 'delayed',
            reverse_output_write=name == 'identity-4-to-6',
            on_path='identity_refill' if kind in ('identity','views','delayed') else 'fallback',
            checks=8+(4 if kind == 'delayed' else 0)+(4 if name == 'identity-4-to-6' else 0)))
    return result


def fixed_plan():
    return dict(schema='E028-identity-refill-screen-v2', cases=cases(), expected_cases=10,
        expected_checks=88, expected_phases=53, cache_limit_bytes=0,
        mlx_observation_budget_bytes=1 << 30, physical_observation_budget_bytes=2 << 30,
        child_alarm_seconds=120, physical_reading_semantics='separate-int-bounds;current=max(top,aux-current);peak=max(top,aux-peak);aux-peak>=aux-current',
        layers=[dict(kind='attention',Hkv=4,D=256,KV_dtype='bfloat16'),
                dict(kind='GDN',conv_shape=[3,10240],conv_dtype='bfloat16',S_shape=[48,128,128],S_dtype='float32')],
        phases=['before_case','after_rebuild_graph','after_outputs_eval','after_source_writes','after_case_release'],
        extra_phase='after_output_write only identity-4-to-6',
        raw_storage='complete safetensors at each path; byte-identical snapshots may be hardlinked after SHA verification',
        global_phases=['initial','final_release only on PASS'],
        source_writes=dict(delayed_before_output_eval=-8.0,after_output_eval=-16.0), output_write=-32.0,
        mutation='Exactly the first scalar of every present old/staging/output array; all other bytes unchanged',
        source_pattern='BF16/F32 finite uint words: low=(i+i//127+seed)%128, sign=(i+i//127+seed)&1; no RNG',
        view_layout='old KV: backing[B+1,4,L+2,512][1:,:,1:L+1,::2]; analogous staging, conv and S offset/strided views',
        gate='finite + complete original dtype/shape/bytes + IDs/lens + branch witness + independent buffer ownership',
        true_empty_keep_and_new='not supported by native rebuild; not invented as a successful case',
        performance_qualified=False, full_model_qualified=False, observed_limits_not_OS_hard_caps=True)


def source_pins():
    for path, expected in FIXED_PINS.items(): need(sha(path) == expected, 'fixed source SHA: '+str(path))
    return {**{str(p):h for p,h in FIXED_PINS.items()},
            **{str(p):sha(p) for p in (Path(__file__).resolve(),HERE/'candidate.py')}}


def stop_signal(signum, frame=None):
    if signum == signal.SIGALRM: raise TimeoutError('120s E028 screen SIGALRM deadline')
    raise InterruptedError('E028 cancellation '+signal.Signals(signum).name)


def read_tensors(path):
    """Read saved payloads directly; no dtype conversion or MLX load."""
    raw = Path(path).read_bytes(); need(len(raw) >= 8, 'missing safetensors prefix')
    n = struct.unpack('<Q',raw[:8])[0]
    need(0 < n <= 1 << 20 and 8+n <= len(raw), 'safetensors header bounds')
    header = json.loads(raw[8:8+n]); payload = raw[8+n:]; result = {}; spans = []
    for name, row in header.items():
        if name == '__metadata__': continue
        dtype, shape, (lo,hi) = row['dtype'],row['shape'],row['data_offsets']
        need(dtype in ('BF16','F32') and all(type(x) is int and x >= 0 for x in shape), 'raw dtype/shape')
        need(0 <= lo <= hi <= len(payload) and hi-lo == math.prod(shape)*(2 if dtype == 'BF16' else 4), 'raw byte bounds')
        result[name] = dict(dtype=dtype,shape=shape,raw=payload[lo:hi]); spans.append((lo,hi))
    cursor = 0
    for lo,hi in sorted(spans): need(lo == cursor, 'raw overlap/gap'); cursor = hi
    need(cursor == len(payload), 'raw trailing payload')
    return result


def dump(mx, out, path, arrays):
    need(bool(arrays), 'empty tensor snapshot')
    for a in arrays.values(): need(a.dtype in (mx.bfloat16,mx.float32), 'unexpected state dtype')
    mx.eval(*arrays.values())
    target = out/path; target.parent.mkdir(parents=True,exist_ok=True)
    mx.save_safetensors(str(target), arrays)
    raw = read_tensors(target); need(set(raw) == set(arrays), 'saved tensor keys')
    metadata = {}
    for key, value in arrays.items():
        d = raw[key]; finite = bool(mx.all(mx.isfinite(value)).item())
        metadata[key] = dict(shape=list(value.shape), dtype=str(value.dtype), encoding=d['dtype'],
            bytes=len(d['raw']), sha256=hashlib.sha256(d['raw']).hexdigest(), finite=finite)
        need(d['shape'] == list(value.shape), 'saved tensor shape')
    return dict(path=path,sha256=sha(target),tensors=metadata,finite=all(d['finite'] for d in metadata.values()))


def scalar_bytes(value, dtype):
    f32 = struct.pack('<f',value)
    return f32[2:] if dtype == 'BF16' else f32


def compare(out, result, case, label, left, right, emit, mutation=None):
    a,b = read_tensors(out/left['path']), read_tensors(out/right['path'])
    need(sha(out/left['path']) == left['sha256'] and sha(out/right['path']) == right['sha256'], 'snapshot changed')
    exact = set(a) == set(b) and left['finite'] and right['finite']
    differences = []
    for key in sorted(set(a)|set(b)):
        if key not in a or key not in b:
            differences.append(dict(key=key,reason='missing key')); exact=False; continue
        x,y = a[key],b[key]; expected = x['raw']
        if mutation is not None:
            word = scalar_bytes(mutation,x['dtype']); expected = word+expected[len(word):]
        same = x['shape'] == y['shape'] and x['dtype'] == y['dtype'] and expected == y['raw']
        if not same:
            first = next((i for i,(p,q) in enumerate(zip(expected,y['raw'])) if p != q),None)
            differences.append(dict(key=key,first_differing_byte=first))
        exact &= same
    row = dict(index=len(result['checks']),case=case,label=label,status='PASS' if exact else 'FAIL',
        finite=left['finite'] and right['finite'],bytes_equal=bool(exact),
        expected_first_scalar_write=mutation,rawpair=dict(reference=left,candidate=right),differences=differences)
    result['checks'].append(row); write(out/'checks.json',result['checks']); emit(dict(event='comparison',**row))
    if not exact:
        if not (out/'first-difference.json').exists(): write(out/'first-difference.json',row)
        raise ValueError('strict state gate rejected: '+case+'/'+label)


def pattern(mx, shape, dtype, seed):
    index = mx.arange(math.prod(shape),dtype=mx.uint32)
    mixed = index+index//127+seed
    low = mixed%128; sign = mixed&1
    if dtype == mx.bfloat16:
        words = (low | 0x3e00 | (sign << 15)).astype(mx.uint16)
    else:
        words = (low << 16) | 0x3e800000 | (sign << 31)
    return words.view(dtype).reshape(shape)


def value(mx, shape, dtype, seed, views):
    if not views: return pattern(mx,shape,dtype,seed)
    expanded = list(shape); expanded[0] += 1; expanded[-1] *= 2
    if len(shape) == 4: expanded[2] += 2
    parent = pattern(mx,tuple(expanded),dtype,seed)
    slices = [slice(None)]*len(shape); slices[0] = slice(1,shape[0]+1); slices[-1] = slice(None,None,2)
    if len(shape) == 4: slices[2] = slice(1,shape[2]+1)
    return parent[tuple(slices)]


def make_arm(mx, cache, cls, case, enabled):
    model = SimpleNamespace(layers=[SimpleNamespace(is_attn=True),SimpleNamespace(is_attn=False)])
    batch = cls(model) if enabled is None else cls(model,enabled=enabled)
    B,L = case['old_B'],case['old_L']; views = case['views']
    batch.req_ids = [f'old-{i}' for i in range(B)]; batch.lens = [920+7*i for i in range(B)]
    old = []
    if B:
        old = [cache.BatchKV(value(mx,(B,4,L,256),mx.bfloat16,11,views),value(mx,(B,4,L,256),mx.bfloat16,23,views)),
               cache.BatchState(value(mx,(B,3,10240),mx.bfloat16,37,views),value(mx,(B,48,128,128),mx.float32,41,views))]
        batch.layers = list(old)
    staging = []
    for i,offset in enumerate(case['new_offsets']):
        attn,gdn = cache.AttnCache(),cache.GDNCache()
        attn.keys = value(mx,(1,4,1024,256),mx.bfloat16,53+i*17,views)
        attn.values = value(mx,(1,4,1024,256),mx.bfloat16,67+i*17,views)
        gdn.conv_state = value(mx,(3,10240),mx.bfloat16,79+i*17,views)
        gdn.S = value(mx,(48,128,128),mx.float32,97+i*17,views)
        attn.offset = gdn.offset = offset
        staging.append((f'new-{i}',[attn,gdn],case['new_need']))
    return dict(batch=batch,old=old,staging=staging)


def sources(arm):
    result = {}
    if arm['old']:
        attn,gdn = arm['old']; result.update(old_K=attn.K,old_V=attn.V,old_conv=gdn.conv,old_S=gdn.S)
    for i,(_,caches,_) in enumerate(arm['staging']):
        attn,gdn = caches
        result.update({f'new{i}_K':attn.keys,f'new{i}_V':attn.values,f'new{i}_conv':gdn.conv_state,f'new{i}_S':gdn.S})
    return result


def outputs(arm):
    attn,gdn = arm['batch'].layers
    return dict(K=attn.K,V=attn.V,conv=gdn.conv,S=gdn.S)


def mutate(arrays, scalar):
    for value in arrays.values(): value[(0,)*value.ndim] = scalar


def expected_metadata(case):
    return dict(req_ids=[f'old-{i}' for i in case['keep']]+[f'new-{i}' for i in range(case['new_B'])],
                lens=[920+7*i for i in case['keep']]+case['new_offsets'],
                shapes=dict(K=[case['output_B'],4,case['output_L'],256],V=[case['output_B'],4,case['output_L'],256],
                    conv=[case['output_B'],3,10240],S=[case['output_B'],48,128,128]))


def run_case(mx, cache, candidate, out, result, case, ledger, emit, raw_catalog):
    name = case['id']; row = dict(id=name,status='INCOMPLETE',snapshots={},witness={})
    result['cases'].append(row); case_dir = out/'cases'/name; case_dir.mkdir(parents=True)
    def save(): write(case_dir/'case.json',row)
    def snap(label, arrays):
        record = dump(mx,out,f'cases/{name}/{label}.safetensors',arrays)
        key = record['sha256']; target = out/record['path']
        if key in raw_catalog:
            prior = out/raw_catalog[key]
            need(sha(prior) == key, 'deduplicated snapshot changed')
            target.unlink(); os.link(prior,target)
            record['identical_payload_of'] = raw_catalog[key]
        else:
            raw_catalog[key] = record['path']
        row['snapshots'][label] = record; save(); return record
    def check(label,left,right,mutation=None): compare(out,result,name,label,left,right,emit,mutation)
    first_check = len(result['checks']); arms = {}; oracle = None
    try:
        ledger.phase('before_case',name)
        if case['delayed']:
            oracle = make_arm(mx,cache,cache.DecodeBatch,case,None)
            mx.eval(*sources(oracle).values())
            oracle_inputs = snap('unwritten-native-oracle-inputs',sources(oracle))
            oracle['batch'].rebuild(case['keep'],oracle['staging'])
            oracle_output = snap('unwritten-native-oracle',outputs(oracle))
            oracle = None; gc.collect(); mx.synchronize(); mx.clear_cache()
        cls = candidate.make_batch_class(cache,mx)
        arms = {label:make_arm(mx,cache,cls,case,enabled) for label,enabled in [('OFF',False),('ON',True)]}
        source_arrays = {label:sources(arm) for label,arm in arms.items()}
        mx.eval(*source_arrays['OFF'].values(),*source_arrays['ON'].values())
        need(not ({id(a) for a in source_arrays['OFF'].values()} & {id(a) for a in source_arrays['ON'].values()}), 'arms share array objects')
        left_bytes = int(mx.get_array_buffer_size(list(source_arrays['OFF'].values())))
        right_bytes = int(mx.get_array_buffer_size(list(source_arrays['ON'].values())))
        union = int(mx.get_array_buffer_size(list(source_arrays['OFF'].values())+list(source_arrays['ON'].values())))
        need(left_bytes > 0 and right_bytes > 0 and union == left_bytes+right_bytes, 'arms share backing buffers')
        row['independent_sources'] = dict(OFF_backing_bytes=left_bytes,ON_backing_bytes=right_bytes,union_backing_bytes=union,disjoint=True)
        initial = {label:snap(label+'-inputs-before',source_arrays[label]) for label in arms}
        check('initial-inputs-OFF-vs-ON',initial['OFF'],initial['ON'])
        if case['delayed']:
            need(oracle_inputs['sha256'] == initial['OFF']['sha256'], 'unwritten oracle input bytes differ')
            row['oracle_inputs_identical_to_initial_sources'] = True
        for label,arm in arms.items():
            arm['batch'].rebuild(case['keep'],arm['staging'])
            batch = arm['batch']; expected_path = case['on_path'] if label == 'ON' else 'fallback'
            witness = dict(path=batch.last_path,reason=batch.last_reason,counters=dict(batch.counters))
            need(witness['path'] == expected_path and witness['counters'] == dict(identity_refill=int(expected_path=='identity_refill'),fallback=int(expected_path=='fallback')), 'branch witness mismatch')
            wanted = expected_metadata(case)
            actual = dict(req_ids=list(batch.req_ids),lens=list(batch.lens),shapes={k:list(v.shape) for k,v in outputs(arm).items()})
            need(actual == wanted, 'IDs/lens/output layout mismatch')
            row['witness'][label] = witness; row.setdefault('metadata',{})[label] = actual
            check(label+'-rebuild-inputs-readonly',initial[label],snap(label+'-inputs-after-rebuild',source_arrays[label]))
        save(); ledger.phase('after_rebuild_graph',name)
        source_baseline = dict(initial)
        if case['delayed']:
            for label in arms:
                mutate(source_arrays[label],-8.0)
                changed = snap(label+'-inputs-before-delayed-eval',source_arrays[label])
                check(label+'-pre-eval-source-write',initial[label],changed,-8.0)
                source_baseline[label] = changed
        first_outputs = {label:snap(label+'-outputs-first-eval',outputs(arm)) for label,arm in arms.items()}
        check('rebuild-outputs-OFF-vs-ON',first_outputs['OFF'],first_outputs['ON'])
        if case['delayed']:
            for label in arms: check(label+'-delayed-output-vs-unwritten-oracle',oracle_output,first_outputs[label])
        ledger.phase('after_outputs_eval',name)
        post_sources = {}
        for label,arm in arms.items():
            mutate(source_arrays[label],-16.0)
            post_sources[label] = snap(label+'-inputs-after-source-write',source_arrays[label])
            check(label+'-source-write-exact',source_baseline[label],post_sources[label],-16.0)
            after = snap(label+'-outputs-after-source-write',outputs(arm))
            check(label+'-output-stable-after-source-write',first_outputs[label],after)
        ledger.phase('after_source_writes',name)
        if case['reverse_output_write']:
            for label,arm in arms.items():
                mutate(outputs(arm),-32.0)
                changed = snap(label+'-outputs-after-output-write',outputs(arm))
                check(label+'-output-write-exact',first_outputs[label],changed,-32.0)
                after = snap(label+'-inputs-after-output-write',source_arrays[label])
                check(label+'-source-stable-after-output-write',post_sources[label],after)
            ledger.phase('after_output_write',name)
        need(len(result['checks'])-first_check == case['checks'], 'per-case check coverage')
        row['status'] = 'PASS'; save()
    except BaseException as error:
        row.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error))); save()
        # Preserve any remaining live original sources without masking the first error.
        for label,arm in arms.items():
            try: snap(label+'-inputs-on-error',sources(arm))
            except BaseException as secondary:
                row.setdefault('evidence_errors',[]).append(dict(arm=label,type=type(secondary).__name__,message=str(secondary)))
        save(); raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(); mode.add_argument('--describe',action='store_true'); mode.add_argument('--run',action='store_true')
    ap.add_argument('--out',type=Path); args = ap.parse_args(argv)
    if not args.run: print(json.dumps(fixed_plan(),indent=2)); return 0
    if args.out is None: ap.error('--run requires --out NEW')
    args.out.mkdir(parents=True,exist_ok=False); started = time.monotonic()
    result = dict(schema='E028-identity-refill-result-v2',status='INCOMPLETE',contract=fixed_plan(),
        cases=[],checks=[],physical_receipts=[],source_pins_rechecked=False,device_run=False,
        pid=os.getpid(),ppid=os.getppid(),elapsed_wall_s=0,performance_qualified=False,full_model_qualified=False)
    write(args.out/'complete.json',result); write(args.out/'plan.json',result['contract'])
    previous = {s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
    mx = None
    with (args.out/'events.jsonl').open('x') as events:
        def emit(row): events.write(json.dumps(dict(**row,elapsed_wall_s=time.monotonic()-started))+'\n'); events.flush()
        try:
            for sig in previous: signal.signal(sig,stop_signal)
            signal.setitimer(signal.ITIMER_REAL,max(.001,120-(time.monotonic()-started)))
            result['sources'] = source_pins()
            shared,physical = load('e028_shared_ledger',LEDGER),load('e028_physical',PHYSICAL)
            import mlx.core as mx
            result['device_run'] = True
            result['device'] = dict(name=mx.device_info().get('device_name'),architecture=mx.metal.device_info().get('architecture'),mlx_version=importlib.metadata.version('mlx'))
            need(result['device'] == dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3') and mx.default_device() == mx.gpu, 'fixed M5 Ultra GPU/runtime required')
            mx.set_cache_limit(0); mx.synchronize(); mx.clear_cache(); mx.reset_peak_memory()
            cache,candidate = load('e028_native_cache',CACHE),load('e028_identity_candidate',HERE/'candidate.py')
            ledger = shared.Ledger(args.out,result,mx,physical,started); ledger.phase('initial')
            raw_catalog = {}
            for case in cases():
                run_case(mx,cache,candidate,args.out,result,case,ledger,emit,raw_catalog)
                gc.collect(); mx.synchronize(); mx.clear_cache(); ledger.phase('after_case_release',case['id'])
            need(len(result['cases']) == 10 and len(result['checks']) == 88, 'coverage mismatch')
            need(source_pins() == result['sources'], 'source changed')
            result['source_pins_rechecked'] = True
            gc.collect(); mx.synchronize(); mx.clear_cache(); ledger.phase('final_release')
            need(ledger.count == 53, 'phase coverage mismatch')
            result['status'] = 'PASS'
        except BaseException as error:
            result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
            result['elapsed_wall_s'] = time.monotonic()-started; write(args.out/'complete.json',result)
        finally:
            try:
                if mx is not None and result['status'] != 'PASS': gc.collect(); mx.synchronize(); mx.clear_cache()
            except BaseException as error: result['cleanup_error'] = dict(type=type(error).__name__,message=str(error))
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                for sig,handler in previous.items(): signal.signal(sig,handler)
            result['elapsed_wall_s'] = time.monotonic()-started; write(args.out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],out=str(args.out))))
    return int(result['status'] != 'PASS')


if __name__ == '__main__': raise SystemExit(main())
