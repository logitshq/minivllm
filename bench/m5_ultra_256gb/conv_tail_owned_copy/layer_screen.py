"""E026 real-weight/synthetic-input single GDN layer diagnostic, no performance.

Default/help are CPU-safe. Explicit --run uses one fresh process/arm and the
already prepared 14-tensor bundle. The owner must enforce a 120s hard deadline,
physical/resource contract and direct reap. No server, full model or mx.load of
the original shard. PASS is local completeness/finite/route/unchanged-parameter
qualification ONLY; an independent raw-byte audit must compare the three arms.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
ARMS = ('off', 'on-lazy', 'on-materialize')
LENGTHS = (1, 19, 935, 978)
KINDS = ('fresh', 'continuing')
CASE_PHASES = ('before_prefill', 'after_hook', 'after_prefill_build', 'after_normal_product_eval',
               'after_common_validation_eval', 'after_raw_artifacts', 'after_drop_observer_refs', 'after_case_release')
MEMORY_BYTES, FOOTPRINT_BYTES = 1 << 30, 2 << 30
WEIGHTS_SHA = '8ee0ebd9df9385ec35baf6d0a0ae80bf780cf80a5e255b34d6fe2fed560df77a'
INTEGRATION_SHA = '906dcfece0ce249d3fc06e968ec552d0ae20b665086e5812c54453fa47780ad1'
CANDIDATE_SHA = '4d9658d34734e3c252375c4b76fcb2eccf83555121d7eb51d7f3e81eb2c48ccc'
CORE_MANIFEST_SHA = '93bf4b26318a1a7820c00eecec5b134f93f1f516435e4b9709e91e8b186a4050'


def need(ok, message):
    if not ok: raise ValueError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 << 20): h.update(chunk)
    return h.hexdigest()


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract(arm=None):
    return dict(schema='E026-real-weight-layer-v1', arm=arm, layer_index=0,
        lengths=list(LENGTHS), cache_kinds=list(KINDS), cases=8,
        case_ids=[f'T{t}-{kind}' for t in LENGTHS for kind in KINDS],
        cache_limit_bytes=0, mlx_observation_budget_bytes=MEMORY_BYTES,
        physical_observation_budget_bytes=FOOTPRINT_BYTES, child_alarm_seconds=120,
        phase_count=68, case_phases=list(CASE_PHASES),
        x_words='uint16: 0x3c00 | ((17*i+11)&127) | (((i+11)&1)<<15)',
        conv_seed_words='uint16: 0x3b00 | ((17*i+23)&127) | (((i+23)&1)<<15)',
        S_seed_words='uint32: 0x37800000 | ((13*i+37)&0x7fffff) | (((i+37)&1)<<31)',
        continuing_offset=7, continuing_prefix_forward=False,
        normal_eval='only actual layer output; no additional state/observer eval',
        common_eval='output, cache.S, cache.conv_state, actual conv_y, original conv tail',
        observer='sys.setprofile exact original causal_conv1d code return, count=1; previous profiler restored',
        parameter_evidence='all 14 full raw before/after per arm; every case full-byte hash/finite before/after; changed parameters dumped',
        scope='Real fixed layer-0 weights plus deterministic synthetic inputs; no actual D/full-model/HTTP/performance claim',
        memory_limit_scope='MLX and OS historical peak observations; not an allocator or OS hard cap; no E023 pressure causation claim')


def verify_sources(package_root):
    root = Path(package_root).resolve()
    manifest_path = root.parent / 'native-source.json'
    need(file_sha(manifest_path) == CORE_MANIFEST_SHA, 'fixed frozen core manifest SHA')
    manifest = json.loads(manifest_path.read_text())
    core = root / 'tutorial_code/miniengine'
    pins = {str(manifest_path): CORE_MANIFEST_SHA}
    for relative, expected in manifest['files'].items():
        path = (core / relative).resolve()
        need(path.is_relative_to(core), 'source path escapes core')
        need(file_sha(path) == expected, 'core source SHA: ' + relative)
        pins[str(path)] = expected
    for name, expected in (('layer_weights.py', WEIGHTS_SHA), ('integration.py', INTEGRATION_SHA),
                           ('candidate.py', CANDIDATE_SHA)):
        path = HERE / name
        need(file_sha(path) == expected, 'screen support SHA: ' + name)
        pins[str(path)] = expected
    pins[str(Path(__file__).resolve())] = file_sha(__file__)
    return pins


def parse_footprint(raw, pid):
    need(raw.get('unit') == 'byte' and type(raw.get('bytes per unit')) is int
         and raw.get('bytes per unit') == 1, 'footprint units')
    need(raw.get('errors') == [] and raw.get('warnings') == [], 'footprint errors/warnings')
    rows = raw.get('processes')
    need(type(rows) is list and len(rows) == 1 and type(rows[0].get('pid')) is int
         and type(pid) is int and rows[0].get('pid') == pid, 'footprint exact PID')
    row = rows[0]; aux = row.get('auxiliary', {})
    values = (row.get('footprint'), aux.get('phys_footprint'), aux.get('phys_footprint_peak'))
    need(all(type(v) is int and 0 <= v <= FOOTPRINT_BYTES for v in values), 'physical current/peak >2GiB or invalid')
    need(values[0] == values[1] and values[2] >= values[1], 'physical current/peak inconsistent')
    return values[1], values[2]


def memory_guard(row):
    for key in ('active_bytes', 'peak_bytes', 'cache_bytes', 'unique_buffer_bytes'):
        need(type(row[key]) is int and row[key] >= 0, 'invalid memory sample')
    need(row['peak_bytes'] <= MEMORY_BYTES and row['active_bytes'] + row['cache_bytes'] <= MEMORY_BYTES,
         'MLX observed budget exceeded 1GiB')
    need(row['unique_buffer_bytes'] <= row['active_bytes'], 'unique buffers exceed active')


class Ledger:
    def __init__(self, out, result, mx, started):
        self.out, self.result, self.mx, self.started = out, result, mx, started
        self.count = 0

    def phase(self, phase, groups, case='global'):
        mx = self.mx
        row = dict(index=self.count, case=case, phase=phase, pid=os.getpid(), ppid=os.getppid(),
                   monotonic_s=time.monotonic(), elapsed_s=time.monotonic()-self.started,
                   active_bytes=int(mx.get_active_memory()), peak_bytes=int(mx.get_peak_memory()),
                   cache_bytes=int(mx.get_cache_memory()), groups_are_evaluated=True,
                   unique_buffer_bytes=int(mx.get_array_buffer_size(*groups.values())),
                   group_buffer_bytes={k:int(mx.get_array_buffer_size(v)) for k,v in groups.items()})
        raw_path = self.out / f'physical-{self.count:03d}.raw.json'
        cmd = ['/usr/bin/footprint','-p',str(os.getpid()),'--noCategories','-f','bytes','-j',str(raw_path)]
        receipt = dict(path=raw_path.name, pid=os.getpid(), phase=phase, case=case)
        try:
            probe = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            write(self.out / f'physical-{self.count:03d}.command.json',
                  dict(command=cmd,returncode=probe.returncode,stdout=probe.stdout,stderr=probe.stderr))
            need(probe.returncode == 0 and raw_path.is_file(), 'self footprint command failed')
            receipt['sha256'] = file_sha(raw_path)
            self.result['physical_receipts'].append(receipt)
            current, peak = parse_footprint(json.loads(raw_path.read_text()), os.getpid())
            row.update(physical_receipt=receipt, physical_current_bytes=current, physical_peak_bytes=peak)
            memory_guard(row)
        except BaseException as error:
            row.update(status='FAIL',error_type=type(error).__name__,error=str(error))
            raise
        finally:
            with (self.out / 'phases.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n'); stream.flush()
            self.count += 1
        return row


@contextmanager
def observe_convolution(code, observed):
    """Observe actual original return; never replace model function or tuple."""
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if previous is not None: previous(frame, event, arg)
        if frame.f_code is code and event == 'return':
            observed['count'] += 1
            # A Python exception can produce a return event with arg=None.
            # Record it without replacing the original failure from the model.
            if type(arg) is tuple and len(arg) == 2 and observed['count'] == 1:
                observed['tuple'] = arg
            else:
                observed['invalid_return'] = True
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)
        observed['restored'] = sys.getprofile() is previous


def seed_words(np, size, kind):
    index = np.arange(size, dtype=np.uint32)
    if kind == 'S':
        return (np.uint32(0x37800000) | ((13*index+37) & 0x7fffff)
                | (((index+37)&1) << 31)).astype('<u4')
    base, salt = (0x3c00, 11) if kind == 'x' else (0x3b00, 23)
    return (base | ((17*index+salt)&127) | (((index+salt)&1) << 15)).astype('<u2')


def seed(mx, np, shape, kind):
    words = seed_words(np, __import__('math').prod(shape), kind).reshape(shape)
    return mx.array(words).view(mx.float32 if kind == 'S' else mx.bfloat16)


def array_record(mx, np, value, out, relative=None):
    types = ((mx.bfloat16,'bfloat16',mx.uint16,'<u2','uint16-le',0x7f80),
             (mx.float32,'float32',mx.uint32,'<u4','uint32-le',0x7f800000),
             (mx.uint32,'uint32',mx.uint32,'<u4','uint32-le',None),
             (mx.uint8,'uint8',mx.uint8,'u1','uint8',None))
    selected = next((r for r in types if value.dtype == r[0]), None)
    need(selected is not None, 'unexpected original dtype')
    _, dtype, unsigned, host_dtype, encoding, exponent = selected
    host = np.asarray(value.view(unsigned)).astype(host_dtype, copy=False)
    raw = host.tobytes()
    finite = None if exponent is None else bool(np.all((host & exponent) != exponent))
    row = dict(sha256=digest(raw), bytes=len(raw), dtype=dtype, shape=list(value.shape), encoding=encoding, finite=finite)
    if relative is not None:
        path = out / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw); row['file'] = relative
    return row


def flat_parameters(layer):
    result = {}
    def visit(value, prefix):
        if isinstance(value, dict):
            for key, item in value.items(): visit(item, prefix + '.' + key if prefix else key)
        else:
            need(hasattr(value,'dtype') and hasattr(value,'shape'), 'non-array parameter')
            result[prefix] = value
    visit(layer.parameters(), '')
    return result


def parameter_records(mx, np, parameters, out, directory=None):
    return {name:array_record(mx,np,value,out,None if directory is None else f'{directory}/{name}.bin')
            for name,value in sorted(parameters.items())}


def require_finite(rows):
    need(all(row is None or row['finite'] is not False for row in rows.values()), 'nonfinite original-dtype tensor')


def without_files(rows):
    return {name:{k:v for k,v in row.items() if k != 'file'} for name,row in rows.items()}


def same_parameters(mx, np, layer, baseline, out, failure_dir):
    params = flat_parameters(layer)
    rows = parameter_records(mx,np,params,out)
    if rows != without_files(baseline):
        for name,value in params.items():
            if rows.get(name) != without_files(baseline).get(name):
                array_record(mx,np,value,out,f'{failure_dir}/{name}.bin')
        raise ValueError('parameter bits/shape/dtype/finite changed')
    require_finite(rows)
    return rows


def one_case(mx, np, gdn, cache_cls, integration, layer, parameters, baseline, arm, length, kind, out, ledger):
    case_id = f'T{length}-{kind}'
    directory = out / 'cases' / case_id; directory.mkdir(parents=True,exist_ok=False)
    row = dict(case=case_id,status='INCOMPLETE',length=length,kind=kind,layer_index=0)
    observed, witness = {'count':0}, []
    def save(): write(directory/'case.json',row)
    save()
    try:
        cache = cache_cls()
        x = seed(mx,np,(length,5120),'x')
        if kind == 'continuing':
            cache.conv_state = seed(mx,np,(3,10240),'conv')
            cache.S = seed(mx,np,(48,128,128),'S'); cache.offset = 7
        seed_sources = dict(x=x,conv_seed=cache.conv_state,S_seed=cache.S)
        seeds = [v for v in (x,cache.conv_state,cache.S) if v is not None]
        mx.eval(*seeds)
        row['offset_before'] = cache.offset
        row['input'] = dict(x=array_record(mx,np,x,out,f'cases/{case_id}/input-x.bin'),
                            conv_seed=None if cache.conv_state is None else array_record(mx,np,cache.conv_state,out,f'cases/{case_id}/seed-conv.bin'),
                            S_seed=None if cache.S is None else array_record(mx,np,cache.S,out,f'cases/{case_id}/seed-S.bin'))
        require_finite(row['input'])
        row['parameters_before'] = same_parameters(mx,np,layer,baseline,out,f'cases/{case_id}/changed-parameters-before')
        ledger.phase('before_prefill',{'weights':parameters,'inputs':seeds},case_id)
        def on_call(receipt):
            witness.append(receipt)
            with (out/'witness.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(case=case_id,**receipt))+'\n'); stream.flush()
            need(receipt['eligible'] and receipt['layer']==0 and receipt['route']=='single-layer-eager'
                 and receipt['enabled']==(arm!='off') and receipt['materialize']==(arm=='on-materialize')
                 and receipt['replaced']==(arm!='off'), 'required hook route not exercised')
            # Pending output, tail and observed convolution are intentionally excluded.
            ledger.phase('after_hook',{'weights':parameters,'inputs':seeds},case_id)
        with integration.PrefillTailScope(gdn,{0:layer},enabled=arm!='off',materialize=arm=='on-materialize',on_call=on_call) as scope:
            with observe_convolution(gdn.causal_conv1d.__code__,observed):
                output = scope.single_layer(0,x,cache)
        need(observed['count']==1 and observed['restored'] and len(witness)==1
             and 'tuple' in observed and not observed.get('invalid_return'), 'actual convolution/profile coverage')
        row['hook_witness'], row['profiler'] = witness[0], dict(count=observed['count'],restored=True,
            original_source_sha256=integration.CORE_PINS['model/gdn.py'],observer='exact-original-code-return')
        ledger.phase('after_prefill_build',{'weights':parameters,'inputs':seeds},case_id)
        mx.eval(output)  # The only normal-product evaluation in this diagnostic.
        ledger.phase('after_normal_product_eval',{'weights':parameters,'inputs':seeds,'output':output},case_id)
        conv_y, original_tail = observed['tuple']
        mx.eval(cache.S,cache.conv_state,conv_y,original_tail)
        arrays = dict(output=output,conv_y=conv_y,conv_state=cache.conv_state,S=cache.S,original_conv_tail=original_tail)
        ledger.phase('after_common_validation_eval',{'weights':parameters,'inputs':seeds,'products':arrays},case_id)
        row['arrays'] = {name:array_record(mx,np,value,out,f'cases/{case_id}/{name}.bin') for name,value in arrays.items()}
        require_finite(row['arrays'])
        expected = dict(output=([length,5120],'bfloat16'),conv_y=([length,10240],'bfloat16'),
                        conv_state=([3,10240],'bfloat16'),S=([48,128,128],'float32'),
                        original_conv_tail=([3,10240],'bfloat16'))
        need(all((row['arrays'][name]['shape'],row['arrays'][name]['dtype'])==value
                 for name,value in expected.items()), 'product shape/dtype changed')
        need(without_files({'tail':row['arrays']['conv_state']})==without_files({'tail':row['arrays']['original_conv_tail']}),
             'compact tail bits differ from actual original convolution tail')
        row['offset_after'] = cache.offset
        need(cache.offset==row['offset_before']+length,'cache logical offset')
        row['input_after'] = {name:None if value is None else array_record(mx,np,value,out)
                              for name,value in seed_sources.items()}
        need(row['input_after']=={name:None if value is None else {k:v for k,v in value.items() if k!='file'}
                                 for name,value in row['input'].items()},'input/initial cache seed mutated')
        row['parameters_after'] = same_parameters(mx,np,layer,baseline,out,f'cases/{case_id}/changed-parameters-after')
        ledger.phase('after_raw_artifacts',{'weights':parameters,'inputs':seeds,'products':arrays},case_id)
        arrays.clear(); observed.clear(); del conv_y, original_tail
        del x, seeds, seed_sources
        gc.collect(); mx.synchronize(); mx.clear_cache()
        ledger.phase('after_drop_observer_refs',{'weights':parameters,'output':output,'state':[cache.conv_state,cache.S]},case_id)
        del output, cache
        gc.collect(); mx.synchronize(); mx.clear_cache()
        ledger.phase('after_case_release',{'weights':parameters},case_id)
        row['status']='PASS'; save()
        return row
    except BaseException as error:
        row.update(status='FAIL',error_type=type(error).__name__,error=str(error),
                   partial_profiler={k:v for k,v in observed.items() if k!='tuple'},hook_witnesses=witness)
        save()
        raise
    finally:
        observed.clear()  # Never retain raw exception objects or model frames in receipts.


def execute(args):
    args.out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    result = dict(status='INCOMPLETE',arm=args.arm,pid=os.getpid(),ppid=os.getppid(),device_run=True,
                  contract=contract(args.arm),cases=[],physical_receipts=[])
    def save(): write(args.out/'complete.json',result)
    def stop(*unused): raise TimeoutError('120s child deadline; owner enforces hard wall bound')
    need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm unsupported')
    previous = {sig:signal.signal(sig,stop) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL,120); save()
    try:
        result['sources']=verify_sources(args.package_root)
        weights=load_file('e026_layer_weights',HERE/'layer_weights.py')
        manifest=weights.verify_bundle(args.bundle_manifest,args.bundle_sha256)
        result.update(bundle_manifest_sha256=args.bundle_sha256,bundle_sha256=manifest['bundle']['sha256'])
        import mlx.core as mx
        import numpy as np
        candidate=load_file('e026_layer_candidate',HERE/'candidate.py')
        need(candidate.supported_backend(mx) and sys.byteorder=='little','pinned backend/little-endian required')
        need(callable(getattr(mx,'get_array_buffer_size',None)),'buffer API required')
        sys.path.insert(0,str(args.package_root.resolve()))
        gdn=importlib.import_module('tutorial_code.miniengine.model.gdn')
        config=importlib.import_module('tutorial_code.miniengine.config')
        caches=importlib.import_module('tutorial_code.miniengine.cache')
        for name,module in list(sys.modules.items()):
            if name.startswith('tutorial_code.miniengine') and getattr(module,'__file__',None):
                path=str(Path(module.__file__).resolve())
                need(path in result['sources'] and file_sha(path)==result['sources'][path],'loaded core outside pins: '+name)
        integration=load_file('e026_layer_integration',HERE/'integration.py')
        result['runtime']=dict(mlx_file=mx.__file__,numpy_file=np.__file__,device=mx.device_info(),metal=mx.metal.device_info())
        mx.set_cache_limit(0); gc.collect(); mx.synchronize(); mx.clear_cache(); mx.reset_peak_memory()
        initial_active=int(mx.get_active_memory())
        result['initial_active_bytes']=initial_active
        layer=gdn.GatedDeltaNet(config.ModelConfig.from_dir(args.bundle_manifest.parent))
        layer_ref=weakref.ref(layer)
        loaded=weights.load_arrays(mx,np,args.bundle_manifest.parent,manifest)
        layer.load_weights(loaded,strict=True); del loaded
        parameters=flat_parameters(layer)
        need(set(parameters)=={r['parameter'] for r in manifest['tensors']},'exact loaded parameter keys')
        mx.eval(*parameters.values())
        ledger=Ledger(args.out,result,mx,started)
        ledger.phase('after_weights_load',{'weights':parameters})
        result['parameters_before']=parameter_records(mx,np,parameters,args.out,'parameters-before')
        require_finite(result['parameters_before'])
        ledger.phase('after_parameters_before',{'weights':parameters}); save()
        for length in LENGTHS:
            for kind in KINDS:
                result['cases'].append(one_case(mx,np,gdn,caches.GDNCache,integration,layer,parameters,
                    result['parameters_before'],args.arm,length,kind,args.out,ledger)); save()
        result['parameters_after']=parameter_records(mx,np,flat_parameters(layer),args.out,'parameters-after')
        require_finite(result['parameters_after'])
        need(without_files(result['parameters_before'])==without_files(result['parameters_after']),'final parameters changed')
        ledger.phase('after_parameters_after',{'weights':parameters})
        del parameters,layer
        gc.collect(); mx.synchronize(); mx.clear_cache()
        need(layer_ref() is None,'layer remains referenced after explicit release')
        need(verify_sources(args.package_root)==result['sources'],'sources changed during run')
        weights.verify_bundle(args.bundle_manifest,args.bundle_sha256)
        # norm_metal intentionally retains its one epsilon array in EPS_VALUES.
        # Report that known global root; an empty caller tree is not proof of
        # zero live device arrays or allocator/runtime memory.
        norm=importlib.import_module('tutorial_code.miniengine.kernels.norm_metal')
        final=ledger.phase('after_layer_release',{'persistent_norm_eps':norm.EPS_VALUES})
        result['release']=dict(layer_weakref_dead=True,initial_active_bytes=initial_active,
            final_active_bytes=final['active_bytes'],active_delta_bytes=final['active_bytes']-initial_active,
            known_persistent_buffer_bytes=final['unique_buffer_bytes'],
            scope='Python layer is gone; active delta includes runtime/global roots, not a total-allocation-release assertion')
        need(ledger.count==68 and len(result['cases'])==8,'incomplete phase/case matrix')
        result.update(status='PASS',source_pins_rechecked=True,bundle_rechecked=True,
                      cross_arm_equality='PENDING_INDEPENDENT_RAW_AUDIT',performance_qualified=False)
    except BaseException as error:
        result.update(status='FAIL',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in previous.items(): signal.signal(sig,handler)
        result['elapsed_wall_s']=time.monotonic()-started
        save()
    print(json.dumps(dict(status=result['status'],out=str(args.out))))
    return int(result['status']!='PASS')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--bundle-manifest',type=Path)
    parser.add_argument('--bundle-sha256')
    parser.add_argument('--package-root',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    if not args.run:
        if args.out or args.bundle_manifest or args.bundle_sha256 or args.package_root:
            parser.error('execution paths require --run; default has no side effects')
        print(json.dumps(contract(args.arm),indent=2)); return 0
    if any(getattr(args,k) is None for k in ('arm','bundle_manifest','bundle_sha256','package_root','out')):
        parser.error('--run requires arm, bundle-manifest, bundle-sha256, package-root and out')
    return execute(args)


if __name__=='__main__':
    raise SystemExit(main())
