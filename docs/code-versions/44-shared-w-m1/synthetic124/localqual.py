"""V124 synthetic local strict gate. Default inert; GPU imports only in gpu()."""
from pathlib import Path
import argparse,copy,gc,hashlib,importlib.metadata,importlib.util,json,os,sys,traceback
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;P=D.parent;SOURCE=P/'v124-shared-weight-m1-source-build1'
PYTHON='/Users/ryanli/code/minivllm/.venv/bin/python'
CONTROLS={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
SHAPES=((48,5120),(17408,5120),(5120,6144),(5120,17408));ROWS=(2,4,5);BANKS=('random','codes_signed_boundaries','extreme_cancellation');LIMIT=1<<30;CACHE=64<<20
SCALE_CODES=(0x00,0x80,0x01,0x81,0x30,0x38,0x40,0xb8,0x7e,0xfe)
SOURCE_MANIFEST='a633fa7557edccb725101150772c4ea0fd7a492e889dad8e60be91a9fb967e9f'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def selected(env):return {k:v for k,v in env.items() if k.startswith(('MLX_','METAL_','DYLD_','OMLX_','MONOLITH_','LITHOS_')) or k in ('PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE')}
def source_helper():
 s=importlib.util.spec_from_file_location('v124_frozen_shared_m1',SOURCE/'shared_m1.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def identity():
 assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules),'MLX preimported'
 assert sha(SOURCE/'manifest.json')==SOURCE_MANIFEST
 m=json.loads((SOURCE/'manifest.json').read_text())
 for n,v in m['files'].items():assert sha(SOURCE/n)==v['sha256'],n
 pins=json.loads((D/'pins.json').read_text())
 for n,v in pins['files'].items():assert sha(n)==v,n
 return {'status':'PREIMPORT_PINS_PASS','source_manifest_sha256':SOURCE_MANIFEST,'local_pins_sha256':sha(D/'pins.json'),'files':pins['files'],'GPU_used':False}

class StrictMismatch(Exception):pass
class InvalidFixture(Exception):pass

def gpu(out,ident):
    import numpy as np
    import mlx.core as mx
    sys.path.insert(0,str(P))
    from resource_snapshot_power_v1 import snapshot,deltas,guard
    assert sys.executable==PYTHON and importlib.metadata.version('mlx')==importlib.metadata.version('mlx-metal')=='0.32.3'
    assert selected(os.environ)==CONTROLS,'selected environment differs'
    device=mx.metal.device_info();assert mx.default_device()==mx.gpu and device['device_name']=='Apple M4 Max' and device['architecture']=='applegpu_g16s'
    mx.set_memory_limit(LIMIT);mx.set_cache_limit(CACHE)
    write(out/'runtime.json',{'mlx_file':mx.__file__,'numpy_file':np.__file__,'mlx_version':'0.32.3','numpy_version':np.__version__,'device':device,'controls':CONTROLS,'memory_limit_bytes':LIMIT,'allocator_cache_limit_bytes':CACHE,'model_loaded':False,'checkpoint_read':False,'math_policy_override':False,'performance_run':False})
    helper=source_helper();checks=[];cases=[];readonly=[];resources=[];current={};error=None
    def probe(label):
        mx.synchronize();raw=snapshot(label,os.getpid());write(out/('resource-%02d-raw.json'%len(resources)),raw)
        r=copy.deepcopy(raw);r['mlx_memory']={'active':mx.get_active_memory(),'cache':mx.get_cache_memory(),'peak':mx.get_peak_memory()};resources.append(r);deltas(resources);write(out/'resources.json',resources);guard(resources)
        assert r['mlx_memory']['active']<=LIMIT and r['mlx_memory']['peak']<=LIMIT,'1GiB active/peak hard gate'
        assert r['power_metadata']['status']=='known' and r['power_metadata']['source']=='AC Power'
        assert r['processinfo_thermal_metadata']['status']=='known' and r['processinfo_thermal_metadata']['code'] in (0,1)
        assert r['pressure_level_value']==1 and r['exact_owned_pid_metadata']['status']=='known'
        assert all(x['owned'] for x in r['known_inference_processes']),'known foreign inference process'
        assert r['swap_used_mb']<=resources[0]['swap_used_mb'] and r['vm_counters']['Swapouts']==resources[0]['vm_counters']['Swapouts']
    def observed(a):
        v=a.view(mx.uint16) if a.dtype==mx.bfloat16 else a;mx.eval(v);b=memoryview(v)
        assert b.c_contiguous and b.nbytes==a.nbytes,'canonical zero-copy buffer required'
        return tuple(s//b.itemsize for s in b.strides),b.cast('B')
    def fingerprint(a):
        strides,b=observed(a);return {'shape':list(a.shape),'dtype':str(a.dtype),'strides':list(strides),'nbytes':a.nbytes,'sha256':hashlib.sha256(b).hexdigest(),'owner_id':id(a)}
    def ro(label):
        if not current:return
        after={k:fingerprint(v) for k,v in current['owners'].items()};ok=after==current['before'];readonly.append({'label':label,'case':current['case'],'before':current['before'],'after':after,'unchanged':ok});write(out/'readonly.json',readonly);assert ok,'input owner/bytes mutation'
    def compare(reference,actual,label,case,x):
        mx.eval(reference,actual);assert reference.shape==actual.shape==(case['T'],case['N']) and reference.dtype==actual.dtype==mx.bfloat16
        rb=bytes(observed(reference)[1]);ab=bytes(observed(actual)[1]);rf=bool(mx.all(mx.isfinite(reference)).item());af=bool(mx.all(mx.isfinite(actual)).item())
        rec={'case':case,'label':label,'shape':list(actual.shape),'dtype':'bfloat16','reference_finite':rf,'actual_finite':af,'reference_sha256':hashlib.sha256(rb).hexdigest(),'actual_sha256':hashlib.sha256(ab).hexdigest(),'nbytes':len(rb),'exact':rf and af and rb==ab,'row_checks':[{'row':i,'reference_sha256':hashlib.sha256(rb[i*case['N']*2:(i+1)*case['N']*2]).hexdigest(),'actual_sha256':hashlib.sha256(ab[i*case['N']*2:(i+1)*case['N']*2]).hexdigest()} for i in range(case['T'])]}
        checks.append(rec);write(out/'checks.json',checks)
        if not rf:
            write(out/'first-failure.json',{**rec,'stage':'INVALID_FIXTURE_NATIVE_NONFINITE','kernel_failure_claim':False});raise InvalidFixture('native reference is nonfinite; fixture invalid, not candidate failure')
        if not rec['exact']:
            aw=np.frombuffer(ab,dtype='<u2').reshape(actual.shape);rw=np.frombuffer(rb,dtype='<u2').reshape(reference.shape);diff=np.argwhere(aw!=rw);rec.update({'differing_elements':len(diff),'first_index':diff[0].tolist() if len(diff) else None,'stage':label,'first_word_actual':int(aw[tuple(diff[0])]) if len(diff) else None,'first_word_reference':int(rw[tuple(diff[0])]) if len(diff) else None})
            write(out/'first-failure.json',rec);(out/'diagnostic-reference.bf16.raw').write_bytes(rb);(out/'diagnostic-actual.bf16.raw').write_bytes(ab);(out/'diagnostic-input.bf16.raw').write_bytes(observed(x)[1]);raise StrictMismatch('strict BF16 bytes including signed zero; first mismatch STOP')
    def weights(n,k,bank,seed):
        rng=np.random.default_rng(seed)
        if bank=='random':w=rng.integers(0,1<<32,size=(n,k//8),dtype=np.uint32)
        else:
            pair=np.array([0x76543210,0xfedcba98] if bank=='codes_signed_boundaries' else [0x77777777,0xffffffff],dtype=np.uint32)
            w=np.tile(pair,(n,k//16))
        scales=np.resize(np.array(SCALE_CODES,dtype=np.uint8),n*(k//16)).reshape(n,k//16)
        return mx.array(w,dtype=mx.uint32),mx.array(scales,dtype=mx.uint8)
    def activations(k,bank,seed):
        if bank=='random':return mx.array(np.clip(np.random.default_rng(seed).standard_normal((5,k)),-4,4).astype(np.float32),dtype=mx.bfloat16)
        words=np.zeros((5,k),dtype=np.uint16)
        if bank=='codes_signed_boundaries':
            for row in range(5):
                words[row,1::2]=0x8000
                for j,pos in enumerate((0,15,16,31,32,511,512,k-1)):words[row,(pos+row)%k]=(0x3f80,0xbf80,0x3f00,0xbf00)[(j+row)%4]
            words[2]=words[0] # genuine repeated row alongside different rows
        else:
            pattern=np.array([0,0x8000,1,0x8001,0x5380,0xd380,0x3f80,0xbf80],dtype=np.uint16)
            for row in range(5):words[row]=np.resize(np.roll(pattern,row),k)
            words[2]=words[0]
        return mx.array(words,dtype=mx.uint16).view(mx.bfloat16)
    def run_case(w,s,op,prepared,control,x,meta):
        nonlocal current
        mx.eval(w,s,x);owners={'weight':w,'scales':s,'x':x};current={'case':meta,'owners':owners,'before':{k:fingerprint(v) for k,v in owners.items()}}
        try:
            # All references are computed from these fixed independent inputs, never supplied output values.
            arows=[mx.quantized_matmul(x[i:i+1],w,s,transpose=True,**helper.Q) for i in range(meta['T'])];native=mx.concatenate(arows,axis=0);mx.eval(native)
            if not bool(mx.all(mx.isfinite(native)).item()):
                compare(native,native,'native-fixture-finite',meta,x)
            crows=[control(inputs=[w,s,x[i:i+1]],template=[('T',mx.bfloat16)],grid=(64,meta['N']//8,1),threadgroup=(64,1,1),output_shapes=[(1,meta['N'])],output_dtypes=[mx.bfloat16])[0] for i in range(meta['T'])];c=mx.concatenate(crows,axis=0)
            compare(native,c,'original-expression-control-vs-native-M1',meta,x);ro('after-source-control')
            b=prepared.project(x,observed_x_strides=observed(x)[0]);compare(native,b,'shared-W-vs-native-M1',meta,x);ro('after-shared')
            cases.append({**meta,'input_metadata':current['before'],'strict_matrix_checks':2,'strict_rows_each_arm':meta['T']});write(out/'cases.json',cases)
        finally:ro('case-finally');current={}
    try:
        probe('before-synthetic-weights')
        for shape_index,(n,k) in enumerate(SHAPES):
            control=mx.fast.metal_kernel(name='v124_local_original_expression_control_n%d_k%d'%(n,k),input_names=['weight','scales','x'],output_names=['output'],source=(SOURCE/'control_body.metal').read_text(),header=(SOURCE/'kernel_header.metal').read_text()+'\nconstant int v124_k = %d;\nconstant int v124_n = %d;\n'%(k,n),ensure_row_contiguous=False)
            for bank_index,bank in enumerate(BANKS):
                seed=124000+shape_index*100+bank_index;w,s=weights(n,k,bank,seed);mx.eval(w,s)
                op=type('OriginalSyntheticOp',(),{})();op.weight=w;op.scales=s;op.q=dict(helper.Q);op.biases=None
                prepared=helper.prepare(mx,op,weight_strides=observed(w)[0],scale_strides=observed(s)[0],runtime_version='0.32.3',device_name=device['device_name'],architecture=device['architecture']);assert prepared.weight is w and prepared.scales is s
                pool=activations(k,bank,seed+19);mx.eval(pool)
                for t in ROWS:run_case(w,s,op,prepared,control,pool[:t],{'N':n,'K':k,'T':t,'bank':bank,'seed':seed,'row_permutation':False})
                if bank=='codes_signed_boundaries':
                    x=mx.take(pool,mx.array([4,2,0,3,1],dtype=mx.int32),axis=0);mx.eval(x);run_case(w,s,op,prepared,control,x,{'N':n,'K':k,'T':5,'bank':bank,'seed':seed,'row_permutation':True});del x
                del prepared,op,w,s,pool;gc.collect();mx.synchronize();probe('shape%d-%s-complete'%(shape_index,bank))
            del control;gc.collect()
        assert len(cases)==40 and len(checks)==80 and all(x['exact'] for x in checks)
    except BaseException as e:
        error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()};write(out/'error.json',error)
    finally:
        final_error=None
        try:ro('worker-finally');probe('finally')
        except BaseException as e:final_error={'type':type(e).__name__,'message':str(e)}
        status='PASS_LOCAL_V124_SHARED_WEIGHT_M1_STRICT_SYNTHETIC' if error is None and final_error is None else 'FAIL_STOP_LOCAL_V124'
        write(out/'receipt.json',{'status':status,'source_manifest_sha256':SOURCE_MANIFEST,'local_pins_sha256':ident['local_pins_sha256'],'cases_completed':len(cases),'strict_matrix_checks':len(checks),'all_exact':all(x['exact'] for x in checks),'strict_rows_per_arm_completed':sum(x['T'] for x in cases),'planned_cases':40,'planned_matrix_checks':80,'synthetic_only':True,'model_loaded':False,'performance_run':False,'full_model_qualified':False,'adopted':False,'readonly_records':len(readonly),'readonly_all_pass':all(x['unchanged'] for x in readonly),'maximum_observed_active':max((x['mlx_memory']['active'] for x in resources),default=0),'maximum_observed_peak':max((x['mlx_memory']['peak'] for x in resources),default=0),'error':error,'final_resource_or_readonly_error':final_error})
    if error or final_error:raise SystemExit(1)

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-gpu',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args();ident=identity()
 if not a.execute_gpu:print(json.dumps({'status':'INERT_CPU_SOURCE_PINS_PASS','GPU_used':False,'source_manifest_sha256':SOURCE_MANIFEST,'planned_cases':40,'planned_matrix_checks':80}));return
 assert a.out is not None and a.out.is_absolute() and not a.out.exists(),'fresh absolute output required';a.out.mkdir();write(a.out/'preimport-identity.json',ident);gpu(a.out,ident)
if __name__=='__main__':main()
