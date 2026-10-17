"""Bounded shared-W M1 numerical diagnostic. Default/help never import MLX.

--run --out NEW [--bundle-manifest PATH --bundle-sha256 SHA] runs 27 synthetic
cases, plus 6 real-weight/synthetic-X cases when the fixed E026 bundle is given.
Every case independently recomputes native M1 rows, checks the original-expression
control, then checks shared-W, including signed-zero bits. No speed measurement,
model loading, HTTP, propagated-state or full-model qualification is performed.
An external direct owner must enforce the deadline/resources/reap: a Python
120s alarm cannot preempt a stalled GPU. Cache=0; observed MLX budget=1GiB,
self physical readings each <=2GiB. These observations are not OS hard caps.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'docs/code-versions/44-shared-w-m1/kernel124'
LAYOUT_SOURCE = SOURCE.parent / 'real127/layout_probe.metal'
SUPPORT = HERE.parent / 'conv_tail_owned_copy'
SHAPES = ((48,5120),(1024,5120),(10240,5120))
ROWS = (2,4,5)
BANKS = ('random','codes_signed_boundaries','extreme_cancellation')
SCALE_CODES = (0x00,0x80,0x01,0x81,0x30,0x38,0x40,0xb8,0x7e,0xfe)
Q = dict(group_size=16,bits=4,mode='nvfp4')
LIMIT = 1 << 30
FIXED_PINS = {
    LAYOUT_SOURCE:'880b0ea0d9d92b01c6c1d1188610761d470d77ae40a44c77a507cc0d4aa2f8c0',
    SOURCE/'LICENSE-MLX.txt':'ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01',
    SOURCE/'kernel_header.metal':'17759af8d59a7ba185040630de07af69dd3335538e19094bd1f541c2b1140c6b',
    SOURCE/'control_body.metal':'81660cb7363649cddec120a22b65ad59544b5d4bee7e557b958dce7e801e0b1d',
    SOURCE/'kernel_body.metal':'54746243dbaf9b73734dcf8337691016c53b5b68e277afa01451d7a72dd3e2b0',
    SOURCE/'shared_impl.metal':'6dfe076f5a45af62aa14c644826791f3aa143d2e92bc230a0f07c39886fcecc9',
    SUPPORT/'layer_weights.py':'8ee0ebd9df9385ec35baf6d0a0ae80bf780cf80a5e255b34d6fe2fed560df77a',
    SUPPORT/'layer_screen.py':'a0423d0648b19236f972598ce5acb5e8e9cdd617fadb2646f1c7fef01d9ec302',
}


def need(ok,message):
    if not ok: raise ValueError(message)


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4<<20):h.update(block)
    return h.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def banks(with_bundle):
    result=[dict(id=f'synthetic-N{n}-{bank}',N=n,K=k,bank=bank,
                 seed=124000+i*100+j,real_weights=False)
            for i,(n,k) in enumerate(SHAPES) for j,bank in enumerate(BANKS)]
    if with_bundle:
        result += [dict(id=f'real-{name}',N=n,K=5120,bank='random',seed=126000+i,
                        real_weights=True,projection=name)
                   for i,(name,n) in enumerate((('in_proj_a',48),('in_proj_qkv',10240)))]
    return result


def contract(with_bundle=False):
    cases=[dict(case=f'{b["id"]}-T{t}',T=t,**b) for b in banks(with_bundle) for t in ROWS]
    return dict(schema='M5-shared-w-m1-screen-v1',cases=cases,expected_cases=len(cases),
        expected_comparisons=2*len(cases),expected_phases=2+2*len(cases),
        cache_limit_bytes=0,mlx_observation_budget_bytes=LIMIT,
        physical_observation_budget_bytes=2<<30,child_alarm_seconds=120,
        physical_reading_semantics='separate-int-bounds;current=max(top,aux-current);peak=max(top,aux-peak);aux-peak>=aux-current',
        quantization=Q,reference='concat independent native quantized_matmul M1 calls',
        gate='finite + exact BF16 bytes/shape/dtype including signed zero; control first; no tolerance',
        bank_inputs='M4 V124 three bank definitions; each raw input preserved; X is synthetic in every case',
        layout_evidence='pinned Metal GPU descriptor probe; host memoryview only describes raw host bytes',
        expected_layouts=10*len(banks(with_bundle)),
        counter_scope='Python callable construction and returned lazy graphs; raw eval independently proves completion',
        full_model_qualified=False,performance_qualified=False,observed_limits_not_OS_hard_caps=True)


def source_pins():
    pins={str(path):expected for path,expected in FIXED_PINS.items()}
    for path,expected in FIXED_PINS.items():need(sha(path)==expected,'fixed source SHA: '+str(path))
    for path in (HERE/'candidate.py',Path(__file__).resolve(),SOURCE/'LICENSE-MLX.txt'):
        pins[str(path)]=sha(path)
    return pins


def stop_signal(signum,frame=None):
    if signum==signal.SIGALRM:raise TimeoutError('120s shared-W screen SIGALRM deadline')
    raise InterruptedError('shared-W screen cancellation '+signal.Signals(signum).name)


def synthetic(np,n,k,bank,seed):
    rng=np.random.default_rng(seed)
    if bank=='random':
        weight=rng.integers(0,1<<32,size=(n,k//8),dtype=np.uint32)
        # Same finite clipped-normal source as M4 V124; conversion is BF16 on device.
        x=np.clip(np.random.default_rng(seed+19).standard_normal((5,k)),-4,4).astype(np.float32)
    else:
        pair=[0x76543210,0xfedcba98] if bank=='codes_signed_boundaries' else [0x77777777,0xffffffff]
        weight=np.tile(np.array(pair,dtype=np.uint32),(n,k//16))
        x=np.zeros((5,k),dtype=np.uint16)
        for row in range(5):
            if bank=='codes_signed_boundaries':
                x[row,1::2]=0x8000
                for j,pos in enumerate((0,15,16,31,32,511,512,k-1)):
                    x[row,(pos+row)%k]=(0x3f80,0xbf80,0x3f00,0xbf00)[(j+row)%4]
            else:
                pattern=np.array([0,0x8000,1,0x8001,0x5380,0xd380,0x3f80,0xbf80],dtype=np.uint16)
                x[row]=np.resize(np.roll(pattern,row),k)
        x[2]=x[0]
    scales=np.resize(np.array(SCALE_CODES,dtype=np.uint8),n*(k//16)).reshape(n,k//16)
    return x,weight,scales


def bundle_pair(np,path,manifest,projection):
    """Read only the two original packed tensors; no model or converted repack."""
    result={}
    with (Path(path).parent/'weights.bin').open('rb') as stream:
        for suffix,dtype in (('weight','<u4'),('scales','u1')):
            key=projection+'.'+suffix
            matches=[r for r in manifest['tensors'] if r['parameter']==key]
            need(len(matches)==1,'exact bundle projection tensor: '+key)
            row=matches[0];stream.seek(row['bundle_offsets'][0]);raw=stream.read(row['bytes'])
            need(hashlib.sha256(raw).hexdigest()==row['sha256'],'bundle tensor changed: '+key)
            result[suffix]=np.frombuffer(raw,dtype=dtype).reshape(row['shape']).copy()
    return result['weight'],result['scales']


def observe(mx,value):
    view=value.view(mx.uint16) if value.dtype==mx.bfloat16 else value
    mx.eval(view)
    buffer=memoryview(view)
    need(buffer.c_contiguous and buffer.nbytes==value.nbytes,'canonical host buffer required for raw bytes')
    return tuple(stride//buffer.itemsize for stride in buffer.strides),buffer.cast('B').tobytes()


def array_record(mx,value,out,file=None):
    strides,raw=observe(mx,value)
    dtypes={mx.bfloat16:('bfloat16','uint16-le'),mx.uint32:('uint32','uint32-le'),mx.uint8:('uint8','uint8')}
    need(value.dtype in dtypes,'unexpected array dtype')
    dtype,encoding=dtypes[value.dtype]
    row=dict(shape=list(value.shape),dtype=dtype,strides=list(strides),bytes=len(raw),
             sha256=hashlib.sha256(raw).hexdigest(),encoding=encoding,owner_id=id(value),
             finite=bool(mx.all(mx.isfinite(value)).item()) if value.dtype==mx.bfloat16 else None)
    if file is not None:
        target=out/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        row['file']=file
    return row


def without_file(row):return {k:v for k,v in row.items() if k!='file'}


def layout_probe(mx):
    return mx.fast.metal_kernel(name='m5_shared_w_observe_original_layout_v1',input_names=['value'],
        output_names=['descriptor'],source=LAYOUT_SOURCE.read_text(),ensure_row_contiguous=False)


def gpu_layout(mx,kernel,value,out,file,label):
    need(value.ndim==2,'GPU descriptor requires original rank2 operand')
    descriptor=kernel(inputs=[value],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(4,)],
                      output_dtypes=[mx.uint64])[0]
    mx.eval(descriptor)
    raw=memoryview(descriptor).cast('B').tobytes()
    need(len(raw)==32,'GPU descriptor byte size')
    words=list(struct.unpack('<4Q',raw))
    path=out/file;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    record=dict(label=label,input_owner_id=id(value),shape=words[:2],actual_strides_elements=words[2:],
        ensure_row_contiguous=False,source_sha256=FIXED_PINS[LAYOUT_SOURCE],
        descriptor=dict(file=file,sha256=hashlib.sha256(raw).hexdigest(),bytes=32,dtype='uint64',
                        shape=[4],encoding='uint64-le'))
    write(path.with_suffix('.json'),record)
    need(words[:2]==list(value.shape) and words[2:]==[value.shape[1],1],
         'actual GPU descriptor is not canonical; no normalization permitted')
    return record


def compare(out,case,label,left,right):
    meta=('shape','dtype','strides','bytes','encoding')
    a=(out/left['file']).read_bytes();b=(out/right['file']).read_bytes()
    need(hashlib.sha256(a).hexdigest()==left['sha256'] and hashlib.sha256(b).hexdigest()==right['sha256'],
         'saved output bytes/SHA mismatch')
    exact=all(left[k]==right[k] for k in meta) and left['finite'] is True and right['finite'] is True and a==b
    row=dict(case=case,label=label,status='PASS' if exact else 'FAIL',exact=exact,
             reference=left,actual=right)
    if not exact:
        row['first_differing_byte']=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
        if row['first_differing_byte'] is None and len(a)!=len(b):row['first_differing_byte']=min(len(a),len(b))
        if not (out/'first-difference.json').exists():write(out/'first-difference.json',row)
    return row


class Ledger:
    def __init__(self,out,result,mx,support,started):
        self.out,self.result,self.mx,self.support,self.started=out,result,mx,support,started
        self.count=0

    def phase(self,phase,case='global'):
        mx=self.mx
        row=dict(index=self.count,phase=phase,case=case,pid=os.getpid(),ppid=os.getppid(),
                 elapsed_s=time.monotonic()-self.started,active_bytes=int(mx.get_active_memory()),
                 peak_bytes=int(mx.get_peak_memory()),cache_bytes=int(mx.get_cache_memory()))
        path=self.out/f'physical-{self.count:03d}.raw.json'
        command=['/usr/bin/footprint','-p',str(os.getpid()),'--noCategories','-f','bytes','-j',str(path)]
        try:
            probe=subprocess.run(command,capture_output=True,text=True,timeout=5)
            write(self.out/f'physical-{self.count:03d}.command.json',
                  dict(command=command,returncode=probe.returncode,stdout=probe.stdout,stderr=probe.stderr))
            need(probe.returncode==0 and path.is_file(),'self footprint failed')
            raw=json.loads(path.read_text());current,peak=self.support.parse_footprint(raw,os.getpid())
            process=raw['processes'][0]
            receipt=dict(path=path.name,sha256=sha(path),pid=os.getpid(),phase=phase,case=case)
            self.result['physical_receipts'].append(receipt)
            row.update(physical_receipt=receipt,physical_current_bytes=current,physical_peak_bytes=peak,
                       physical_top_bytes=process['footprint'],
                       physical_aux_current_bytes=process['auxiliary']['phys_footprint'],
                       physical_aux_peak_bytes=process['auxiliary']['phys_footprint_peak'])
            need(all(type(row[k]) is int and 0<=row[k] for k in ('active_bytes','peak_bytes','cache_bytes')),
                 'invalid MLX memory reading')
            need(row['peak_bytes']<=LIMIT and row['active_bytes']+row['cache_bytes']<=LIMIT,
                 'MLX 1GiB observation budget exceeded')
            row['status']='PASS'
        except BaseException as error:
            row.update(status='FAIL',error_type=type(error).__name__,error=str(error));raise
        finally:
            with (self.out/'phases.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n');stream.flush()
            self.count+=1


def original_control(mx,n,k):
    return mx.fast.metal_kernel(name=f'm5_shared_w_original_control_n{n}_k{k}',
        input_names=['weight','scales','x'],output_names=['output'],
        source=(SOURCE/'control_body.metal').read_text(),
        header=(SOURCE/'kernel_header.metal').read_text()+f'\nconstant int v124_k = {k};\nconstant int v124_n = {n};\n',
        ensure_row_contiguous=False)


def one_case(mx,out,ledger,result,bank,t,pool,op,prepared,control,weight_before,layout_kernel):
    name=f'{bank["id"]}-T{t}';directory=out/'cases'/name;directory.mkdir(parents=True)
    row=dict(case=name,T=t,N=bank['N'],K=bank['K'],bank=bank['id'],status='INCOMPLETE')
    result['cases'].append(row)
    def save():write(directory/'case.json',row)
    x=pool[:t];outputs={};primary=None
    try:
        row['layout_before']=gpu_layout(mx,layout_kernel,x,out,f'cases/{name}/x-layout-before.bin',name+'/x-before')
        row['inputs_before']=dict(x=array_record(mx,x,out,f'cases/{name}/input-x-before.bin'),**weight_before)
        need(row['inputs_before']['x']['finite'] is True,'nonfinite synthetic input')
        save();ledger.phase('before_case',name)
        n,k=bank['N'],bank['K']
        native=mx.concatenate([mx.quantized_matmul(x[i:i+1],op.weight,op.scales,None,transpose=True,**Q)
                               for i in range(t)],axis=0)
        outputs['native']=array_record(mx,native,out,f'cases/{name}/native.bf16.bin')
        row['outputs']=outputs;save()
        need(outputs['native']['shape']==[t,n] and outputs['native']['dtype']=='bfloat16'
             and outputs['native']['finite'] is True,'native M1 reference invalid; no candidate claim')
        control_y=mx.concatenate([control(inputs=[op.weight,op.scales,x[i:i+1]],template=[('T',mx.bfloat16)],
                    grid=(64,n//8,1),threadgroup=(64,1,1),output_shapes=[(1,n)],output_dtypes=[mx.bfloat16])[0]
                    for i in range(t)],axis=0)
        outputs['control']=array_record(mx,control_y,out,f'cases/{name}/control.bf16.bin');save()
        check=compare(out,name,'original-expression-control-vs-native-M1',outputs['native'],outputs['control'])
        result['checks'].append(check);write(out/'checks.json',result['checks'])
        need(check['exact'],'original-expression control failed; candidate not admitted')
        count=prepared.launch_count
        shared=prepared.project(x,observed_x_strides=tuple(row['layout_before']['actual_strides_elements']))
        outputs['shared']=array_record(mx,shared,out,f'cases/{name}/shared.bf16.bin');save()
        row['route']=dict(launch_count_before=count,launch_count_after=prepared.launch_count,
                         kernel_build_count=prepared.kernel_build_count,required_route='shared-W',fallback=False)
        need(prepared.launch_count==count+1 and prepared.kernel_build_count==1,'required shared-W launch/build missing')
        check=compare(out,name,'shared-W-vs-native-M1',outputs['native'],outputs['shared'])
        result['checks'].append(check);write(out/'checks.json',result['checks'])
        need(check['exact'],'shared-W strict numerical mismatch')
    except BaseException as error:
        primary=type(error).__name__
        row.update(status='FAIL',error_type=primary,error=str(error));save();raise
    finally:
        try:
            row['layout_after']=gpu_layout(mx,layout_kernel,x,out,f'cases/{name}/x-layout-after.bin',name+'/x-after')
            need(all(row['layout_after'][k]==row['layout_before'][k] for k in
                     ('input_owner_id','shape','actual_strides_elements')),'original GPU input descriptor changed')
            after=dict(x=array_record(mx,x,out,f'cases/{name}/input-x-after.bin'),
                       weight=array_record(mx,op.weight,out),scales=array_record(mx,op.scales,out))
            row['inputs_after']=after
            before=row.get('inputs_before',{})
            unchanged=set(before)==set(after) and all(without_file(before[k])==without_file(v) for k,v in after.items())
            row['inputs_unchanged']=unchanged
            if not unchanged:
                for key,value in (('weight',op.weight),('scales',op.scales)):
                    array_record(mx,value,out,f'cases/{name}/changed-{key}.bin')
                raise ValueError('original input owner/bytes/metadata changed')
            ledger.phase('after_case',name)
            if primary is None:row['status']='PASS'
        except BaseException as error:
            row.update(status='FAIL',exit_error_type=type(error).__name__,exit_error=str(error))
            if primary is None:raise
        finally:save()


def execute(args):
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result=dict(status='INCOMPLETE',pid=os.getpid(),ppid=os.getppid(),device_run=False,
                contract=contract(args.bundle_manifest is not None),cases=[],checks=[],bank_receipts=[],
                physical_receipts=[],performance_qualified=False,full_model_qualified=False)
    def save():write(args.out/'complete.json',result)
    need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm unsupported')
    previous={sig:signal.signal(sig,stop_signal) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL,120);save()
    try:
        result['sources']=source_pins();save()
        for path in result['sources']:
            dst=args.out/'frozen-source'/Path(path).relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(Path(path).read_bytes())
        candidate=load('m5_shared_w_candidate',HERE/'candidate.py')
        support=load('m5_shared_w_memory_support',SUPPORT/'layer_screen.py')
        weights=load('m5_shared_w_bundle_support',SUPPORT/'layer_weights.py')
        manifest=weights.verify_bundle(args.bundle_manifest,args.bundle_sha256) if args.bundle_manifest else None
        result['bundle_manifest_sha256']=args.bundle_sha256
        result['bundle_sha256']=manifest['bundle']['sha256'] if manifest else None
        if manifest:write(args.out/'bundle-manifest.json',manifest)
        import numpy as np
        import mlx.core as mx
        mx.set_cache_limit(0);mx.synchronize();mx.clear_cache();mx.reset_peak_memory()
        device=mx.metal.device_info()
        runtime=importlib.metadata.version('mlx')
        need(runtime=='0.32.3' and mx.default_device()==mx.gpu and device['device_name']=='Apple M5 Ultra'
             and device['architecture']=='applegpu_g17d','pinned M5 Ultra MLX GPU required')
        result.update(device_run=True,runtime=dict(mlx_file=mx.__file__,numpy_file=np.__file__,mlx_version=runtime,
                                                  numpy_version=np.__version__,device=device))
        ledger=Ledger(args.out,result,mx,support,started);ledger.phase('initial')
        layout_kernel=layout_probe(mx)
        for bank in banks(manifest is not None):
            xh,wh,sh=synthetic(np,1 if bank['real_weights'] else bank['N'],bank['K'],bank['bank'],bank['seed'])
            if bank['real_weights']:wh,sh=bundle_pair(np,args.bundle_manifest,manifest,bank['projection'])
            pool=mx.array(xh).astype(mx.bfloat16) if xh.dtype==np.float32 else mx.array(xh).view(mx.bfloat16)
            op=SimpleNamespace(weight=mx.array(wh),scales=mx.array(sh),q=dict(Q),biases=None)
            del xh,wh,sh
            layouts={key:gpu_layout(mx,layout_kernel,getattr(op,key),args.out,
                     f'banks/{bank["id"]}/{key}-layout-before.bin',bank['id']+'/'+key+'-before')
                     for key in ('weight','scales')}
            wb={key:array_record(mx,getattr(op,key),args.out,f'banks/{bank["id"]}/{key}.bin')
                for key in ('weight','scales')}
            prepared=candidate.prepare(mx,op,enabled=True,weight_strides=tuple(layouts['weight']['actual_strides_elements']),
                scale_strides=tuple(layouts['scales']['actual_strides_elements']),runtime_version=runtime,
                device_name=device['device_name'],architecture=device['architecture'])
            need(prepared is not None and prepared.weight is op.weight and prepared.scales is op.scales,
                 'prepared input handles changed')
            control=original_control(mx,bank['N'],bank['K'])
            for t in ROWS:one_case(mx,args.out,ledger,result,bank,t,pool,op,prepared,control,wb,layout_kernel);save()
            need(prepared.launch_count==3 and prepared.kernel_build_count==1,'one kernel per three-row bank reuse')
            after={key:array_record(mx,getattr(op,key),args.out,f'banks/{bank["id"]}/{key}-after.bin')
                   for key in ('weight','scales')}
            layouts_after={key:gpu_layout(mx,layout_kernel,getattr(op,key),args.out,
                f'banks/{bank["id"]}/{key}-layout-after.bin',bank['id']+'/'+key+'-after')
                for key in ('weight','scales')}
            need(all(without_file(wb[k])==without_file(after[k]) for k in wb),'bank original weight bytes changed')
            need(all(layouts[k][field]==layouts_after[k][field] for k in layouts for field in
                     ('input_owner_id','shape','actual_strides_elements')),'bank original GPU weight layout changed')
            result['bank_receipts'].append(dict(bank=bank['id'],launch_count=prepared.launch_count,
                kernel_build_count=prepared.kernel_build_count,inputs_before=wb,inputs_after=after,
                layouts_before=layouts,layouts_after=layouts_after))
            del pool,op,prepared,control,wb
            gc.collect();mx.synchronize();mx.clear_cache();save()
        del layout_kernel
        gc.collect();mx.synchronize();mx.clear_cache()
        ledger.phase('final_release')
        for path,expected in result['sources'].items():need(sha(path)==expected,'source changed: '+path)
        result['source_pins_rechecked']=True
        if manifest:weights.verify_bundle(args.bundle_manifest,args.bundle_sha256)
        result['bundle_rechecked']=manifest is not None
        need(len(result['cases'])==result['contract']['expected_cases'] and all(r['status']=='PASS' for r in result['cases']),
             'case coverage incomplete')
        need(len(result['checks'])==result['contract']['expected_comparisons'] and all(r['exact'] for r in result['checks']),
             'comparison coverage incomplete')
        need(ledger.count==result['contract']['expected_phases'],'phase coverage incomplete')
        result['descriptor_count']=len(list(args.out.glob('banks/*/*-layout-*.bin')))+len(list(args.out.glob('cases/*/x-layout-*.bin')))
        need(result['descriptor_count']==result['contract']['expected_layouts'],'GPU descriptor coverage incomplete')
        result['status']='PASS'
    except BaseException as error:
        result.update(status='FAIL',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in previous.items():signal.signal(sig,handler)
        result['elapsed_wall_s']=time.monotonic()-started;save()
    print(json.dumps(dict(status=result['status'],out=str(args.out))))
    return int(result['status']!='PASS')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--out',type=Path)
    parser.add_argument('--bundle-manifest',type=Path);parser.add_argument('--bundle-sha256')
    args=parser.parse_args(argv)
    if not args.run:
        if args.out or args.bundle_manifest or args.bundle_sha256:parser.error('execution arguments require --run')
        print(json.dumps(contract(),indent=2));return 0
    if args.out is None or bool(args.bundle_manifest)!=bool(args.bundle_sha256):
        parser.error('--run requires --out; bundle manifest and SHA must be paired')
    return execute(args)


if __name__=='__main__':raise SystemExit(main())
