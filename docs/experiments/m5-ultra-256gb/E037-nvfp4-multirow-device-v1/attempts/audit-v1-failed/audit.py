"""Independent E037 raw-byte audit. Standard library only; no device work."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import traceback

EXPECTED_PINS = {
 'candidate.py':'ee9c5eef0e1282125310ee6661be1ec5ce64dc93a82add48a6d6c17f8958816f',
 'nvfp4_qmv.py':'d79f6b3c44eb0003f9c6a4b5ece79c1d8bdd896fc6095c4458e4ea400b1c9c32',
 'smoke.py':'cac89f54ce7ded33e09b7b8e6bfa61caa03af1ae80f2c36b4b2e658afa0e8213'}
SHAPES=((512,64),(5120,256),(5120,6144))
DTYPES={'BF16':2,'U32':4,'U8':1}
def need(ok,msg):
    if not ok: raise ValueError(msg)
def unique(pairs):
    d={}
    for k,v in pairs:
        need(k not in d,'duplicate JSON key');d[k]=v
    return d
def read(p):return json.loads(p.read_bytes(),object_pairs_hook=unique)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def tensors(path,expected):
    raw=path.read_bytes();need(len(raw)>=8,'short safetensors')
    n=struct.unpack('<Q',raw[:8])[0];need(2<=n<=1<<20 and n+8<=len(raw),'header bound')
    header=json.loads(raw[8:8+n],object_pairs_hook=unique);header.pop('__metadata__',None)
    need(set(header)==set(expected),'tensor key coverage: '+str(path))
    result={};spans=[]
    for key,(dtype,shape) in expected.items():
        m=header[key];need(set(m)=={'dtype','shape','data_offsets'} and m['dtype']==dtype and m['shape']==shape,'dtype/shape: '+key)
        lo,hi=m['data_offsets'];need(type(lo) is type(hi) is int and 0<=lo<=hi and hi-lo==math.prod(shape)*DTYPES[dtype],'tensor span')
        result[key]=raw[8+n+lo:8+n+hi];need(len(result[key])==hi-lo,'short tensor');spans.append((lo,hi))
    cursor=0
    for lo,hi in sorted(spans):need(lo==cursor,'gap/overlap');cursor=hi
    need(8+n+cursor==len(raw),'trailing/missing payload');return result
def finite(raw):
    return all((v[0]&0x7f80)!=0x7f80 for v in struct.iter_unpack('<H',raw))
def bf16(value):
    raw=struct.pack('<f',value);need(raw[:2]==b'\x00\x00','recipe not exactly BF16');return raw[2:]
def compare(a,b,N):
    need(len(a)==len(b),'paired output length')
    first=next((i for i in range(0,len(a),2) if a[i:i+2]!=b[i:i+2]),None)
    return dict(equal=a==b,first_mismatch=None if first is None else dict(row=first//(2*N),column=first//2%N,left=a[first:first+2].hex(),right=b[first:first+2].hex()))
def run(E):
    D=E/'device';X=E/'execution';C=E/'smoke-cpu'
    driver=read(X/'final.json');inv=read(X/'invocation.json');producer=read(D/'result.json');di=read(D/'invocation.json')
    need(driver['command']==inv['command'] and driver['returncode']==0 and not driver.get('parent_timeout') and driver['case_receipts']==36,'actual execution closure')
    need(driver['status']=='PASS_SYNTHETIC_BYTES_ONLY' and driver['performance_qualified'] is False,'driver scope')
    need(inv['command'][1:3]==['-I','-B'] and inv['command'][4:6]==['--run','--out'] and len(inv['command'])==7,'fixed explicit command')
    need(di['argv']==inv['command'][3:] and di['deadline_seconds']==120 and di['limit_bytes']==256<<20,'child invocation')
    origin=Path(inv['command'][-1]);need(origin.is_absolute(),'absolute original output')
    need(read(X/'stdout.txt')==producer and (X/'stderr.txt').read_bytes()==b'','actual stdout/stderr')
    need(producer['status']=='PASS_SYNTHETIC_BYTES_ONLY' and producer['cases']==36 and producer['full_model_qualified'] is False and producer['performance_qualified'] is False,'producer scope')
    runtime=read(D/'runtime.json');need(runtime['version']=='0.32.3' and runtime['name']=='Apple M5 Ultra' and runtime['architecture']=='applegpu_g17d','recorded runtime')
    incomplete=read(D/'incomplete.json');need(incomplete['status']=='INCOMPLETE_UNTIL_RESULT','retain initial intent')
    source_checks={}
    for name,h in EXPECTED_PINS.items():
        p=C/'sources'/name;need(sha(p)==h,'frozen source pin '+name);source_checks[name]=dict(path=str(p.relative_to(E)),sha256=h)
    need(sha(X/'smoke-source.py')==di['smoke_sha256']==EXPECTED_PINS['smoke.py'],'actually captured smoke source')
    need({Path(p).name:h for p,h in di['pins'].items()}=={k:v for k,v in EXPECTED_PINS.items() if k!='smoke.py'} and len(di['pins'])==2,'child source binding')
    need(Path(inv['command'][3]).name=='smoke.py','actual entry')
    cpu_results={}
    for dirname,count in (('smoke-cpu',1),('mechanism-cpu',5)):
        R=E/dirname;r=read(R/'result.json');before=read(R/'source-before.json');after=read(R/'source-after.json')
        need(r['status']=='PASS' and r['tests_run']==count and not any(r[k] for k in ('failures','errors','skipped','forbidden')) and r['exception'] is None and before==after,'CPU receipt closure')
        for original,h in before.items():need(sha(R/'sources'/Path(original).name)==h,'CPU archived source')
        cpu_results[dirname]=dict(tests=count,source_unchanged=True,scope=r['scope'])
    def receipt(ref,relative):
        need(set(ref)=={'path','bytes','sha256'} and Path(ref['path']).relative_to(origin)==Path(relative),'original artifact path')
        p=D/relative;need(p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],'retained artifact SHA');return p
    cases=[];expected_files={'invocation.json','incomplete.json','runtime.json','result.json'};source_values=0;packed_values=0
    for K,N in SHAPES:
        base=f'K{K}-N{N}';folder=D/base;quant=read(folder/'quantization.json')
        packed=tensors(folder/'packed.safetensors',{'w':('U32',[N,K//8]),'scales':('U8',[N,K//16])});packed_values+=sum(len(v) for v in packed.values());del packed
        expected_files.update((base+'/quantization.json',base+'/packed.safetensors'))
        offsets=list(range(0,N,256));need(len(quant['source'])==len(offsets),'source band completeness')
        dense_words=[bf16((i-63)/16) for i in range(127)]
        patterns={phase:b''.join(dense_words[(col+phase)%127] for col in range(K)) for phase in range(127)}
        for off,ref in zip(offsets,quant['source']):
            relative=f'{base}/source-band-{off:05d}.safetensors';expected_files.add(relative)
            rows=min(256,N-off);raw=tensors(receipt(ref,relative),{'dense':('BF16',[rows,K])})['dense']
            need(all(raw[r*K*2:(r+1)*K*2]==patterns[((off+r)*13)%127] for r in range(rows)),'original BF16 source recipe')
            source_values+=rows*K
        for T in (2,4,8):
            for kind in (('finite','zero','negative-zero','stress') if K==512 else ('finite',)):
                layout_results=[]
                for stride in (1,2):
                    relative=f'{base}/T{T}-{kind}-stride{stride}';folder=D/relative
                    expected_files.update(relative+'/'+name for name in ('intent.json','inputs.safetensors','outputs.safetensors','result.json'))
                    intent=read(folder/'intent.json');result=read(folder/'result.json')
                    need({k:intent[k] for k in ('T','K','N','kind','column_stride2')}==dict(T=T,K=K,N=N,kind=kind,column_stride2=stride==2),'case intent')
                    receipt(intent['weights'],base+'/packed.safetensors')
                    inputs=tensors(receipt(result['inputs'],relative+'/inputs.safetensors'),{'x':('BF16',[T,K])})['x']
                    expected_x=(b'\x00\x00' if kind=='zero' else b'\x00\x80')*(T*K) if kind in ('zero','negative-zero') else b''.join(bf16(((i%37)-18)*(128 if kind=='stress' else 0.125)) for i in range(T*K))
                    need(inputs==expected_x and finite(inputs),'exact original x recipe including signed zero')
                    output=tensors(receipt(result['outputs'],relative+'/outputs.safetensors'),{key:('BF16',[T,N]) for key in ('native_b1','custom_b1','candidate')})
                    cmp={a+'__'+b:compare(output[a],output[b],N) for a,b in (('candidate','native_b1'),('candidate','custom_b1'),('custom_b1','native_b1'))}
                    hashes={k:hashlib.sha256(v).hexdigest() for k,v in output.items()}
                    fin=all(finite(v) for v in output.values());ok=fin and all(v['equal'] for v in cmp.values())
                    need(result['comparisons']==cmp and result['original_dtype_sha256']==hashes and result['finite']==fin and result['status']==('PASS' if ok else 'FAIL'),'producer/raw agreement')
                    need(set(result['memory'])=={'active','peak','cache','buffers'} and all(type(v) is int and 0<=v<256<<20 for v in result['memory'].values()),'recorded memory scope/bounds')
                    cases.append(dict(case=relative,T=T,K=K,N=N,kind=kind,declared_stride=stride,status='PASS' if ok else 'FAIL',finite=fin,comparisons=cmp,original_dtype_sha256=hashes,input_sha256=hashlib.sha256(inputs).hexdigest(),elements_per_route=T*N,reported_memory=result['memory']))
                    layout_results.append((inputs,output))
                need(layout_results[0]==layout_results[1],'saved logical values differ between declared layout pair')
    actual_files={str(p.relative_to(D)) for p in D.rglob('*') if p.is_file()}
    need(actual_files==expected_files and len(cases)==36,'complete exact raw inventory/case matrix')
    need(all(x['status']=='PASS' for x in cases),'raw case failure')
    peak=max(x['reported_memory']['peak'] for x in cases);need(producer['memory']==dict(active=0,peak=peak,cache=0,buffers=0),'reported cumulative final memory')
    return dict(schema='E037-independent-raw-audit-v1',status='PASS',cases=cases,case_count=36,output_matrices=108,
        elements_per_route=sum(x['elements_per_route'] for x in cases),pairwise_word_comparisons=3*sum(x['elements_per_route'] for x in cases),
        source_BF16_values_verified=source_values,packed_payload_bytes=packed_values,device_files=len(actual_files),source_checks=source_checks,cpu_results=cpu_results,
        reported_peak_bytes=peak,reported_peak_mib=peak/(1<<20),runtime_receipt=runtime,
        independently_proven=['Complete exact saved case matrix and safetensors layout/span coverage','Full original BF16 output bytes finite and three-way equal in all cases','Original x values including negative zero and all BF16 source-band values equal the fixed recipes','Packed weight/scale shapes, full-file hashes, source pins, driver/child/CPU receipt consistency','Declared stride1/2 pairs have identical saved logical inputs and outputs'],
        receipt_or_source_only=['Actual native M1 dispatch, NVFP4 quantizer execution and kernel compilation follow source/producer execution records; no hardware trace','Physical stride/view/staging behavior is not preserved by contiguous safetensors','Device/runtime identity and allocator active/cache/peak are producer records; not independent RSS/system memory measurements','Returned process exit and service coexistence are execution context, not a performance isolation proof'],
        full_model_qualified=False,performance_qualified=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=None
    try:result=run(a.evidence)
    except BaseException as e:result=dict(status='FAIL',error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','cpu_results','source_checks','independently_proven','receipt_or_source_only')},sort_keys=True))
    return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
