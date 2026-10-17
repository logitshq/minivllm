"""CPU-only E028 v2/v3 full-byte prefix and native concat reconstruction.

No MLX/NumPy, candidate imports, model loading, process or service actions.
Every mismatch is retained in JSONL; no tolerance, sampling or skipped tail.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / '.artifacts/m5-ultra-256gb'


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(path.read_text())
def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def byte_compare(left,right,label,out):
    """Both bytes objects and files are compared fully; enumerate every mismatch."""
    import io
    a=io.BytesIO(left) if isinstance(left,bytes) else left.open('rb')
    b=io.BytesIO(right) if isinstance(right,bytes) else right.open('rb')
    ah,bh=hashlib.sha256(),hashlib.sha256(); offset=diffs=na=nb=0; first=None; sink=None
    mismatch_path=out/(label.replace('/','__')+'-differences.jsonl')
    try:
        while True:
            x,y=a.read(4<<20),b.read(4<<20)
            if not x and not y:break
            ah.update(x);bh.update(y);na+=len(x);nb+=len(y)
            if x!=y:
                if sink is None:sink=mismatch_path.open('x')
                for i in range(max(len(x),len(y))):
                    p=x[i] if i<len(x) else None; q=y[i] if i<len(y) else None
                    if p!=q:
                        at=offset+i
                        if first is None:first=at
                        diffs+=1;sink.write(json.dumps(dict(byte=at,expected=p,actual=q))+'\n')
            offset+=max(len(x),len(y))
    finally:
        a.close();b.close()
        if sink is not None:sink.close()
    return dict(status='PASS' if diffs==0 else 'FAIL',expected_bytes=na,actual_bytes=nb,
        expected_sha256=ah.hexdigest(),actual_sha256=bh.hexdigest(),differing_bytes=diffs,
        first_differing_byte=first,differences_file=mismatch_path.name if diffs else None)


def tensor_file(path):
    raw=path.read_bytes();need(len(raw)>=8,'safetensors prefix')
    n=struct.unpack('<Q',raw[:8])[0];need(0<n<=(1<<20) and 8+n<=len(raw),'header bounds')
    def unique(pairs):
        out={}
        for k,v in pairs:need(k not in out,'duplicate header key');out[k]=v
        return out
    header=json.loads(raw[8:8+n],object_pairs_hook=unique);data=raw[8+n:];spans=[];result={}
    for key,row in header.items():
        if key=='__metadata__':continue
        need(set(row)=={'dtype','shape','data_offsets'},'tensor metadata keys')
        dtype,shape,(lo,hi)=row['dtype'],row['shape'],row['data_offsets']
        need(dtype in ('BF16','F32') and all(type(i) is int and i>=0 for i in shape),'dtype/shape')
        need(type(lo) is int and type(hi) is int and 0<=lo<=hi<=len(data),'offsets')
        need(hi-lo==math.prod(shape)*(2 if dtype=='BF16' else 4),'tensor byte count')
        spans.append((lo,hi));result[key]=dict(dtype=dtype,shape=shape,raw=data[lo:hi])
    cursor=0
    for lo,hi in sorted(spans):need(lo==cursor,'payload overlap/gap');cursor=hi
    need(cursor==len(data),'payload trailing bytes')
    return result


def tensor(tensors,key,shape,dtype):
    value=tensors[key];need(value['shape']==shape and value['dtype']==dtype,'source metadata: '+key)
    return value['raw']


def reconstruct(source,B):
    """Literal identity-row concat, native per-head prefix padding; no arithmetic."""
    need(set(source)=={'old_'+k for k in ('K','V','conv','S')}|
         {f'new{i}_{k}' for i in (0,1) for k in ('K','V','conv','S')},'input keys')
    output={}
    for key in ('K','V'):
        value=bytearray(tensor(source,'old_'+key,[B,4,1536,256],'BF16'))
        for i,offset in enumerate((935,954)):
            staging=tensor(source,f'new{i}_{key}',[1,4,1024,256],'BF16')
            for head in range(4):
                lo=head*1024*256*2
                value.extend(staging[lo:lo+offset*256*2])
                value.extend(bytes((1536-offset)*256*2))
        output[key]=dict(shape=[B+2,4,1536,256],dtype='BF16',raw=bytes(value))
    for key,shape,dtype in [('conv',[3,10240],'BF16'),('S',[48,128,128],'F32')]:
        value=tensor(source,'old_'+key,[B,*shape],dtype)
        value+=b''.join(tensor(source,f'new{i}_{key}',shape,dtype) for i in (0,1))
        output[key]=dict(shape=[B+2,*shape],dtype=dtype,raw=value)
    return output


def run(out):
    v2=BASE/'identity-refill-v2-owner/child/screen';v3=BASE/'identity-refill-v3-owner/child/screen'
    complete2,complete3=read(v2/'complete.json'),read(v3/'complete.json')
    need(complete2['status']=='FAIL' and len(complete2['checks'])==11,'v2 prefix identity')
    need(complete3['status']=='PASS' and len(complete3['checks'])==88,'v3 run identity')
    left={p.relative_to(v2).as_posix():p for p in v2.rglob('*.safetensors')}
    right={p.relative_to(v3).as_posix():p for p in v3.rglob('*.safetensors')}
    common=sorted(left.keys()&right.keys());left_only=sorted(left.keys()-right.keys())
    need(len(left)==16 and len(right)==108 and len(common)==14,'actual prefix file counts')
    need(left_only==[f'cases/identity-4-to-6/{a}-inputs-on-error.safetensors' for a in ('OFF','ON')],'unmatched v2 evidence')
    files={}
    def note(path):
        relative=path.relative_to(ROOT).as_posix()
        if relative not in files:files[relative]=dict(bytes=path.stat().st_size,sha256=sha(path))
    for path in (v2/'complete.json',v3/'complete.json'):note(path)
    compared=[]
    for name in common:
        note(left[name]);note(right[name])
        compared.append(dict(kind='same_relative_path',v2=name,v3=name,
            **byte_compare(left[name],right[name],'same/'+name,out)))
    for name in left_only:
        match=name.replace('-inputs-on-error.','-inputs-after-rebuild.')
        note(left[name]);note(right[match])
        compared.append(dict(kind='v2_error_to_v3_same_prewrite_boundary',v2=name,v3=match,
            **byte_compare(left[name],right[match],'error/'+name,out)))
    reconstruction=[]
    for B in (2,4):
        name=f'identity-{B}-to-{B+2}';case_root=v3/'cases'/name
        metadata=read(case_root/'case.json');note(case_root/'case.json')
        expected_meta=dict(req_ids=[f'old-{i}' for i in range(B)]+['new-0','new-1'],
            lens=[920+7*i for i in range(B)]+[935,954])
        for arm in ('OFF','ON'):
            need(all(metadata['metadata'][arm][k]==v for k,v in expected_meta.items()),'IDs/lens')
            source=case_root/f'{arm}-inputs-before.safetensors';note(source)
            expected=reconstruct(tensor_file(source),B)
            stages=['first-eval','after-source-write']+(['after-output-write'] if B==4 else [])
            for stage in stages:
                actual_file=case_root/f'{arm}-outputs-{stage}.safetensors';note(actual_file)
                actual=tensor_file(actual_file);need(set(actual)==set(expected),'output keys')
                for key,reference in expected.items():
                    row=actual[key];need(row['shape']==reference['shape'] and row['dtype']==reference['dtype'],'output metadata')
                    wanted=reference['raw']
                    if stage=='after-output-write':
                        scalar=struct.pack('<f',-32.0);scalar=scalar[2:] if reference['dtype']=='BF16' else scalar
                        wanted=scalar+wanted[len(scalar):]
                    reconstruction.append(dict(case=name,arm=arm,stage=stage,tensor=key,
                        shape=reference['shape'],dtype=reference['dtype'],source=source.relative_to(v3).as_posix(),
                        actual=actual_file.relative_to(v3).as_posix(),
                        **byte_compare(wanted,row['raw'],f'concat/{name}/{arm}/{stage}/{key}',out)))
    need(len(reconstruction)==40,'full B2/B4 output reconstruction coverage')
    passed=all(row['status']=='PASS' for row in compared+reconstruction)
    report=dict(schema='E028-v3-cross-version-byte-audit-v1',status='PASS' if passed else 'FAIL',
        method='stdlib-only safetensors parse and full bytes; no MLX/runtime import; every differing byte retained if found',
        v2_status='FAIL_RESOURCE',v3_reported_status='PASS',
        cross_version=dict(v2_files=16,v3_files=108,same_path_files=14,semantic_error_files=2,
            comparisons=compared,v3_paths_without_v2_counterpart=sorted(right.keys()-left.keys()),
            compared_bytes_per_version=sum(row['expected_bytes'] for row in compared)),
        reconstruction=dict(tensor_comparisons=40,metadata_and_IDS_lens_pass=True,comparisons=reconstruction,
            compared_bytes_per_side=sum(row['expected_bytes'] for row in reconstruction),
            rule='Old identity rows verbatim; each new K/V head copies offset935/954 then zero pads to1536; conv/S row concat. B4 reverse output write changes first scalar only to -32.'),
        scope='Checks synthetic prefix identity and B2/B4 state reconstruction only. Main v3 audit separately owns all10cases, resources, sources and lifecycle.',
        full_model_qualified=False,performance_qualified=False,
        auditor=dict(path=Path(__file__).relative_to(ROOT).as_posix(),sha256=sha(Path(__file__))))
    manifest=dict(schema='E028-cross-version-input-manifest-v1',files=files)
    write(out/'inputs-manifest.json',manifest)
    report['inputs_manifest_sha256']=sha(out/'inputs-manifest.json')
    write(out/'result.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    need(not (args.out/'result.json').exists(),'result already exists; use --out NEW')
    try:report=run(args.out)
    except BaseException as error:
        import traceback
        report=dict(schema='E028-v3-cross-version-byte-audit-v1',status='FAIL',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        write(args.out/'result.json',report)
    print(json.dumps({k:report[k] for k in ('schema','status')}))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
