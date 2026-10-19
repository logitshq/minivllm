"""Explicit local-projection screening; deterministic inputs are NOT real hidden states."""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import signal
import statistics
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
MODEL = Path('/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6')
MANIFEST = ROOT/'bench/m5_ultra_256gb/omlx_d512_formal/prereg-formal-v1/inputs/model-manifest.json'
PINS = {str(Path(__file__).with_name('candidate.py')):'361fe483cde02a13c505287652249016f74b16d1d179c3741da843a3060cf71a',
        str(MANIFEST):'de8ae0d26b5db015b9dc690bcaaec86cbbd5f1435c439baca22c589c7441fa1d'}
PROJECTIONS = (('head','language_model.lm_head',248320,3,'910ede76e14d0da40deb3bca75126c4a4989641e2f664c927adf58473e2cc249'),
               ('up','language_model.model.layers.0.mlp.up_proj',17408,1,'5ee4fe36fedff37be5707fae8d414da5f4fbf11f23c74dd0c3a97d7384f497e7'))
LIMIT, DEADLINE, K = 4 << 30, 120, 5120
Q = dict(transpose=True,group_size=16,bits=4,mode='nvfp4')

def need(ok,msg):
    if not ok: raise ValueError(msg)
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def stat(p):
    s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def load(p):
    need(sha(p)==PINS[str(p)],'source changed');sp=importlib.util.spec_from_file_location('_probe_'+p.stem,p)
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def extract(folder,base,N,shard,header_sha,manifest,index):
    """Read full source SHA once, then copy ONLY the two selected payload ranges."""
    source=MODEL/f'model-{shard:05d}-of-00003.safetensors';before=stat(source)
    need(sha(source)==manifest['files'][source.name],'original full shard SHA')
    with source.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];need(n<1<<20,'header bound');raw=f.read(n)
    need(hashlib.sha256(raw).hexdigest()==header_sha,'header SHA');header=json.loads(raw)
    (folder/'original-header.json').write_bytes(raw);new={};parts=[];offset=0
    for short,suffix,dtype,width,size in (('w','weight','U32',K//8,4),('scales','scales','U8',K//16,1)):
        key=base+'.'+suffix;m=header[key];lo,hi=m['data_offsets']
        need(index[key]==source.name and m['dtype']==dtype and m['shape']==[N,width] and 0<=lo<hi<=before[2]-8-n and hi-lo==N*width*size,'selected tensor')
        new[short]=dict(dtype=dtype,shape=m['shape'],data_offsets=[offset,offset+hi-lo]);offset+=hi-lo
        parts.append(dict(name=key,start=8+n+lo,bytes=hi-lo,dtype=dtype,shape=m['shape']))
    encoded=json.dumps(new,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    target=folder/'selected-weights.safetensors'
    with source.open('rb') as src,target.open('xb') as dst:
        dst.write(struct.pack('<Q',len(encoded))+encoded)
        for part in parts:
            src.seek(part['start']);left=part['bytes'];h=hashlib.sha256()
            while left:
                b=src.read(min(left,8<<20));need(b,'short source slice');dst.write(b);h.update(b);left-=len(b)
            part['sha256']=h.hexdigest()
    with target.open('rb') as check:
        check.seek(8+len(encoded))
        for part in parts:
            left=part['bytes'];h=hashlib.sha256()
            while left:
                b=check.read(min(left,8<<20));need(b,'short copied slice');h.update(b);left-=len(b)
            need(h.hexdigest()==part['sha256'],'copied slice readback SHA')
        need(check.read(1)==b'','unexpected copy tail')
    need(stat(source)==before,'source changed during extraction')
    record=dict(source=str(source),source_stat=before,source_full_sha256=manifest['files'][source.name],full_sha_checked=True,header_sha256=header_sha,
                slices=parts,selected_file=str(target),selected_sha256=sha(target),selected_bytes=target.stat().st_size)
    write(folder/'weights.json',record);return target,record

def memory(mx):
    mx.synchronize();m=dict(active=mx.get_active_memory(),peak=mx.get_peak_memory(),cache=mx.get_cache_memory())
    need(max(m.values())<LIMIT,'4GiB allocator bound');return m

def qualify(mx,folder,x,routes,T,N):
    mx.save_safetensors(str(folder/'inputs.safetensors'),{'x':x});out={}
    for name,fn in routes.items():out[name]=fn();mx.eval(out[name]);memory(mx)
    need(all(a.shape==(T,N) and a.dtype==mx.bfloat16 for a in out.values()),'full output shape/dtype')
    path=folder/'outputs.safetensors';mx.save_safetensors(str(path),out)
    raw=path.read_bytes();n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n]);data={}
    for name in routes:
        h=header[name];lo,hi=h['data_offsets'];need(h['dtype']=='BF16' and h['shape']==[T,N] and hi-lo==2*T*N,'saved output layout');data[name]=raw[8+n+lo:8+n+hi]
    finite=all(all(v[0]&0x7f80!=0x7f80 for v in struct.iter_unpack('<H',b)) for b in data.values());comparisons={}
    for a,b in (('candidate','native_m8'),):
        first=next((i for i in range(0,len(data[a]),2) if data[a][i:i+2]!=data[b][i:i+2]),None) if data[a]!=data[b] else None
        comparisons[a+'__'+b]=dict(equal=data[a]==data[b],first_mismatch=None if first is None else dict(row=first//(2*N),column=first//2%N,left=data[a][first:first+2].hex(),right=data[b][first:first+2].hex()))
    result=dict(status='PASS' if finite and all(c['equal'] for c in comparisons.values()) else 'FAIL',finite=finite,comparisons=comparisons,
                original_dtype_sha256={k:hashlib.sha256(b).hexdigest() for k,b in data.items()},inputs_sha256=sha(folder/'inputs.safetensors'),outputs_sha256=sha(path),memory=memory(mx))
    write(folder/'numerical.json',result);need(result['status']=='PASS','raw BF16 qualification failed; no timing for this case')

def timing(mx,folder,routes):
    """Every sample constructs a new graph; both pre/post sync and staging included."""
    def once(fn):
        start=time.perf_counter_ns();mx.synchronize();value=fn();mx.eval(value);mx.synchronize();del value
        return time.perf_counter_ns()-start
    for fn in routes.values():
        for _ in range(5):once(fn)
    summary={}
    with (folder/'timings.jsonl').open('x') as log:
        for baseline in ('native_m8',):
            groups=[]
            for group in range(5):
                values={baseline:[], 'candidate':[]}
                for pair in range(20):
                    order=(baseline,'candidate') if (group*20+pair)%2==0 else ('candidate',baseline)
                    for position,name in enumerate(order):
                        ns=once(routes[name]);need(ns>0,'positive elapsed');values[name].append(ns)
                        log.write(json.dumps(dict(baseline=baseline,group=group,pair=pair,position=position,route=name,elapsed_ns=ns))+'\n');log.flush()
                med={k:statistics.median(v) for k,v in values.items()};groups.append(dict(median_ns=med,ratio_baseline_over_candidate=med[baseline]/med['candidate'],memory=memory(mx)))
            summary[baseline]=dict(groups=groups,median_group_ratio=statistics.median(g['ratio_baseline_over_candidate'] for g in groups))
    write(folder/'timing-summary.json',summary);return summary

def device(out):
    import mlx.core as mx
    identity=dict(version=importlib.metadata.version('mlx'),name=mx.device_info().get('device_name'),architecture=mx.metal.device_info().get('architecture'))
    write(out/'runtime.json',identity);need(identity==dict(version='0.32.3',name='Apple M5 Ultra',architecture='applegpu_g17d') and mx.default_device()==mx.gpu,'fixed runtime')
    mx.set_memory_limit(LIMIT);mx.set_cache_limit(256<<20)
    project=load(Path(__file__).with_name('candidate.py')).build(mx,enabled=True)
    need(project is not None,'candidate rejected');manifest=json.loads(MANIFEST.read_text());need(Path(manifest['model'])==MODEL,'fixed checkpoint')
    for name in ('config.json','model.safetensors.index.json'):need(sha(MODEL/name)==manifest['files'][name],'model metadata SHA')
    config=json.loads((MODEL/'config.json').read_text());need(config['quantization']==dict(group_size=16,bits=4,mode='nvfp4'),'quantization')
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];results=[];sources=[]
    for label,base,N,shard,h in PROJECTIONS:
        folder=out/label;folder.mkdir();selected,record=extract(folder,base,N,shard,h,manifest,index);sources.append(record)
        arrays=mx.load(str(selected));w,s=arrays['w'],arrays['scales'];mx.eval(w,s);memory(mx)
        for T in (8,):
            case=folder/f'T{T}';case.mkdir();u=mx.arange(T*K,dtype=mx.uint32)*1664525+1013904223
            x=((((u>>16)&1023).astype(mx.float32)-512).reshape(T,K)/256).astype(mx.bfloat16);mx.eval(x);del u
            routes=dict(native_m8=lambda:mx.quantized_matmul(x,w,s,**Q),candidate=lambda:project(x,w,s))
            qualify(mx,case,x,routes,T,N);results.append(dict(projection=label,T=T,K=K,N=N,timing=timing(mx,case,routes)))
            del routes,x;gc.collect();mx.clear_cache();memory(mx)
        del arrays,w,s;gc.collect();mx.clear_cache();memory(mx)
    need(all(stat(Path(r['source']))==r['source_stat'] for r in sources),'source stat changed')
    need(all(sha(Path(p))==h for p,h in PINS.items()),'source pin closure')
    return dict(status='PASS_LOCAL_PROJECTION_ONLY',cases=results,memory=memory(mx),full_model_qualified=False,engine_performance_qualified=False)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args(argv)
    if not a.run:
        need(a.out is None,'--out requires --run');print('DESCRIPTION ONLY: real head/up weights, synthetic BF16 x, T8 native-M8 vs candidate; local projection screening only.');return 0
    need(__name__=='__main__' and a.out is not None and a.out.is_absolute() and not a.out.exists() and a.out.parent.resolve()==a.out.parent,'standalone fresh absolute output')
    need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing timer');a.out.mkdir()
    write(a.out/'invocation.json',dict(argv=sys.argv,source_sha256=sha(Path(__file__)),pins=PINS,deadline_seconds=DEADLINE,limit_bytes=LIMIT,cache_limit_bytes=256<<20,
        input_scope='Deterministic uint32 LCG-derived BF16; not captured model hidden states.',timing='5 warmups per route; 5 groups x20 paired AB/BA samples per baseline; fresh operations+eval+sync+staging; 200 samples/case, 400 total.'))
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,DEADLINE)
    try:
        need(all(sha(Path(p))==h for p,h in PINS.items()),'initial source pins');result=device(a.out);write(a.out/'result.json',result);print(json.dumps(result));return 0
    except BaseException as e:
        write(a.out/'failure.json',dict(status='FAIL',type=type(e).__name__,message=str(e)));raise
if __name__=='__main__':raise SystemExit(main())
