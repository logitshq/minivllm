"""Offline raw-byte GLM expert stacking. Default describes; only --run copies payload.

No model/backend imports, mmap, arithmetic conversion or source writes. Metadata
planning reads headers and small sidecars. Payload copies use <=8 MiB buffers;
metadata dictionaries have a separate (larger) footprint. Output is a fresh path.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'.artifacts/m5-ultra-256gb/glm-flash-download-v1'
SOURCE=ROOT/'.artifacts/models/huggingface/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e'
SITE=ROOT/'.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages'
BUFFER=8<<20
SHARD_BYTES=4<<30
RESERVE=64<<30
DARWIN_F_NOCACHE=48  # Darwin SDK sys/fcntl.h; per-descriptor, never F_GLOBAL_NOCACHE.
DTYPES={'U32':4,'F16':2,'BF16':2,'F32':4}
RX=re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|down_proj|up_proj)\.(weight|scales|biases)$')
REVIEWED={
 'omlx/utils/model_loading.py':'0a298f42afcc6d9b7d8f3be7465bd90d9480c98992b46419e53a621bcd0551c2',
 'omlx/patches/mlx_vlm_mtp/glm5_next_vlm_runtime.py':'d2abbc013d2d5c6e3bb63124e3a0c51adaef64342e730884bbc36ebe4c53bb28',
 'mlx_lm/models/deepseek_v32.py':'c2140067fb3e35d5f195b051b2189d1170ba11d144a73aac810cc859f5d2fe4d',
 'omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/glm5_next.py':'dd5363e3ef445b0c28cf87969e89aef3d87212e223f003367c839408acc62077',
 'omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py':'1e449afa2f435d6b8c8a1ca4ffb697fd0476262572b536330a913f282022ca5a',
 'omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/config.py':'ac86282630beb6411e62b9938760d5644c831abfdb53779064a794ae348ba43f',
 'mlx_vlm/utils.py':'2045e18d1a16222d07b157251e4a5108b2b12bf740b885f9682a071c8dc59134'}

def need(ok,message):
    if not ok:raise ValueError(message)
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def unique(pairs):
    result={}
    for k,v in pairs:
        need(k not in result,'duplicate JSON key');result[k]=v
    return result
def loads(raw):return json.loads(raw,object_pairs_hook=unique)
def small(path,limit=32<<20):
    need(path.stat().st_size<=limit,'bounded metadata required: '+str(path));return path.read_bytes()
def identity(path):
    st=path.stat()
    return dict(resolved=str(path.resolve()),size=st.st_size,dev=st.st_dev,ino=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
def payload_io_policy():
    if sys.platform=='darwin':
        return dict(name='darwin_F_NOCACHE',command=DARWIN_F_NOCACHE,argument=1,required=True,
            scope='Per descriptor before payload IO; not a total-memory or zero-swap guarantee.')
    need(sys.platform.startswith('linux'),'unsupported payload IO platform')
    return dict(name='not_applicable_linux_cpu_fixture',required=False,
        scope='Portable tiny fixtures only; actual specialized CLI run requires Darwin.')

@contextmanager
def payload_open(path,mode):
    """Bounded buffered IO with per-FD Darwin cache bypass, failing closed."""
    need(mode in ('rb','xb'),'read or exclusive-write payload mode')
    policy=payload_io_policy()
    with path.open(mode) as f:
        if policy['required']:
            need(fcntl.fcntl(f.fileno(),DARWIN_F_NOCACHE,1)==0,'F_NOCACHE refused')
        yield f

def sha_file(path,buffer=BUFFER,*,payload=False):
    need(type(buffer) is int and 0<buffer<=BUFFER,'bounded buffer');h=hashlib.sha256()
    with (payload_open(path,'rb') if payload else path.open('rb')) as f:
        for data in iter(lambda:f.read(buffer),b''):h.update(data)
    return h.hexdigest()
def write_json(path,value):
    with path.open('xb') as f:f.write(encoded(value)+b'\n');f.flush();os.fsync(f.fileno())
def reference(path):
    return dict(path=str(path),sha256=digest(small(path)),identity=identity(path))
def verify_reference(ref):
    p=Path(ref['path']);need(identity(p)==ref['identity'] and digest(small(p))==ref['sha256'],'input metadata changed: '+str(p));return p

def header(path):
    before=identity(path)
    with path.open('rb') as f:
        prefix=f.read(8);need(len(prefix)==8,'short safetensors prefix')
        n=struct.unpack('<Q',prefix)[0];need(2<=n<=16<<20 and n+8<=before['size'],'bounded safetensors header')
        raw=f.read(n);need(len(raw)==n,'truncated header')
    obj=loads(raw);need(type(obj) is dict,'header object');metadata=obj.pop('__metadata__',{})
    need(type(metadata) is dict and all(type(k) is type(v) is str for k,v in metadata.items()),'string safetensors metadata')
    intervals=[]
    for key,row in obj.items():
        need(type(key) is str and type(row) is dict and set(row)=={'dtype','shape','data_offsets'},'tensor header fields')
        shape=row['shape'];offsets=row['data_offsets'];dtype=row['dtype']
        need(dtype in DTYPES and type(shape) is list and all(type(v) is int and v>0 for v in shape),'known nonempty tensor dtype/shape')
        need(type(offsets) is list and len(offsets)==2 and all(type(v) is int and v>=0 for v in offsets),'integer offsets')
        need(offsets[1]-offsets[0]==math.prod(shape)*DTYPES[dtype],'exact tensor byte length')
        intervals.append((offsets[0],offsets[1]))
    end=0
    for start,stop in sorted(intervals):need(start==end,'tensor gap/overlap');end=stop
    need(end+8+n==before['size'] and before==identity(path),'complete stable shard')
    return obj,dict(identity=before,header_sha256=digest(prefix+raw),header_bytes=8+n,payload_bytes=end,metadata=metadata)

def outputs_for(tensors,config):
    """Only leading-axis stack, exactly the original DSV32 layer/expert order."""
    tc=config['text_config'];layers=tc['num_hidden_layers'];experts=tc['n_routed_experts']
    need(type(layers) is int and layers>0 and type(experts) is int and experts>0,'model dimensions')
    q=config['quantization'];need(q==config['quantization_config'] and q.get('mode','affine')=='affine','same affine quantization maps')
    need(q['group_size']==64 and q['bits']==4,'original quantization defaults')
    groups={};out={}
    for key,row in tensors.items():
        match=RX.fullmatch(key)
        if match and int(match[1])<layers:
            layer,expert=int(match[1]),int(match[2]);target=f'model.language_model.layers.{layer}.mlp.switch_mlp.{match[3]}.{match[4]}'
            need(expert not in groups.setdefault(target,{}),'duplicate expert');groups[target][expert]=(key,row)
        else:
            need('.mlp.switch_mlp.' not in key,'source must be the original unstacked checkpoint')
            if '.mlp.experts.' in key:need(match is not None,'unrecognized expert key')
            out[key]=dict(key=key,dtype=row['dtype'],shape=row['shape'],parts=[dict(key=key,**row)])
    expected_layers={i for i,t in enumerate(tc['mlp_layer_types']) if i<layers and t=='sparse'}
    need(len(tc['mlp_layer_types'])==layers,'complete layer-type sequence')
    expected={f'model.language_model.layers.{l}.mlp.switch_mlp.{p}.{s}' for l in expected_layers for p in ('gate_proj','up_proj','down_proj') for s in ('weight','scales','biases')}
    need(set(groups)==expected,'complete exact backbone groups')
    for target,members in groups.items():
        need(set(members)==set(range(experts)),'missing/out-of-range expert')
        rows=[members[e] for e in range(experts)];first=rows[0][1]
        need(all(r['shape']==first['shape'] and r['dtype']==first['dtype'] for _,r in rows),'mixed expert shape/dtype')
        layer=int(target.split('.')[3]);projection=target.split('.')[-2];suffix=target.split('.')[-1]
        runtime_path=f'model.layers.{layer}.mlp.switch_mlp.{projection}'
        quant=q.get(runtime_path,dict(group_size=q['group_size'],bits=q['bits']))
        need(quant==dict(group_size=64,bits=5 if projection=='down_proj' else 4),'exact mixed expert quantization')
        dims=tc['moe_intermediate_size'] if projection=='down_proj' else tc['hidden_size']
        height=tc['hidden_size'] if projection=='down_proj' else tc['moe_intermediate_size']
        want=[height,dims*quant['bits']//32] if suffix=='weight' else [height,dims//64]
        need(dims%64==0 and first['shape']==want and first['dtype']==('U32' if suffix=='weight' else 'F16'),'packed affine expert shape/dtype')
        need(target not in out,'target collision');out[target]=dict(key=target,dtype=first['dtype'],shape=[experts]+first['shape'],parts=[dict(key=k,**r) for k,r in rows])
    return [out[k] for k in sorted(out)]

def inspect(source,download_ref,inventory_ref,runtime_pins):
    """Headers/sidecars only. Does not read or verify tensor payload bytes."""
    source=source.resolve();download=loads(small(verify_reference(download_ref)));inventory=loads(small(verify_reference(inventory_ref)))
    need(download['snapshot']==inventory['snapshot']==str(source),'exact receipt snapshot')
    need(download['status']=='PASS_PINNED_DOWNLOAD_AND_SHA_ONLY' and inventory['status']=='PASS_HEADER_INVENTORY_ONLY','original receipts')
    for path,h in runtime_pins.items():need(digest(small(Path(path)))==h,'reviewed loader source changed')
    names={x['name']:x for x in download['shards']};known={x['name']:x for x in inventory['shards']}
    need(len(names)==len(download['shards']) and set(names)==set(known),'unique matching receipt shards')
    need(all(Path(n).name==n and n.endswith('.safetensors') for n in names),'plain shard filenames')
    index_ref=reference(source/'model.safetensors.index.json');index=loads(small(source/'model.safetensors.index.json'))
    weights=index['weight_map'];need(type(weights) is dict and set(weights.values())==set(names),'complete original index')
    sidecars={};shards={};tensors={};metadatas=[]
    for p in sorted(source.iterdir()):
        if p.name in names or p.name=='model.safetensors.index.json':continue
        need(p.is_file() and p.suffix!='.safetensors' and not p.name.startswith('.') and p.name not in {'input-plan.json','invocation.json','repack-manifest.jsonl','complete.json','failure.json'},'plain unreserved sidecars only')
        sidecars[p.name]=reference(p)
    config=loads(small(source/'config.json'));need(config['model_type']=='glm5_next','GLM architecture')
    for name in sorted(names):
        rows,meta=header(source/name);expected=known[name]
        need(meta['header_sha256']==expected['header_sha256'] and meta['header_bytes']==expected['header_bytes'] and meta['identity']['size']==names[name]['bytes']==expected['file_bytes'],'retained header/stat identity')
        need(names[name]['sha256_verified'] is True and re.fullmatch('[0-9a-f]{64}',names[name]['sha256']),'prior payload SHA receipt')
        shards[name]=dict(**meta,expected_full_sha256=names[name]['sha256']);metadatas.append(meta['metadata'])
        for key,row in rows.items():
            need(key not in tensors and weights.get(key)==name,'index/header duplicate or disagreement')
            a,b=row['data_offsets'];tensors[key]=dict(dtype=row['dtype'],shape=row['shape'],shard=name,offset=meta['header_bytes']+a,bytes=b-a)
    need(set(tensors)==set(weights) and all(m==metadatas[0] for m in metadatas),'complete headers/shared format metadata')
    outputs=outputs_for(tensors,config);payload=sum(r['bytes'] for r in tensors.values())
    need(payload==sum(math.prod(x['shape'])*DTYPES[x['dtype']] for x in outputs),'payload conserved')
    core=dict(schema='GLM-expert-repack-plan-v1',source=str(source),candidate_sha256=sha_file(Path(__file__)),
        download=download_ref,inventory=inventory_ref,runtime_source_pins=runtime_pins,index=index_ref,sidecars=sidecars,shards=shards,
        safetensors_metadata=metadatas[0],payload_io_policy=payload_io_policy(),layout_sha256=digest(encoded(outputs)),
        summary=dict(input_tensors=len(tensors),output_tensors=len(outputs),stacked_tensors=sum(len(x['parts'])>1 for x in outputs),payload_bytes=payload,
            source_file_bytes=sum(v['identity']['size'] for v in shards.values()),layers=config['text_config']['num_hidden_layers'],experts=config['text_config']['n_routed_experts']),
        scope='Header/layout proof only; original full shard hashes are historical until explicit run rechecks them. No loader/GPU/quality/performance claim.')
    return core,outputs

def shard_groups(outputs,limit):
    need(type(limit) is int and limit>0,'positive shard size');groups=[];group=[];used=0
    for row in outputs:
        n=math.prod(row['shape'])*DTYPES[row['dtype']];need(n<=limit,'single tensor exceeds shard limit')
        if group and used+n>limit:groups.append(group);group=[];used=0
        group.append(row);used+=n
    if group:groups.append(group)
    return groups

def output_header(rows,metadata):
    out={'__metadata__':metadata};offset=0
    for row in rows:
        n=math.prod(row['shape'])*DTYPES[row['dtype']];out[row['key']]=dict(dtype=row['dtype'],shape=row['shape'],data_offsets=[offset,offset+n]);offset+=n
    raw=encoded(out);raw+=b' '*((-len(raw))%8)
    return struct.pack('<Q',len(raw))+raw,out

def execute(plan,out,*,buffer=BUFFER,shard_limit=SHARD_BYTES,reserve=RESERVE):
    need(type(buffer) is int and 0<buffer<=BUFFER and type(reserve) is int and reserve>=0,'bounded copying parameters')
    source=Path(plan['source']);out=Path(out).absolute()
    need(out.parent.resolve()==out.parent and out.parent.is_dir() and not out.exists() and not out.is_symlink(),'fresh canonical-parent output')
    need(not out.is_relative_to(source) and not source.is_relative_to(out),'source/output must not overlap')
    current,outputs=inspect(source,plan['download'],plan['inventory'],plan['runtime_source_pins']);need(current==plan,'exact unchanged plan/source/metadata')
    groups=shard_groups(outputs,shard_limit);headers=[output_header(g,plan['safetensors_metadata'])[0] for g in groups]
    required=plan['summary']['payload_bytes']+sum(map(len,headers))+sum(x['identity']['size'] for x in plan['sidecars'].values())+(64<<20)
    disk=os.statvfs(out.parent);available=disk.f_bavail*disk.f_frsize;need(available>=required+reserve,'insufficient disk including explicit reserve')
    # Full original payload SHA first, before any output directory is created.
    for name,entry in plan['shards'].items():
        p=source/name;need(identity(p)==entry['identity'] and sha_file(p,buffer,payload=True)==entry['expected_full_sha256'] and identity(p)==entry['identity'],'original full payload SHA/stat changed: '+name)
    out.mkdir();work=out;write_json(out/'input-plan.json',plan)
    write_json(out/'invocation.json',dict(candidate_sha256=plan['candidate_sha256'],buffer_bytes=buffer,shard_limit_bytes=shard_limit,reserve_bytes=reserve,disk_available_bytes=available,estimated_output_bytes=required,payload_io_policy=plan['payload_io_policy'],source_payload_SHA_rechecked=True))
    index={};files={};manifest_path=out/'repack-manifest.jsonl';written=0
    try:
        with manifest_path.open('x') as manifest:
            for number,(rows,prefix) in enumerate(zip(groups,headers),1):
                name=f'model-{number:05d}-of-{len(groups):05d}.safetensors';temp=work/(name+'.partial');filehash=hashlib.sha256();filehash.update(prefix);rows_meta=[]
                with payload_open(temp,'xb') as dst:
                    dst.write(prefix)
                    for row in rows:
                        group_hash=hashlib.sha256();parts=[];start=dst.tell()
                        for part in row['parts']:
                            p=source/part['shard'];need(identity(p)==plan['shards'][part['shard']]['identity'],'source changed during copy')
                            h=hashlib.sha256();remaining=part['bytes']
                            with payload_open(p,'rb') as src:
                                src.seek(part['offset'])
                                while remaining:
                                    raw=src.read(min(buffer,remaining));need(bool(raw),'truncated source payload');remaining-=len(raw)
                                    dst.write(raw);h.update(raw);group_hash.update(raw);filehash.update(raw)
                            parts.append(dict(key=part['key'],shard=part['shard'],offset=part['offset'],bytes=part['bytes'],sha256=h.hexdigest()))
                        count=dst.tell()-start;need(count==math.prod(row['shape'])*DTYPES[row['dtype']],'exact copied group bytes')
                        rows_meta.append(dict(key=row['key'],dtype=row['dtype'],shape=row['shape'],output_shard=name,output_offset=start,bytes=count,sha256=group_hash.hexdigest(),parts=parts))
                    dst.flush();os.fsync(dst.fileno())
                # Independent sequential output reread: header + every full group.
                with payload_open(temp,'rb') as check:
                    need(check.read(len(prefix))==prefix,'output header readback');verified=hashlib.sha256(prefix)
                    for row in rows_meta:
                        h=hashlib.sha256();remaining=row['bytes']
                        while remaining:
                            raw=check.read(min(buffer,remaining));need(bool(raw),'truncated output');remaining-=len(raw);h.update(raw);verified.update(raw)
                        need(h.hexdigest()==row['sha256'],'output group byte hash mismatch')
                    need(check.read(1)==b'' and verified.hexdigest()==filehash.hexdigest(),'output file byte hash mismatch')
                final=work/name;need(not final.exists(),'new final shard');temp.rename(final)
                files[name]=dict(sha256=filehash.hexdigest(),bytes=final.stat().st_size)
                for row in rows_meta:manifest.write(json.dumps(row,separators=(',',':'))+'\n');index[row['key']]=name;written+=1
                manifest.flush();os.fsync(manifest.fileno())
        for name,entry in plan['shards'].items():need(identity(source/name)==entry['identity'],'original source changed after copy')
        for source_path,h in plan['runtime_source_pins'].items():need(digest(small(Path(source_path)))==h,'loader source changed after copy')
        need(sha_file(Path(__file__))==plan['candidate_sha256'],'repacker source changed')
        need(written==plan['summary']['output_tensors'] and len(index)==written,'exact finalized output count')
        for name,ref in plan['sidecars'].items():
            p=verify_reference(ref);raw=small(p)
            with (out/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            need(digest(small(out/name))==ref['sha256'],'sidecar exact copy')
        # Index is the last checkpoint file published; no omitted tensor, dtype or config edit.
        index_value=dict(metadata=dict(total_size=plan['summary']['payload_bytes']),weight_map=index)
        write_json(out/'model.safetensors.index.json',index_value)
        result=dict(schema='GLM-expert-repack-result-v1',status='PASS_BYTES_ONLY',source_plan_sha256=digest(encoded(plan)+b'\n'),candidate_sha256=plan['candidate_sha256'],
            files=files,index_sha256=sha_file(out/'model.safetensors.index.json'),manifest_sha256=sha_file(manifest_path),summary=plan['summary'],
            payload_io_policy=plan['payload_io_policy'],source_payload_SHA_rechecked=True,output_groups_reread=True,original_unchanged_stat=True,runtime_load_executed=False,performance_qualified=False,
            scope='Original packed tensor bytes copied/stacked with no numeric conversion; complete source SHA before copy, group hashes and independent output reread. Actual official loader compatibility and physical-memory improvement still require a new device run.')
        write_json(out/'complete.json',result);return result
    except BaseException as error:
        write_json(out/'failure.json',dict(status='FAIL',type=type(error).__name__,message=str(error),completed_tensors=written,index_present=(out/'model.safetensors.index.json').exists(),scope='Partial output retained, never a completed checkpoint.'))
        raise

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group();mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256');p.add_argument('--out',type=Path);a=p.parse_args(argv)
    if not a.prepare and not a.run:
        need(not any((a.plan,a.plan_sha256,a.out)),'explicit mode required');print('DESCRIPTION ONLY: --prepare reads headers/metadata; --run copies the original ~190 GiB into a fresh checkpoint, no GPU.');return 0
    if a.prepare:
        need(a.out and not a.plan and not a.plan_sha256,'prepare requires only a fresh output JSON path')
        core,_=inspect(SOURCE,reference(BASE/'complete.json'),reference(BASE/'header-inventory.json'),{str(SITE/k):v for k,v in REVIEWED.items()})
        need(core['summary']['layers']==45 and core['summary']['experts']==288 and core['summary']['stacked_tensors']==378,'exact GLM target')
        write_json(a.out,core);print(json.dumps(dict(status='HEADER_PLAN_ONLY',path=str(a.out),sha256=sha_file(a.out),summary=core['summary'])));return 0
    need(a.plan and a.plan_sha256 and a.out,'run requires explicit plan/hash/fresh output')
    need(sys.platform=='darwin','actual specialized checkpoint run requires Darwin F_NOCACHE')
    need(digest(small(a.plan))==a.plan_sha256,'explicit plan SHA');plan=loads(small(a.plan))
    need(plan['source']==str(SOURCE) and plan['summary']['layers']==45 and plan['summary']['experts']==288 and plan['summary']['stacked_tensors']==378,'actual specialized source only')
    result=execute(plan,a.out);print(json.dumps({k:result[k] for k in ('status','summary','source_payload_SHA_rechecked','output_groups_reread','runtime_load_executed','performance_qualified')}));return 0

if __name__=='__main__':raise SystemExit(main())
