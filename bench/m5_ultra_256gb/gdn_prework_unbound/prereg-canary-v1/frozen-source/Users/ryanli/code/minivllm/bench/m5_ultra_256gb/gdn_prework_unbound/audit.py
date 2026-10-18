"""Independent stdlib-only raw canary audit. Never imports MLX/numpy/producer.

CLI --child CLOSED_DIRECTORY --plan-sha256 SHA --out NEW_JSON. Re-read all60
saved tensor files, 54 full output/state array pairs, 24 continued-state pairs
and six full-vocabulary argmax results. Noncaptured reference rollout tokens,
parameter invariance and evaluated-dispatch evidence remain producer receipts;
this is not independent execution of the model, full-model or performance proof.
Owner/resource/restoration auditing is deliberately separate.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

HERE=Path(__file__).resolve().parent
QUALIFIER_SHA='f3088a98d2aca6b4f267e7d4c258efbbf9f83e1beccb6745c88ae42343f88e89'
CANDIDATE_SHA='1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260'
OLD_PLAN_SHA='5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
SUPPORT_SHA='26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'
DTYPES={'BF16':(2,'mlx.core.bfloat16'),'F32':(4,'mlx.core.float32'),'I32':(4,'mlx.core.int32')}
SCOPE='Local same-reference-activation canary only; offline full saved bytes, device comparison/parameter receipts. No full-model rollout, HTTP, resource, restoration or performance qualification.'


class Reject(ValueError): pass
class Incomplete(ValueError): pass
def need(ok,message):
    if not ok: raise Reject(message)
def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def unique(pairs):
    out={}
    for key,value in pairs:
        need(key not in out,'duplicate JSON key: '+key);out[key]=value
    return out
def read(path):
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique)
def same(a,b,message):
    need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)
def pinned(ref):
    need(type(ref) is dict and set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute(),'exact absolute pinned ref')
    need(sha(ref['path'])==ref['sha256'],'pinned bytes changed: '+ref['path']);return read(ref['path'])
def local(child,name):
    p=Path(name);need(not p.is_absolute() and '..' not in p.parts,'local artifact path')
    path=child/p;need(path.resolve().is_relative_to(child) and not path.is_symlink(),'local regular artifact')
    return path


def words(payload,encoding):
    result=array('H' if encoding=='BF16' else 'I');result.frombytes(payload)
    if sys.byteorder!='little':result.byteswap()
    return result


def finite(payload,encoding):
    if encoding=='I32':return True
    mask=0x7f80 if encoding=='BF16' else 0x7f800000
    return all((x&mask)!=mask for x in words(payload,encoding))


def tensors(raw):
    """Small safetensors parser with contiguous exact ranges and raw finiteness."""
    need(8<=len(raw)<=16*1024**2,'bounded canary tensor file')
    n,=struct.unpack_from('<Q',raw);need(0<n<=1024**2 and 8+n<=len(raw),'tensor header bound')
    header=json.loads(raw[8:8+n],object_pairs_hook=unique)
    need(header.pop('__metadata__',None) in (None,{}),'unexpected tensor metadata')
    result={};cursor=0;data=raw[8+n:]
    for name,item in sorted(header.items(),key=lambda kv:kv[1]['data_offsets'][0]):
        need(set(item)=={'dtype','shape','data_offsets'} and item['dtype'] in DTYPES,'original tensor schema/dtype')
        shape=item['shape'];lo,hi=item['data_offsets'];width=DTYPES[item['dtype']][0]
        need(type(shape) is list and all(type(x) is int and x>=0 for x in shape)
             and type(lo) is type(hi) is int and lo==cursor and hi-lo==math.prod(shape)*width and hi<=len(data),
             'tensor shape/contiguous payload range')
        payload=data[lo:hi];need(finite(payload,item['dtype']),'nonfinite original tensor: '+name)
        result[name]=dict(encoding=item['dtype'],shape=shape,raw=payload);cursor=hi
    need(cursor==len(data) and bool(result),'trailing/empty tensor payload');return result


def raw_file(child,desc,expected,seen):
    need(set(desc)=={'path','sha256','finite','tensors','metadata'} and desc['finite'] is True,'saved tensor receipt')
    path=local(child,desc['path']);raw=path.read_bytes();need(sha_bytes(raw)==desc['sha256'],'raw file SHA: '+desc['path'])
    got=tensors(raw);need(set(got)==set(expected)==set(desc['tensors']),'complete raw tensor keys')
    metadata={'array':'root'} if set(expected)=={'root'} else {name.split('/',1)[1]:{'array':name} for name in expected}
    same(desc['metadata'],metadata,'original snapshot metadata/array binding')
    for name,(shape,encodings) in expected.items():
        value=got[name];need(value['shape']==shape and value['encoding'] in encodings,'actual original shape/dtype: '+name)
        declared=desc['tensors'][name]
        same(declared,dict(shape=shape,dtype=DTYPES[value['encoding']][1],encoding=value['encoding'],
            bytes=len(value['raw']),sha256=sha_bytes(value['raw']),finite=True),'tensor payload/hash/finite receipt')
    same(read(path.with_suffix('.json')),desc,'raw adjacent receipt exact')
    need(desc['path'] not in seen,'duplicate raw artifact');seen.add(desc['path'])
    return got


def pair(left,right,label):
    need(left['shape']==right['shape'] and left['encoding']==right['encoding'],'shape/dtype pair: '+label)
    if left['raw']!=right['raw']:
        offset=next(i for i,(a,b) in enumerate(zip(left['raw'],right['raw'])) if a!=b)
        raise Reject(f'{label}: first differing byte {offset}; left={left["raw"][offset]:02x} right={right["raw"][offset]:02x}')


def argmax(tensor):
    need(len(tensor['shape'])==2 and tensor['shape'][0]==1 and tensor['encoding'] in ('BF16','F32'),'full logits layout')
    values=words(tensor['raw'],tensor['encoding']);best=None;index=None
    for i,bits in enumerate(values):
        if tensor['encoding']=='BF16':bits<<=16
        value=struct.unpack('<f',struct.pack('<I',bits))[0]
        if best is None or value>best:best,index=value,i
    need(index is not None,'nonempty complete vocabulary');return index


def dispatch(row,expected,before):
    for key,value in expected.items():same(row.get(key),value,'dispatch point binding: '+key)
    same(row['before'],dict(candidate=before,fallback=0),'continuous actual counter before')
    same(row['after'],dict(candidate=before+1,fallback=0),'one candidate/no fallback after')


def parameter_layout():
    result={}
    for name,k,n in (('in_proj_qkv',5120,10240),('in_proj_z',5120,6144),('in_proj_b',5120,48),
                     ('in_proj_a',5120,48),('out_proj',6144,5120)):
        result[name+'.weight']=([n,k//8],'mlx.core.uint32',4)
        result[name+'.scales']=([n,k//16],'mlx.core.uint8',1)
    result.update({'conv1d_weight':([10240,4],'mlx.core.bfloat16',2),'A_log':([48],'mlx.core.float32',4),
                   'dt_bias':([48],'mlx.core.float32',4),'norm.weight':([128],'mlx.core.bfloat16',2)})
    return result


def expected_contract():
    # Independent fixed receipt, not imported/executed producer code.
    return dict(schema='GDN-prework-real-layer-contract-v1',fixture_selection='shortest/longest original D512; ties by fixture_id',
        starts=[1,129],steps_per_chain=3,gdn_ordinals=[0,24,47],expected_layer_calls=18,
        configuration=dict(max_len=2048,max_seqs=1,max_batched_tokens=16384,cache_limit_gib=2,
            compile_decode=False,fuse_decode=False,gates_compile=True,norm_reuse=False),
        inputs='unmodified actual original-model activation; ordinary eager greedy rollout',
        state='candidate initialized once per layer/fixture, then its own prior output state',
        normal_eos=True,early_eos='INCOMPLETE; no ignore_eos or synthetic continuation',
        exact_arrays=['output:BF16[1,5120]','conv:BF16[1,3,10240]','S:FP32[1,48,128,128]'],
        full_model_qualified=False,http_qualified=False,performance_qualified=False,
        excluded=['candidate full-model rollout','ahead/scheduling','B>1','parameter replacement',
                  'prefill optimization','E024/E028 composition','random/special/strided synthetic coverage'])


def audit(child,plan_sha256):
    child=Path(child).resolve();result=dict(schema='GDN-prework-real-layer-independent-v1',status='INCOMPLETE',
        canary_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,scope=SCOPE,
        audit_source_sha256=sha(__file__),child_directory=str(child),plan_sha256=plan_sha256,raw_files=0,
        full_output_state_pairs=0,causal_state_pairs=0,full_vocabulary_greedy_checks=0,device_comparison_arrays=0)
    try:
        if not (child/'complete.json').exists():raise Incomplete('missing terminal result; no raw files opened')
        complete=read(child/'complete.json');need(complete['schema']=='GDN-prework-real-layer-result-v1','terminal schema')
        if complete['status']=='INCOMPLETE':raise Incomplete('producer incomplete; no qualification')
        need(complete['status']=='PASS' and complete['canary_qualified'] is True,'producer did not complete exact gate')
        same(complete['contract'],expected_contract(),'fixed experiment configuration/coverage/scope')
        for key in ('full_model_qualified','http_qualified','performance_qualified'):need(complete[key] is False,'producer scope')
        need(sha(child/'plan.json')==plan_sha256==complete['plan_sha256'],'exact saved plan')
        plan=read(child/'plan.json')
        need(set(plan)=={'schema','qualifier_sha256','candidate','e024_plan','deadline_s'}
             and plan['schema']=='GDN-prework-real-layer-plan-v1' and plan['qualifier_sha256']==QUALIFIER_SHA
             and type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s']) and 0<plan['deadline_s']<=1800,'fixed child plan')
        need(type(complete['elapsed_wall_s']) in (int,float) and math.isfinite(complete['elapsed_wall_s'])
             and 0<=complete['elapsed_wall_s']<=plan['deadline_s'],'completed within child deadline')
        same(plan['candidate'],dict(path=str(HERE/'candidate.py'),sha256=CANDIDATE_SHA),'candidate identity')
        need(plan['e024_plan']['sha256']==OLD_PLAN_SHA,'original E024 plan SHA')
        original=pinned(plan['e024_plan']);need(sha(child/'inherited-e024-plan.json')==OLD_PLAN_SHA,'saved inherited E024 bytes')
        numeric=pinned(original['e023_plan']);same(read(child/'inherited-e023-plan.json'),numeric,'saved original numerical plan')
        sources={str(HERE/'qualify.py'):QUALIFIER_SHA,str(HERE/'candidate.py'):CANDIDATE_SHA,
            str(HERE.parent/'long_decode_isolation/qualify.py'):'9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e',
            str(HERE.parent/'long_decode_graph/qualify.py'):'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
            str(HERE.parent/'long_decode_graph/model-source-pins.json'):'7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
            original['e023_qualifier']['path']:SUPPORT_SHA}
        same(complete['sources'],sources,'exact six child sources')
        for name,digest in sources.items():need(sha(name)==digest,'current frozen source changed: '+name)
        core=pinned(numeric['source_manifest']);weights=pinned(numeric['model_manifest']);fixtures=pinned(numeric['fixtures'])['fixtures']
        same(core['files'],read(HERE.parent/'long_decode_graph/model-source-pins.json')['files'],'full core source manifest')
        root=Path(numeric['package_root'])/'tutorial_code/miniengine'
        for name,digest in core['files'].items():need(sha(root/name)==digest,'actual frozen core source')
        config_path=Path(numeric['model'])/'config.json';need(sha(config_path)==weights['files']['config.json'],'model config bytes')
        config=read(config_path);tc=config.get('text_config',config);vocab=tc['vocab_size']
        layout=tc.get('layer_types') or ['full_attention' if (i+1)%tc['full_attention_interval']==0 else 'linear_attention' for i in range(tc['num_hidden_layers'])]
        gdn=[i for i,x in enumerate(layout) if x=='linear_attention'];need(len(layout)==64 and len(gdn)==48,'real model layout')
        indices=[gdn[i] for i in (0,24,47)];same(complete['layer_indices'],indices,'actual selected GDN indices')
        ordered=sorted(fixtures,key=lambda f:(len(f['prompt_ids']),f['fixture_id']))
        need(len(fixtures)==8 and len({f['fixture_id'] for f in fixtures})==8,'eight original fixtures')
        chosen=[(ordered[0],1),(ordered[-1],129)]
        for f,_ in chosen:
            need(f['normal_eos'] is True and f['max_output_tokens']==512 and 886<=len(f['prompt_ids'])<=978,'real D512 scope')
            same(read(child/(f['fixture_id']+'-fixture.json')),f,'full original fixture bytes/IDs')
        for field,value in (('reference_chains',2),('candidate_chains',6),('candidate_calls',18),('candidate_compared_calls',18),
                            ('device_run',True),('instance_hooks_restored',True),('source_model_pins_rechecked',True)):
            same(complete[field],value,'complete coverage: '+field)
        same(complete['device'],dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3'),'actual device receipt')
        parameter_sets=[]
        for name in ('parameters-before','parameters-after-reference','parameters-after-candidate'):
            ref=complete[name];need(ref['path']==name+'.json' and sha(child/ref['path'])==ref['sha256'],'parameter receipt SHA')
            rows=read(child/ref['path']);need(set(rows)==set(map(str,indices)),'complete selected layer parameter sets')
            cleaned={}
            for layer,params in rows.items():
                need(set(params)==set(parameter_layout()),'all14 selected-layer parameter tensors')
                cleaned[layer]={}
                for key,row in params.items():
                    shape,dtype,width=parameter_layout()[key]
                    need(row['finite'] is True and type(row['bytes']) is int and row['bytes']>0
                         and row['shape']==shape and row['dtype']==dtype and row['bytes']==math.prod(shape)*width
                         and type(row['sha256']) is str and len(row['sha256'])==64
                         and all(c in '0123456789abcdef' for c in row['sha256']),'parameter byte/finite receipt')
                    cleaned[layer][key]={k:v for k,v in row.items() if k!='handle_id'}
            parameter_sets.append(cleaned)
        same(parameter_sets[0],parameter_sets[1],'reference parameter values unchanged')
        same(parameter_sets[0],parameter_sets[2],'candidate parameter values unchanged')
        events=[];previous=-1
        with (child/'events.jsonl').open() as stream:
            for line in stream:
                need(line.endswith('\n'),'truncated event row');row=json.loads(line,object_pairs_hook=unique)
                t=row['elapsed_wall_s'];need(type(t) in (int,float) and math.isfinite(t) and previous<=t<=complete['elapsed_wall_s'],'ordered event time')
                previous=t;events.append(row);need(len(events)<=256,'bounded fixed event ledger')
        kinds={'memory','reference-token','reference-capture','candidate-dispatch','comparison'}
        need(all(row['event'] in kinds for row in events),'unknown event')
        captured=[r for r in events if r['event']=='reference-capture'];calls=[r for r in events if r['event']=='candidate-dispatch']
        comparisons=[r for r in events if r['event']=='comparison'];need(len(captured)==6 and len(calls)==18 and len(comparisons)==54,'exact event coverage')
        chronology=[('memory',None,None,None,'before-load')]
        for fixture,start in chosen:
            fid=fixture['fixture_id']
            for d in range(1,start+3):
                if d>=start:chronology.append(('reference-capture',fid,None,d,None))
                chronology.append(('reference-token',fid,None,d,None))
            chronology.append(('memory',None,None,None,'reference-chain-released'))
        for fixture,start in chosen:
            for index in indices:
                for d in range(start,start+3):
                    for event,kind in (('comparison','input-before'),('candidate-dispatch',None),('comparison','output'),('comparison','input-unchanged')):
                        chronology.append((event,fixture['fixture_id'],index,d,kind))
        chronology.append(('memory',None,None,None,'after-release'))
        actual_chronology=[(r['event'],r.get('fixture_id'),r.get('layer'),r.get('decode_step'),r.get('phase') if r['event']=='memory' else r.get('kind')) for r in events]
        same(actual_chronology,chronology,'actual event chronology/no extra unknown fixture/dispatch')
        records=read(child/'captures.json');need(len(records)==2,'two captured fixtures')
        seen=set();counters={i:0 for i in indices};capture_index=call_index=comparison_index=0
        base_shapes={'root/conv':([1,3,10240],{'BF16'}),'root/S':([1,48,128,128],{'F32'})}
        for case,(fixture,start) in zip(records,chosen):
            fid=fixture['fixture_id'];prompt=len(fixture['prompt_ids']);raw=case['raw_ids']
            need(case['fixture_id']==fid and case['start']==start and case['prompt_tokens']==prompt
                 and len(case['steps'])==3 and len(raw)==start+3 and all(type(x) is int and 0<=x<vocab for x in raw),'actual capture case/IDs')
            eos=set(fixture['eos_token_ids']);need(not any(x in eos for x in raw[:-1]),'continued through normal EOS')
            tokens=[r for r in events if r['event']=='reference-token' and r['fixture_id']==fid]
            need(len(tokens)==start+2,'all actual reference token steps')
            for n,row in enumerate(tokens,1):
                same([row['decode_step'],row['input_token'],row['output_token'],row['captured']],
                     [n,raw[n-1],raw[n],n>=start],'causal reference token chain')
            inputs={};outputs={}
            for ordinal,step in enumerate(case['steps']):
                d=start+ordinal;event=captured[capture_index];capture_index+=1
                need(event['fixture_id']==fid and event['decode_step']==d and event['used']==prompt+d,'reference capture point')
                need(set(step)==set(map(str,indices)) and set(event['layers'])==set(step),'all selected reference layers')
                need(event['reference_full_logits']['path']==f'captures/{fid}/step-{d}/reference-full-logits.safetensors','actual full-logit artifact path')
                full=raw_file(child,event['reference_full_logits'],{'root':([1,vocab],{'BF16','F32'})},seen)
                need(argmax(full['root'])==raw[d],'actual full-vocabulary greedy output');result['full_vocabulary_greedy_checks']+=1
                for index in indices:
                    record=step[str(index)];same({k:v for k,v in record.items() if k!='candidate_output'},event['layers'][str(index)],'capture event/raw binding')
                    same([record['layer'],record['decode_step'],record['position'],record['used'],record['input_token']],
                         [index,d,prompt+d-1,prompt+d,raw[d-1]],'actual layer input position/token')
                    for key,path in (('reference_input',f'captures/{fid}/step-{d}/layer-{index}/reference-input.safetensors'),
                                     ('reference_output',f'captures/{fid}/step-{d}/layer-{index}/reference-output.safetensors'),
                                     ('candidate_output',f'candidate/{fid}/layer-{index}/step-{d}/output.safetensors')):
                        need(record[key]['path']==path,'fixed original artifact/point path')
                    inp=raw_file(child,record['reference_input'],{**base_shapes,'root/x':([1,5120],{'BF16'})},seen)
                    ref=raw_file(child,record['reference_output'],{**base_shapes,'root/output':([1,5120],{'BF16'})},seen)
                    got=raw_file(child,record['candidate_output'],{**base_shapes,'root/output':([1,5120],{'BF16'})},seen)
                    for name in ref:pair(ref[name],got[name],f'{fid}/{index}/{d}/{name}');result['full_output_state_pairs']+=1
                    inputs[index,ordinal]=inp;outputs[index,ordinal]=got
            # Match actual replay execution order: layer outer, three-step inner.
            for index in indices:
                for ordinal in range(3):
                    point=dict(fixture_id=fid,layer=index,decode_step=start+ordinal,used=prompt+start+ordinal)
                    dispatch(calls[call_index],point,counters[index]);call_index+=1;counters[index]+=1
                    for kind in ('input-before','output','input-unchanged'):
                        row=comparisons[comparison_index];comparison_index+=1
                        for key,value in {**point,'kind':kind}.items():same(row[key],value,'comparison event point')
                        expected=outputs[index,ordinal] if kind=='output' else inputs[index,ordinal]
                        need(len(row['checks'])==3 and {r['name'] for r in row['checks']}==set(expected),'full three-array device checks')
                        label=f'{fid}/layer-{index}/step-{start+ordinal}/'+('complete-output' if kind=='output' else kind)
                        for check in row['checks']:
                            value=expected[check['name']]
                            same(check,dict(check=label,name=check['name'],shape=value['shape'],dtype=DTYPES[value['encoding']][1],
                                 reference_finite=True,candidate_finite=True,bit_equal=True),'original exact comparator receipt')
                            result['device_comparison_arrays']+=1
                    if ordinal:
                        for name in base_shapes:
                            pair(outputs[index,ordinal-1][name],inputs[index,ordinal][name],f'{fid}/{index}/continued-state/{name}')
                            result['causal_state_pairs']+=1
        same(complete['dispatch_counters'],{str(i):dict(candidate=6,fallback=0) for i in indices},'terminal evaluated dispatch counts')
        actual={str(p.relative_to(child)) for p in child.rglob('*.safetensors')}
        need(actual==seen and len(seen)==60,'exact successful raw artifact coverage')
        need(result['full_output_state_pairs']==54 and result['causal_state_pairs']==24
             and result['full_vocabulary_greedy_checks']==6 and result['device_comparison_arrays']==162,'independent fixed comparison counts')
        result.update(status='PASS',canary_qualified=True,raw_files=len(seen),complete_sha256=sha(child/'complete.json'),
            ledger_sha256=sha(child/'events.jsonl'),captures_sha256=sha(child/'captures.json'),source_pins=sources,
            model_weights_scope='Original fixed manifest binding + producer pre/post verification receipt; auditor does not rehash model payloads.',
            parameter_scope='All14 selected-layer original-byte hashes/shape/dtype/finite before/after; parameter payloads are not stored or recomputed offline.')
    except Incomplete as exc:result['first_failure']=str(exc)
    except FileNotFoundError as exc:result['first_failure']='missing required evidence: '+str(exc)
    except BaseException as exc:result.update(status='FAIL',first_failure=type(exc).__name__+': '+str(exc))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    result=audit(a.child,a.plan_sha256)
    with a.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],canary_qualified=result['canary_qualified'],raw_files=result['raw_files'])))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
