"""Independent original-byte B2/B8 local canary audit; stdlib only.

No producer/device import or model execution. Read all140 saved tensor files,
108 complete output/state pairs,48 continued-state pairs,80 full-vocab row
argmax checks. Intermediate reference logits and parameter payloads are not
stored: their source-pinned producer receipts remain explicitly narrower proof.
Owner/resources/restoration and all full-model/HTTP/performance gates are separate.
"""
import argparse
from array import array
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import struct
from types import FunctionType

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'gdn_prework_unbound/audit.py'
BASE_SHA='a3e97dc368da47735d7c8e9fd2bcb40d7decaccb9c5926084d53cffb10495d3d'
QUALIFIER_SHA='8b0f0fc0060d918f1eb3a169e702985642f10d6b03760546941308846256425c'
CANDIDATE_SHA='acd2a23802296b9e420108203cc215c3170cd1eccd9a859e1a787432f98a5353'
B1_QUALIFIER_SHA='f3088a98d2aca6b4f267e7d4c258efbbf9f83e1beccb6745c88ae42343f88e89'
if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('fixed raw auditor changed')
_spec=importlib.util.spec_from_file_location('_gdn_batched_frozen_raw_audit',BASE)
_base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_base)
for _name in ('need','read','sha','sha_bytes','same','pinned','local','words','finite','pair','dispatch','parameter_layout',
              'Reject','Incomplete','OLD_PLAN_SHA','SUPPORT_SHA','DTYPES','unique'):
    globals()[_name]=getattr(_base,_name)
# The only raw parser delta is its per-file bound: B8 S/conv/output needs <32MiB.
_tensor_source=inspect.getsource(_base.tensors)
need(_tensor_source.count('16*1024**2')==1,'fixed parser bound anchor')
_tensor_source=_tensor_source.replace('16*1024**2','32*1024**2',1)
_parser_namespace=dict(vars(_base));exec(compile(ast.parse(_tensor_source),__file__,'exec'),_parser_namespace)
tensors=_parser_namespace['tensors']
raw_file=FunctionType(_base.raw_file.__code__,dict(vars(_base),tensors=tensors),_base.raw_file.__name__)
SCOPE='Only local B2/B8 reference-activation layers; all saved original bytes independently read. No candidate full-model rollout, HTTP, resource, restoration or performance qualification.'


def argmax_rows(tensor):
    need(len(tensor['shape'])==2 and tensor['shape'][0] in (1,2,8) and tensor['shape'][1]>0
         and tensor['encoding']=='BF16','complete batched BF16 vocabulary layout')
    B,V=tensor['shape'];values=words(tensor['raw'],'BF16');result=[]
    need(len(values)==B*V,'complete raw vocabulary bytes')
    for row in range(B):
        best=None;index=None
        for i in range(V):
            value=struct.unpack('<f',struct.pack('<I',values[row*V+i]<<16))[0]
            if best is None or value>best:best,index=value,i
        result.append(index)
    return result


def token_chain(case,cid,B,start,fixtures,tokens,vocab):
    ids=[f['fixture_id'] for f in fixtures];lengths=[len(f['prompt_ids']) for f in fixtures];raw=case['raw_ids']
    same([case['case_id'],case['batch_size'],case['row_ids'],case['start'],case['prompt_tokens']],
         [cid,B,ids,start,lengths],'actual batch IDs/order/history')
    need(len(case['steps'])==3 and len(raw)==B and all(len(row)==start+3 and all(type(t) is int and 0<=t<vocab for t in row) for row in raw),
         'complete original batched token histories')
    for row,f in zip(raw,fixtures):need(not any(t in set(f['eos_token_ids']) for t in row[:-1]),'continued past normal EOS')
    need(len(tokens)==start+2,'every native batched decode token event')
    for d,row in enumerate(tokens,1):
        same([row['batch_size'],row['row_ids'],row['decode_step'],row['input_tokens'],row['output_tokens'],row['captured']],
             [B,ids,d,[r[d-1] for r in raw],[r[d] for r in raw],d>=start],'per-row actual token causality')
    return raw


def expected_contract():
    return dict(schema='GDN-batched-prework-real-layer-contract-v1',batches=[2,8],
        cases=['b2-early','b2-late','b8-early','b8-late'],starts=[1,129],steps_per_chain=3,
        fixture_selection='sort original8 by (prompt length,fixture_id); B2 shortest2 early/longest2 late; B8 all8 both',
        gdn_ordinals=[0,24,47],expected_layer_calls=36,reference_chains=4,candidate_chains=12,
        configuration=dict(max_len=2048,max_seqs=8,max_batched_tokens=16384,cache_limit_gib=2,
            compile_decode=False,fuse_decode=False,gates_compile=True,norm_reuse=False),
        inputs='native batched full-model greedy rollout; each row consumes its own actual prior ID',
        state='independent initial candidate state, then own prior state for all three calls',
        normal_eos=True,early_eos='any row EOS before final window output is INCOMPLETE; no forced tokens or retirement',
        exact_arrays=['output:BF16[B,5120]','conv:BF16[B,3,10240]','S:FP32[B,48,128,128]'],
        saved_reference_logits='20 prefill outputs plus12 captured full-vocab batched outputs; intervening tokens have producer receipts only',
        full_model_qualified=False,http_qualified=False,performance_qualified=False,
        excluded=['candidate full-model rollout','ahead/scheduling','B3..7','parameter replacement',
            'row permutation/removal/refill','prefill optimization','E024/E028 composition','synthetic/strided coverage'])


def audit(child,plan_sha256):
    child=Path(child).resolve();result=dict(schema='GDN-batched-prework-real-layer-independent-v1',status='INCOMPLETE',
        canary_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,scope=SCOPE,
        audit_source_sha256=sha(__file__),child_directory=str(child),plan_sha256=plan_sha256,raw_files=0,
        full_output_state_pairs=0,causal_state_pairs=0,full_vocabulary_greedy_checks=0,device_comparison_arrays=0)
    try:
        if not (child/'complete.json').exists():raise Incomplete('missing terminal result; no raw files opened')
        complete=read(child/'complete.json');need(complete['schema']=='GDN-batched-prework-real-layer-result-v1','terminal schema')
        if complete['status']=='INCOMPLETE':raise Incomplete('producer incomplete; no qualification')
        need(complete['status']=='PASS' and complete['canary_qualified'] is True,'producer did not complete exact gate')
        same(complete['contract'],expected_contract(),'fixed experiment configuration/coverage/scope')
        for key in ('full_model_qualified','http_qualified','performance_qualified'):need(complete[key] is False,'producer scope')
        need(sha(child/'plan.json')==plan_sha256==complete['plan_sha256'],'exact saved plan')
        plan=read(child/'plan.json')
        need(set(plan)=={'schema','qualifier_sha256','candidate','e024_plan','deadline_s'}
             and plan['schema']=='GDN-batched-prework-real-layer-plan-v1' and plan['qualifier_sha256']==QUALIFIER_SHA
             and type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s']) and 0<plan['deadline_s']<=1800,'fixed child plan')
        need(type(complete['elapsed_wall_s']) in (int,float) and math.isfinite(complete['elapsed_wall_s'])
             and 0<=complete['elapsed_wall_s']<=plan['deadline_s'],'completed within child deadline')
        same(plan['candidate'],dict(path=str(HERE/'candidate.py'),sha256=CANDIDATE_SHA),'candidate identity')
        need(plan['e024_plan']['sha256']==OLD_PLAN_SHA,'original E024 plan SHA')
        original=pinned(plan['e024_plan']);need(sha(child/'inherited-e024-plan.json')==OLD_PLAN_SHA,'saved inherited E024 bytes')
        numeric=pinned(original['e023_plan']);same(read(child/'inherited-e023-plan.json'),numeric,'saved original numerical plan')
        sources={str(HERE/'qualify.py'):QUALIFIER_SHA,str(HERE/'candidate.py'):CANDIDATE_SHA,
            str(HERE.parent/'gdn_prework_unbound/qualify.py'):B1_QUALIFIER_SHA,
            str(HERE.parent/'long_decode_isolation/qualify.py'):'9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e',
            str(HERE.parent/'long_decode_graph/qualify.py'):'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
            str(HERE.parent/'long_decode_graph/model-source-pins.json'):'7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
            original['e023_qualifier']['path']:SUPPORT_SHA}
        same(complete['sources'],sources,'exact seven child sources')
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
        need(len(fixtures)==8 and len({f['fixture_id'] for f in fixtures})==8,'eight original distinct fixtures')
        for f in ordered:
            need(f['normal_eos'] is True and type(f['max_output_tokens']) is int and f['max_output_tokens']==512
                 and 886<=len(f['prompt_ids'])<=978 and all(type(t) is int and 0<=t<vocab for t in f['prompt_ids']), 'original D512 fixture scope')
        chosen=[(f'b{B}-{age}',B,start,(ordered[:2] if age=='early' else ordered[-2:]) if B==2 else ordered)
            for B in (2,8) for age,start in (('early',1),('late',129))]
        for cid,B,start,fs in chosen:same(read(child/(cid+'-fixtures.json')),fs,'complete original row fixtures')
        for field,value in (('reference_chains',4),('candidate_chains',12),('candidate_calls',36),('candidate_compared_calls',36),
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
                previous=t;events.append(row);need(len(events)<=512,'bounded exact event ledger')
        chronology=[('memory',None,None,None,'before-load')]
        for cid,B,start,fs in chosen:
            chronology.append(('reference-prefill',cid,None,None,None))
            for d in range(1,start+3):
                if d>=start:chronology.append(('reference-capture',cid,None,d,None))
                chronology.append(('reference-token',cid,None,d,None))
            chronology.append(('memory',None,None,None,'reference-chain-released'))
        for cid,B,start,fs in chosen:
            for index in indices:
                for d in range(start,start+3):
                    for event,kind in (('comparison','input-before'),('candidate-dispatch',None),('comparison','output'),('comparison','input-unchanged')):
                        chronology.append((event,cid,index,d,kind))
        chronology.append(('memory',None,None,None,'after-release'))
        same([(r['event'],r.get('case_id'),r.get('layer'),r.get('decode_step'),r.get('phase') if r['event']=='memory' else r.get('kind')) for r in events],
             chronology,'actual complete event chronology; no stale or extra dispatch')
        captured=[r for r in events if r['event']=='reference-capture'];calls=[r for r in events if r['event']=='candidate-dispatch']
        comparisons=[r for r in events if r['event']=='comparison'];prefills=[r for r in events if r['event']=='reference-prefill']
        records=read(child/'captures.json');need(len(records)==4,'four original batched reference cases')
        seen=set();counters={i:0 for i in indices};capture_index=call_index=comparison_index=0
        for case_number,(case,(cid,B,start,fs)) in enumerate(zip(records,chosen)):
            ids=[f['fixture_id'] for f in fs];lengths=[len(f['prompt_ids']) for f in fs]
            tokens=[r for r in events if r['event']=='reference-token' and r['case_id']==cid]
            raw=token_chain(case,cid,B,start,fs,tokens,vocab)
            pre=prefills[case_number]
            same([pre['case_id'],pre['row_ids'],pre['first_tokens'],pre['reference_full_logits']],
                 [cid,ids,[row[0] for row in raw],case['prefill_logits']],'prefill rows/first tokens/raw binding')
            need(len(case['prefill_logits'])==B,'all actual prefill outputs')
            for r,desc in enumerate(case['prefill_logits']):
                need(desc['path']==f'captures/{cid}/prefill-row-{r}/full-logits.safetensors','prefill artifact path')
                full=raw_file(child,desc,{'root':([1,vocab],{'BF16'})},seen)
                same(argmax_rows(full['root']),[raw[r][0]],'prefill full-vocabulary first greedy ID');result['full_vocabulary_greedy_checks']+=1
            base_shapes={'root/conv':([B,3,10240],{'BF16'}),'root/S':([B,48,128,128],{'F32'})}
            for ordinal,step in enumerate(case['steps']):
                d=start+ordinal;used=[n+d for n in lengths];event=captured[capture_index];capture_index+=1
                same([event['case_id'],event['batch_size'],event['row_ids'],event['decode_step'],event['used']],
                     [cid,B,ids,d,used],'reference actual mixed-row positions')
                need(set(step)==set(map(str,indices))==set(event['layers']),'all three selected layers')
                desc=event['reference_full_logits'];need(desc['path']==f'captures/{cid}/step-{d}/reference-full-logits.safetensors','captured full logits path')
                full=raw_file(child,desc,{'root':([B,vocab],{'BF16'})},seen)
                same(argmax_rows(full['root']),[r[d] for r in raw],'full-vocabulary actual batched greedy IDs')
                result['full_vocabulary_greedy_checks']+=B
                for index in indices:
                    record=step[str(index)]
                    same({k:v for k,v in record.items() if k!='candidate_output'},event['layers'][str(index)],'actual capture descriptor binding')
                    same([record['layer'],record['batch_size'],record['row_ids'],record['decode_step'],record['positions'],record['used'],record['input_tokens']],
                         [index,B,ids,d,[n+d-1 for n in lengths],used,[r[d-1] for r in raw]],'actual layer row/token/position binding')
            # Read one three-step layer chain at a time. Do not retain all B8 states.
            for index in indices:
                previous_state=None
                for ordinal,step in enumerate(case['steps']):
                    d=start+ordinal;record=step[str(index)]
                    for key,path in (('reference_input',f'captures/{cid}/step-{d}/layer-{index}/reference-input.safetensors'),
                                     ('reference_output',f'captures/{cid}/step-{d}/layer-{index}/reference-output.safetensors'),
                                     ('candidate_output',f'candidate/{cid}/layer-{index}/step-{d}/output.safetensors')):
                        need(record[key]['path']==path,'exact layer tensor artifact path')
                    inp=raw_file(child,record['reference_input'],{**base_shapes,'root/x':([B,5120],{'BF16'})},seen)
                    ref=raw_file(child,record['reference_output'],{**base_shapes,'root/output':([B,5120],{'BF16'})},seen)
                    got=raw_file(child,record['candidate_output'],{**base_shapes,'root/output':([B,5120],{'BF16'})},seen)
                    for name in ref:pair(ref[name],got[name],f'{cid}/{index}/{d}/{name}');result['full_output_state_pairs']+=1
                    if previous_state is not None:
                        for name in base_shapes:
                            pair(previous_state[name],inp[name],f'{cid}/{index}/continued-state/{name}');result['causal_state_pairs']+=1
                    previous_state={name:got[name] for name in base_shapes}
                    point=dict(case_id=cid,batch_size=B,row_ids=ids,layer=index,decode_step=d,used=[n+d for n in lengths])
                    dispatch(calls[call_index],point,counters[index]);call_index+=1;counters[index]+=1
                    for kind in ('input-before','output','input-unchanged'):
                        row=comparisons[comparison_index];comparison_index+=1
                        for key,value in {**point,'kind':kind}.items():same(row[key],value,'actual comparison point')
                        expected=got if kind=='output' else inp
                        need(len(row['checks'])==3 and {r['name'] for r in row['checks']}==set(expected),'all original-dtype device checks')
                        label=f'{cid}/layer-{index}/step-{d}/'+('complete-output' if kind=='output' else kind)
                        for check in row['checks']:
                            value=expected[check['name']]
                            same(check,dict(check=label,name=check['name'],shape=value['shape'],dtype=DTYPES[value['encoding']][1],
                                 reference_finite=True,candidate_finite=True,bit_equal=True),'original bit/finite comparison receipt')
                            result['device_comparison_arrays']+=1
                    del inp,ref,got
        same(complete['dispatch_counters'],{str(i):dict(candidate=12,fallback=0) for i in indices},'all actual evaluated candidate counts')
        need({str(p.relative_to(child)) for p in child.rglob('*.safetensors')}==seen and len(seen)==140,'complete original raw tensor inventory')
        need(result['full_output_state_pairs']==108 and result['causal_state_pairs']==48
             and result['full_vocabulary_greedy_checks']==80 and result['device_comparison_arrays']==324,'independent complete batched coverage')
        result.update(status='PASS',canary_qualified=True,raw_files=len(seen),complete_sha256=sha(child/'complete.json'),
            ledger_sha256=sha(child/'events.jsonl'),captures_sha256=sha(child/'captures.json'),source_pins=sources,
            frozen_raw_auditor_sha256=BASE_SHA,
            model_weights_scope='Fixed original manifest + producer pre/post verification; no offline rehash of model payloads.',
            parameter_scope='All14 selected-layer original-byte hash/shape/dtype/finite receipts before/after; no stored parameter payloads.')
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
