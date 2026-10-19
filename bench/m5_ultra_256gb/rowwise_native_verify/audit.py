"""Offline original-byte audit of the nine fixed rowwise forward cases.

No arguments describes the contract; no model, MLX or qualifier imports. The
frozen bounded reader checks every stored float for finiteness and compares
unsigned original words. PASS covers 42 fixed prefixes only, never performance.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
RAW = HERE.parent/'long_decode_isolation/audit_raw.py'
RAW_SHA = 'b3bffc7bd261b9b347b5d299d965273ff2406a92c27576747867aa515be62261'
CANDIDATE_SHA = '5903f60ceef9ff5429e21c0c3d38b6ac67a6a25423d71a6cafc47715ac0f6b34'
BRIDGE = HERE.parent/'exact_verifier_bridge/qualify.py'
BRIDGE_SHA = 'eda7001a1a9b08231ad3111f8b7b677b5a6d464cb065ee444dc23a4b99babd0f'
E024_SHA = '5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
E023_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
CORE = HERE.parent/'long_decode_graph/model-source-pins.json'
CORE_SHA = '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5'
WIDTHS, USED = (2,4,8), (257,1023,1024)
DTYPES = {'BF16':'mlx.core.bfloat16','F32':'mlx.core.float32','I32':'mlx.core.int32'}
EXCLUDED = ('compiled_qualified','rollback_qualified','full_model_qualified','http_qualified','performance_qualified')


def need(ok,message):
    if not ok: raise ValueError(message)


def unique(pairs):
    result={}
    for key,value in pairs:
        need(key not in result,'duplicate JSON key: '+key); result[key]=value
    return result


def parse(data):
    def bad(value): raise ValueError('nonfinite JSON: '+value)
    return json.loads(data,object_pairs_hook=unique,parse_constant=bad)


def read(path): return parse(Path(path).read_bytes())
def same(left,right,label):
    need(json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False),label)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''): digest.update(block)
    return digest.hexdigest()
def finite(value): return type(value) in (int,float) and math.isfinite(value) and value>=0


def raw_reader():
    need(sha(RAW)==RAW_SHA,'frozen original-byte reader source')
    spec=importlib.util.spec_from_file_location('_rowwise_raw_reader',RAW)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def contract():
    return dict(schema='Rowwise-native-verifier-independent-contract-v1',device_execution=False,
        cases=9,prefixes=42,raw_files=96,output_tensor_pairs=5418,reference_vocabulary_rows=24,
        candidate_vocabulary_rows=42,forward_prefix_qualified=False,**{key:False for key in EXCLUDED})


def ref_json(ref,digest=None):
    need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute(),'absolute exact JSON ref')
    need(sha(ref['path'])==ref['sha256'] and (digest is None or ref['sha256']==digest),'JSON ref SHA')
    return read(ref['path'])


def literal(path,name):
    values=[ast.literal_eval(node.value) for node in ast.parse(Path(path).read_text()).body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)]
    need(len(values)==1,'one actual source pin: '+name); return values[0]


def identities(child,complete,plan_sha):
    plan=read(child/'plan.json')
    need(sha(child/'plan.json')==plan_sha==complete['plan_sha256'],'actual registered child plan SHA')
    need(set(plan)=={'schema','qualifier_sha256','candidate','bridge','e024_plan','deadline_s'}
         and plan['schema']=='Rowwise-native-verifier-plan-v1','exact rowwise plan')
    need(finite(plan['deadline_s']) and 0<plan['deadline_s']<=1800,'bounded deadline')
    need(plan['qualifier_sha256']==sha(HERE/'qualify.py'),'actual qualifier source')
    for key,path,digest in (('candidate',HERE/'candidate.py',CANDIDATE_SHA),('bridge',BRIDGE,BRIDGE_SHA)):
        same(plan[key],dict(path=str(path),sha256=digest),'fixed adjacent '+key)
        need(sha(path)==digest and literal(HERE/'qualify.py',key.upper()+'_SHA')==digest,'sealed '+key+' binding')
    old=ref_json(plan['e024_plan'],E024_SHA); inherited=ref_json(old['e023_plan'],E023_SHA)
    same(read(child/'inherited-e024-plan.json'),old,'retained inherited E024 plan')
    same(read(child/'inherited-e023-plan.json'),inherited,'retained inherited E023 plan')
    sources={str(HERE/'qualify.py'):plan['qualifier_sha256'],str(HERE/'candidate.py'):CANDIDATE_SHA,str(BRIDGE):BRIDGE_SHA,
        str(HERE.parent/'long_decode_isolation/qualify.py'):'9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e',
        str(HERE.parent/'long_decode_graph/qualify.py'):'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
        str(CORE):CORE_SHA,old['e023_qualifier']['path']:'26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'}
    same(complete['sources'],sources,'complete exact executable source map')
    need(all(sha(path)==digest for path,digest in sources.items()),'source pins changed')
    source=ref_json(inherited['source_manifest']); same(source['files'],read(CORE)['files'],'complete fixed core manifest')
    root=Path(inherited['package_root'])/'tutorial_code/miniengine'
    need(set(source['files'])=={str(p.relative_to(root)) for p in root.rglob('*.py')}
         and all(sha(root/name)==digest for name,digest in source['files'].items()),'actual complete frozen core')
    manifest=ref_json(inherited['model_manifest']); fixtures=ref_json(inherited['fixtures'])
    need(sha(child/'fixtures.json')==inherited['fixtures']['sha256'],'retained exact original fixtures')
    need(manifest['schema']=='E023-model-v1' and Path(manifest['model']).resolve()==Path(inherited['model']).resolve()
         and manifest['revision']==Path(inherited['model']).resolve().name
         and Path(fixtures['model']).resolve()==Path(inherited['model']).resolve(),'model/fixture identity')
    return selected(fixtures['fixtures']),sources


def selected(fixtures):
    need(len(fixtures)==8 and len({f['fixture_id'] for f in fixtures})==8,'eight distinct original fixtures')
    for f in fixtures:
        need(f['normal_eos'] is True and type(f['max_output_tokens']) is int and f['max_output_tokens']==512
             and 886<=len(f['prompt_ids'])<=978 and all(type(t) is int and 0<=t<248320 for t in f['prompt_ids'])
             and f['eos_token_ids'] and all(type(t) is int and 0<=t<248320 for t in f['eos_token_ids']),'fixed D512 fixture')
    return min(fixtures,key=lambda f:(len(f['prompt_ids']),f['fixture_id']))


def teacher(seed,fixture,used):
    prompt=fixture['prompt_ids'][:256] if used==257 else fixture['prompt_ids']; start=used-len(prompt)
    same({k:seed[k] for k in ('fixture_id','original_fixture','prompt_ids','domain','used','position','start','backing')},
        dict(fixture_id=fixture['fixture_id'],original_fixture=fixture,prompt_ids=prompt,
            domain='diagnostic-prefix256' if used==257 else 'original-full-prompt',used=used,position=used-1,start=start,backing=1536),
        'exact seed domain and absolute position')
    raw=seed['raw_ids']
    need(0<start and start+8<=512 and len(raw)==start+8 and all(type(t) is int and 0<=t<248320 for t in raw),'complete teacher history')
    need(not set(raw[:-1])&set(fixture['eos_token_ids']),'normal EOS before teacher completion')
    same(seed['prefill_token'],raw[0],'prefill ID'); same(seed['tokens'],raw[start-1:start+7],'causal teacher input slice')


def expected_layout(kind,width=1,batch=1):
    result={}
    for i in range(64):
        fields=(('K',[1,4,1536,256],'BF16'),('V',[1,4,1536,256],'BF16')) if i%4==3 else (
            ('conv',[1,3,10240],'BF16'),('S',[1,48,128,128],'F32'))
        result.update({f'root/state/{i}/{key}':(shape,dtype) for key,shape,dtype in fields})
    if kind=='output': return {'root/logits':([1,248320],'BF16'),**result}
    need(kind in ('canonical','input'),'fixed snapshot kind')
    return {**result,'root/tokens':([width],'I32'),'root/positions':([batch],'I32'),'root/lens':([batch],'I32')}


def metadata(kind,position,width=1,token=None,prefix=None):
    state=[{key:{'array':f'root/state/{i}/{key}'} for key in (('K','V') if i%4==3 else ('conv','S'))} for i in range(64)]
    if kind=='output': return dict(logits={'array':'root/logits'},state=state,position=position,lens=position+1,
                                  max_len=position+1,input_token=token,prefix=prefix)
    result=dict(state=state,**{key:{'array':'root/'+key} for key in ('tokens','positions','lens')})
    if kind=='input': result.update(max_len=position+1,mode='decode',prefill_shape=None)
    return result


def layout(index,kind,width=1,batch=1):
    expected=expected_layout(kind,width,batch); need(set(index['tensors'])==set(expected),'all original tensor names')
    for key,(shape,dtype) in expected.items():
        same(index['tensors'][key]['shape'],shape,'original shape: '+key)
        need(index['tensors'][key]['encoding']==dtype,'original dtype: '+key)


class BitMismatch(ValueError):
    def __init__(self,point,contrast):
        first=next(row for row in contrast['tensors'] if not row['bit_equal']); name=first['name']; parts=name.split('/')
        self.details=dict(case=point,array=name,layer=int(parts[2]) if name.startswith('root/state/') else None,
            **first['first_difference'],differing_tensors=contrast['differing_tensors'],
            differing_words=contrast['differing_words'],differing_bytes=contrast['differing_bytes'])
        super().__init__('original-byte mismatch: '+point+'/'+name)


def compare_pair(b,left,right,point):
    comparison=b.compare_snapshots(left,right)
    if not comparison['bit_equal']: raise BitMismatch(point,comparison)
    return comparison


def greedy(b,index):
    raw=b.tensor_bytes(index,'root/logits'); need(len(raw)==248320*2,'full vocabulary')
    words=struct.unpack('<248320H',raw)
    return max(range(248320),key=lambda i:struct.unpack('<f',struct.pack('<I',words[i]<<16))[0])


def events(child,complete):
    expected=[('memory','before-load')]
    for seed in complete['reference_blocks']:
        expected.extend(('reference-token',dict(used=seed['used'],fixture_id=seed['fixture_id'],decode_step=i,
            input_token=seed['raw_ids'][i-1],output_token=seed['raw_ids'][i])) for i in range(1,len(seed['raw_ids'])))
        expected.extend(('case',dict(used=seed['used'],width=w,status='PASS')) for w in WIDTHS)
    expected.append(('memory','after-release')); previous=-1
    with (child/'events.jsonl').open('rb') as stream:
        for kind,wanted in expected:
            line=stream.readline(); need(line.endswith(b'\n'),'missing/partial ordered event')
            row=parse(line); elapsed=row.pop('elapsed_wall_s'); actual=row.pop('event')
            need(finite(elapsed) and previous<=elapsed<=complete['elapsed_wall_s'],'ordered event time')
            previous=elapsed; need(actual==kind,'actual event order')
            if kind=='memory':
                need(set(row)=={'phase','active_bytes','peak_bytes','cache_bytes','not_performance'} and row['phase']==wanted
                     and row['not_performance'] is True and all(type(row[k]) is int and row[k]>=0
                     for k in ('active_bytes','peak_bytes','cache_bytes')),'memory event')
            else: same(row,wanted,'event/terminal teacher and case binding')
        need(not stream.read(1),'unexpected event after release')


def audit(child,plan_sha256):
    child=Path(child).resolve(); seen={}
    result=dict(schema='Rowwise-native-verifier-independent-v1',status='INCOMPLETE',forward_prefix_qualified=False,
        **{key:False for key in EXCLUDED},child_directory=str(child),plan_sha256=plan_sha256,cases=[],raw_files=0,
        limits=['Only nine fixed teacher-forced blocks and all42 prefixes; no acceptance/rollback, compile, HTTP or speed qualification.',
            'Pre-anchor arithmetic and separate device execution/storage ownership are source-bound producer receipts, not independently recomputed histories.',
            'All parameter before/after original-byte receipts are compared; parameter payloads and model/runtime payloads are not rehashed here.',
            'Input preservation is recomputed from saved original bytes, including complete KV padding; no resource/lifecycle qualification.'])
    try:
        complete=read(child/'complete.json'); terminal_sha=sha(child/'complete.json')
        need(complete['schema']=='Rowwise-native-verifier-result-v1','fixed producer schema')
        result['producer_status']=complete['status']; result['producer_error']=complete.get('error')
        need(complete['status'] in ('PASS','FAIL','INCOMPLETE'),'known producer status')
        if not complete.get('sources'):
            result['first_failure']='producer stopped before source binding'; return result
        need(all(complete[key] is False for key in EXCLUDED),'excluded claims stay false')
        fixture,sources=identities(child,complete,plan_sha256); result['source_pins']={**sources,str(RAW):RAW_SHA}; b=raw_reader()
        def saved(desc,path,kind,position,width=1,batch=1,token=None,prefix=None):
            need(desc['path']==path,'fixed raw artifact path'); same(read((child/path).with_suffix('.json')),desc,'raw sidecar binding')
            same(desc['metadata'],metadata(kind,position,width,token,prefix),'original snapshot metadata')
            index=b.snapshot_index(child,desc); layout(index,kind,width,batch); seen[path]=desc['sha256']; return index
        def integers(index,name,expected):
            raw=b.tensor_bytes(index,'root/'+name); same(list(struct.unpack('<'+'i'*len(expected),raw)),expected,'actual I32 '+name)
        def state_pair(left,right,point):
            def states(index): return dict(index,tensors={k:v for k,v in index['tensors'].items() if k.startswith('root/state/')})
            return compare_pair(b,states(left),states(right),point)
        blocks=complete['reference_blocks']; rows=complete['cases']; need(len(blocks)<=3 and len(rows)<=9,'no extra anchors/cases')
        cursor=0
        for used,seed in zip(USED,blocks):
            teacher(seed,fixture,used); p=used-1; base=f'seeds/used{used}'
            canonical=saved(seed['canonical'],base+'/canonical.safetensors','canonical',p)
            after=saved(seed['reference_input_after'],base+'/canonical-after-reference.safetensors','canonical',p)
            compare_pair(b,canonical,after,base+'/canonical-preserved')
            for key,value in (('tokens',seed['tokens'][0]),('positions',p),('lens',used)): integers(canonical,key,[value])
            need(len(seed['reference_outputs'])==8,'eight independent reference prefix outputs')
            references=[]
            for i,desc in enumerate(seed['reference_outputs']):
                ref=saved(desc,base+f'/reference-p{i+1}.safetensors','output',p+i,token=seed['tokens'][i],prefix=i+1)
                same(greedy(b,ref),seed['raw_ids'][seed['start']+i],'full vocabulary teacher continuation'); references.append(ref)
            if used==257:
                record=complete['rejections']; same(read(child/'rejections.json'),record,'retained rejection record')
                need(len(record['cases'])==3,'all three rejection domains')
                for item,label,width,positions,lens in zip(record['cases'],('T1','B2','past1536'),(1,2,2),([256],[256,256],[1535]),([257],[257,257],[1536])):
                    same(item['label'],label,'fixed rejection order'); path='rejections/'+label
                    left=saved(item['input_before'],path+'-before.safetensors','input',positions[0],width,len(positions))
                    right=saved(item['input_after'],path+'-after.safetensors','input',positions[0],width,len(positions))
                    integers(left,'tokens',seed['tokens'][:width]); integers(left,'positions',positions); integers(left,'lens',lens)
                    state_pair(canonical,left,path+'/canonical'); compare_pair(b,left,right,path+'/unchanged')
                    need(item['status']=='PASS','rejection did not complete')
            for width in WIDTHS:
                if cursor==len(rows): break
                row=rows[cursor]; same([row['used'],row['width']],[used,width],'fixed case order'); tag=f'cases/used{used}/T{width}'
                left=saved(row['input_before'],tag+'/input-before.safetensors','input',p,width)
                right=saved(row['input_after'],tag+'/input-after.safetensors','input',p,width)
                for key,value in (('tokens',seed['tokens'][:width]),('positions',[p]),('lens',[used])): integers(left,key,value)
                state_pair(canonical,left,tag+'/canonical'); compare_pair(b,left,right,tag+'/input-preserved')
                need(len(row['prefixes'])<=width,'no extra candidate prefixes'); outputs=[]; pairs=0
                for i,item in enumerate(row['prefixes']):
                    same(item['prefix'],i+1,'prefix order'); same(item['reference'],seed['reference_outputs'][i],'same native reference')
                    candidate=saved(item['candidate'],tag+f'/candidate-p{i+1}.safetensors','output',p+i,token=seed['tokens'][i],prefix=i+1)
                    pairs+=compare_pair(b,references[i],candidate,tag+f'/prefix{i+1}')['tensor_count']; outputs.append(greedy(b,candidate))
                    need(item['status']=='PASS','prefix checks incomplete')
                if len(outputs)!=width: raise FileNotFoundError('missing candidate prefixes: '+tag)
                same(outputs,row['candidate_greedy_ids'],'candidate full saved vocabulary'); same(outputs,seed['raw_ids'][seed['start']:seed['start']+width],'causal output IDs')
                need(row['status']=='PASS','case checks incomplete'); result['cases'].append(dict(case=tag,prefixes=width,output_tensor_pairs=pairs)); cursor+=1
            need(seed['status']=='PASS','reference seed checks incomplete')
        result['raw_files']=len(seen)
        if complete['status']!='PASS' or len(blocks)!=3 or cursor!=9:
            result.update(status='FAIL' if complete['status']=='FAIL' else 'INCOMPLETE',first_failure='producer did not complete all nine cases and postchecks'); return result
        need(complete['forward_prefix_qualified'] is True and complete['source_model_pins_rechecked'] is True
             and complete['device_cleanup_completed'] is True and complete['device_run'] is True
             and not complete.get('error') and not complete.get('first_device_error'),'complete producer gates')
        same(complete['device'],dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3'),'fixed device/runtime receipt')
        need(finite(complete['elapsed_wall_s']),'terminal elapsed time'); values=[]
        for name in ('parameters-before','parameters-after'):
            desc=complete[name]; same(desc,dict(path=name+'.json',sha256=sha(child/(name+'.json')),scope='producer original-byte parameter receipt'),'parameter receipt binding')
            receipt=read(child/desc['path']); b.parameter_receipt(receipt); values.append(b.parameter_values(receipt))
        same(*values,'unchanged1349 parameter value receipts'); events(child,complete)
        need(len(seen)==96 and {str(p.relative_to(child)) for p in child.rglob('*.safetensors')}==set(seen),'all96 raw files and no extras')
        need(not (child/'failure.json').exists(),'producer failure artifact remains')
        need(all(sha(child/path)==digest for path,digest in seen.items()) and sha(child/'complete.json')==terminal_sha,'evidence changed during audit')
        result.update(status='PASS',forward_prefix_qualified=True,complete_sha256=terminal_sha,audit_sha256=sha(__file__),
            output_tensor_pairs=sum(row['output_tensor_pairs'] for row in result['cases']),reference_vocabulary_rows=24,candidate_vocabulary_rows=42,
            parameter_value_receipts=1349,ledger_sha256=sha(child/'events.jsonl'))
    except BitMismatch as exc: result.update(status='FAIL',first_failure=str(exc),first_difference=exc.details)
    except FileNotFoundError as exc: result['first_failure']='missing required evidence: '+str(exc)
    except Exception as exc: result.update(status='FAIL',first_failure=type(exc).__name__+': '+str(exc))
    finally: result['raw_files']=len(seen)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--child',type=Path); p.add_argument('--plan-sha256'); p.add_argument('--out',type=Path)
    args=p.parse_args(argv)
    if args.child is None and args.plan_sha256 is None and args.out is None:
        print(json.dumps(contract(),indent=2)); return 0
    if args.child is None or args.plan_sha256 is None or args.out is None: p.error('all three audit arguments required')
    need(not args.out.exists(),'fresh independent output required'); result=audit(args.child,args.plan_sha256)
    with args.out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__': raise SystemExit(main())
