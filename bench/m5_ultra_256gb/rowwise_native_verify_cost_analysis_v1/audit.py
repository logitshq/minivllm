"""Default-OFF, stdlib-only saved-byte/ledger audit for closed E054 local cost.

No backend, inference, live process/service probe, or model payload read. Every
saved qualification tensor is read; timed calls retain IDs, not full snapshots.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import struct
import time
import traceback

HERE=Path(__file__).resolve().parent; B=HERE.parent; R=B.parents[1]
SCREEN=B/'rowwise_native_verify_cost_v1/screen.py'
RAW=B/'long_decode_isolation/audit_raw.py'
RAW_SHA='b3bffc7bd261b9b347b5d299d965273ff2406a92c27576747867aa515be62261'
SOURCE_PINS={
 str(SCREEN):'679ad62a8b2973c9ea6d767da3135782345ca09c870ff071ff1494005b914d4d',
 str(SCREEN.with_name('execute.py')):'cab8c315c0be075de75ae7bd5f1ea32fc1291dfb9f2defcb2736f22bd5853964',
 str(SCREEN.with_name('runner.py')):'28337c3fea0c58e87915c27a85038afb5a7ab851182a05531289fbc890d0e505',
 str(SCREEN.with_name('terminal_hold.py')):'a370512dbe1a59b1aad9c0585cccda82e56d23718f6fb6846c6bc12949805642',
 str(B/'rowwise_native_verify/qualify.py'):'0a7d81f835be9a1ac2328feec8a455bd85464077cd1661be0350e1772cde701a',
 str(B/'rowwise_native_verify/candidate.py'):'4ce1b1720dc256b01d30a0f9b19b109bde9949050e98b50a31d308f9e71c8f15',
 str(B/'exact_verifier_bridge/qualify.py'):'eda7001a1a9b08231ad3111f8b7b677b5a6d464cb065ee444dc23a4b99babd0f',
 str(B/'nvfp4_multirow_shared/candidate.py'):'d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e'}
DTYPES={'BF16':'mlx.core.bfloat16','F32':'mlx.core.float32','I32':'mlx.core.int32'}


def need(ok,message):
    if not ok: raise ValueError(message)


def unique(pairs):
    result={}
    for key,value in pairs:
        need(key not in result,'duplicate JSON key: '+key); result[key]=value
    return result


def parse(raw):
    def bad(value): raise ValueError('nonfinite JSON: '+value)
    return json.loads(raw,object_pairs_hook=unique,parse_constant=bad)


def read(path): return parse(Path(path).read_bytes())
def same(a,b,label): need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),label)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''): h.update(block)
    return h.hexdigest()


def raw_reader():
    need(sha(RAW)==RAW_SHA,'frozen raw reader source')
    spec=importlib.util.spec_from_file_location('_e054_raw_reader',RAW)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def contract():
    return dict(schema='E054-independent-cost-contract-v1',enabled_by_default=False,raw_files=32,
        full_output_state_pairs=16,tensor_pairs=2064,selected_prefixes=6,timing_count=400,group_count=20,
        device_execution=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False,
        timing_scope='Full verifier+selected k=T state+local holder commit versus native AR(T); no draft/acceptance/scheduler cost.',
        numerical_scope='16 saved full output/state pairs; timed samples retain IDs and source-bound commit checks, not400 full-tensor checks.')


def expected_labels():
    return [label for width in (4,8) for label in
            [f'T{width}/k{k}/{kind}' for k in (1,width//2,width) for kind in ('selected','continuation')]
            +[f'T{width}/{arm}' for arm in ('host','device')]]


def layout(width,canonical=False):
    result={}
    for i in range(64):
        fields=(('K',[1,4,1536,256],'BF16'),('V',[1,4,1536,256],'BF16')) if i%4==3 else (
            ('conv',[1,3,10240],'BF16'),('S',[1,48,128,128],'F32'))
        result.update({f'root/state/{i}/{key}':(shape,dtype) for key,shape,dtype in fields})
    if canonical: result.update({f'root/{key}':([1],'I32') for key in ('tokens','positions','lens')})
    else: result['root/logits']=([width,248320],'BF16')
    return result


def snapshot_metadata():
    return dict(logits={'array':'root/logits'},state=[{key:{'array':f'root/state/{i}/{key}'}
        for key in (('K','V') if i%4==3 else ('conv','S'))} for i in range(64)])


def check_layout(index,width):
    wanted=layout(width); need(set(index['tensors'])==set(wanted),'all129 original tensors including KV padding')
    for key,(shape,dtype) in wanted.items():
        same(index['tensors'][key]['shape'],shape,'full original shape: '+key)
        need(index['tensors'][key]['encoding']==dtype,'original dtype: '+key)


def greedy(raw,width,vocabulary=248320):
    need(len(raw)==width*vocabulary*2,'complete BF16 vocabulary rows')
    result=[]
    for row in range(width):
        words=struct.unpack_from('<'+str(vocabulary)+'H',raw,row*vocabulary*2)
        # BF16 finite bytes are checked by snapshot_index before this function.
        result.append(max(range(vocabulary),key=lambda i:struct.unpack('<f',struct.pack('<I',words[i]<<16))[0]))
    return result


def compare_pair(reader,left,right,label):
    value=reader.compare_snapshots(left,right)
    if not value['bit_equal']:
        first=next(t for t in value['tensors'] if not t['bit_equal'])
        raise ValueError('original-bit mismatch '+label+': '+json.dumps(first,sort_keys=True))
    return dict(label=label,tensor_pairs=value['tensor_count'],bit_equal=True,
                reference_sha256=left['sha256'],actual_sha256=right['sha256'])


def qualification(rows,ids):
    expected=[(w,k) for w in (4,8) for k in (1,w//2,w)]
    same([(r['T'],r['k']) for r in rows],expected,'six ordered selected-prefix receipts')
    for row,(width,k) in zip(rows,expected):
        same({key:row[key] for key in ('T','k','status','committed_lens','pending')},
             dict(T=width,k=k,status='PASS',committed_lens=[1022+k],pending=ids[k-1]),'selected commit/pending receipt')
        same(row['head'],dict(calls=1,status='RETURNED',x=dict(shape=[width,5120],dtype='mlx.core.bfloat16'),
            weight=dict(shape=[248320,640],dtype='mlx.core.uint32'),scales=dict(shape=[248320,320],dtype='mlx.core.uint8'),
            output=dict(shape=[width,248320],dtype='mlx.core.bfloat16')),'actual source-bound head invocation receipt')


def summarize(samples):
    groups=[]; budgets=[]
    for g in range(5):
        ratios=[]; savings=[]
        for p in range(10):
            pair=samples[g*20+p*2:g*20+p*2+2]; times={r['arm']:r['elapsed_ns'] for r in pair}
            ratios.append(times['native']/times['candidate']); savings.append(times['native']-times['candidate'])
        groups.append(dict(group=g,median_paired_ratio=statistics.median(ratios),ratios=ratios))
        budgets.append(dict(group=g,remaining_draft_and_overhead_ns=savings,
                            median_remaining_ns=statistics.median(savings),positive_pairs=sum(n>0 for n in savings)))
    totals={arm:sorted(r['elapsed_ns'] for r in samples if r['arm']==arm) for arm in ('native','candidate')}
    summary=dict(groups=groups,median_group_ratio=statistics.median(g['median_paired_ratio'] for g in groups),
        totals_ns={arm:dict(p50=statistics.median(v),p90=v[math.ceil(.9*len(v))-1]) for arm,v in totals.items()})
    return summary,dict(groups=budgets,median_group_remaining_ns=statistics.median(g['median_remaining_ns'] for g in budgets),
        positive_pairs=sum(g['positive_pairs'] for g in budgets),pair_count=50,
        scope='native AR(T) minus full verifier+selected-state commit, assuming allT proposed tokens are useful; draft and all other overhead EXCLUDED. Negative budgets retained.')


def timing_ledger(path,cases,ids,deadline):
    rows=[]
    with Path(path).open('rb') as stream:
        for line in stream:
            need(line.endswith(b'\n') and len(rows)<400,'closed400-line timing ledger'); rows.append(parse(line))
    need(len(rows)==400 and len(cases)==4,'all400 actual durations/four cases')
    summaries=[]; offset=0
    for case,(width,baseline) in zip(cases,((4,'device'),(4,'host'),(8,'device'),(8,'host'))):
        same({k:case[k] for k in ('T','k','baseline')},dict(T=width,k=width,baseline=baseline),'ordered timing case')
        samples=[]
        for group in range(5):
            for pair in range(10):
                order=['native','candidate'] if (group*10+pair)%2==0 else ['candidate','native']
                for ordinal,arm in enumerate(order):
                    row=rows[offset]; offset+=1
                    need(set(row)=={'T','baseline','group','pair','order','ordinal','arm','elapsed_ns','ids','head'},'exact raw timing fields')
                    expected=dict(T=width,baseline=baseline,group=group,pair=pair,order=order,ordinal=ordinal,arm=arm,ids=ids[:width],head=None)
                    same({k:row[k] for k in expected},expected,'raw chronological AB/BA identity and full outputIDs')
                    need(type(row['elapsed_ns']) is int and 0<row['elapsed_ns']<=deadline*10**9,'positive bounded duration')
                    samples.append({k:v for k,v in row.items() if k not in ('T','baseline')})
        same(case['samples'],samples,'complete samples equal raw ledger')
        summary,budget=summarize(samples); same(case['summary'],summary,'independently recomputed group/tail statistics')
        summaries.append(dict(T=width,baseline=baseline,summary=summary,break_even=budget))
    return summaries,sum(r['elapsed_ns'] for r in rows)


def producer_checks(complete):
    def expected(label,spec):
        return [dict(check=label,name=name,shape=shape,dtype=DTYPES[dtype],reference_finite=True,candidate_finite=True,bit_equal=True)
                for name,(shape,dtype) in spec.items()]
    wanted=[]
    for label in expected_labels():
        width=1 if label.endswith('/continuation') else int(label[1])
        wanted.extend(expected(label,layout(width)))
    for arm in ('canonical','oracle','host','device','candidate'): wanted.extend(expected('canonical-preserved/'+arm,layout(1,True)))
    order=lambda row:(row['check'],row['name'])
    same(sorted(complete['checks'],key=order),sorted(wanted,key=order),'complete source-bound initial comparisons')
    wanted=[]
    for arm in ('canonical','oracle','host','device','candidate'): wanted.extend(expected('post-timing-canonical/'+arm,layout(1,True)))
    same(sorted(complete['post_checks'],key=order),sorted(wanted,key=order),'complete source-bound post-timing comparisons')


def closure(parent_path,parent_sha,child,remember):
    def get(path): remember[str(path)]=sha(path); return read(path)
    parent=get(parent_path); need(remember[str(parent_path)]==parent_sha,'explicit actual parent plan SHA')
    need(parent['schema']=='E054-c1-verifier-cost-owner-plan-v1' and parent['outputs']['numerical']==str(child),'fixed executed parent/child')
    pins=parent['pins']; need(all(pins.get(p)==s for p,s in SOURCE_PINS.items()),'fixed numerical/owner source closure')
    execution=Path(parent['outputs']['execution']); owner=Path(parent['outputs']['screen']); lease=Path(parent['outputs']['lease'])
    # Check settled ownership BEFORE any numerical tensor is opened.
    final=get(execution/'final.json'); state=get(lease/'lease.json'); resume=get(lease/'resume.json'); job=get(lease/'job-0.json')
    need(final['status']=='PASS_CAPTURE_AND_RESTORE' and final['errors']==[] and final['lease_returncode']==0
         and final['service_identity_after']==parent['service']['identity'],'driver successful restore receipt')
    need(state['closed'] is True and state['service_pid']==parent['service']['pid']
         and ' '.join(state['service_identity'].split())==parent['service']['identity'],'closed same-service lease')
    need(resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after'],'resume receipt')
    same(get(execution/'plan.json'),parent,'retained preregistered parent plan')
    actual=get(execution/'execution-plan.json'); same({k:v for k,v in actual.items() if k!='physical_baseline'},parent,'only registered physical baseline added')
    need('physical_baseline' in actual,'saved admission baseline'); same(get(owner/'plan.json'),actual,'same actual owner plan')
    command=[str(R/'.venv/bin/python'),'-I','-B',str(SCREEN.with_name('runner.py')),'--plan',str(execution/'execution-plan.json'),
             '--plan-sha256',remember[str(execution/'execution-plan.json')],'--out',str(owner)]
    same(state['jobs'],[command],'one registered owner job'); same(job['command'],command,'waited exact owner command')
    identity=get(execution/'runner-identity.json')
    need(job['waited'] is True and job['returncode']==0 and job['pid']==state['child_pid']==identity['pid']
         and identity['ppid']==state['owner_pid'],'owner wait identity chain')
    status=get(owner/'result.json'); reap=get(owner/'child-reap.json')
    need(status['schema']=='E054-c1-cost-owner-v1' and status['status']=='PASS_CAPTURE_ONLY' and status['errors']==[]
         and status['child_reaped'] is True and status['child_returncode']==0,'numerical owner PASS/reap')
    need(reap['waited'] is True and reap['absent'] is True and reap['returncode']==0 and reap['errors']==[],'saved direct numerical reap')
    done=get(owner/'numerical-done.json'); ack=get(owner/'numerical-exit-ack.json')
    need(done['schema']=='E054-terminal-hold-v1' and done['pid']==reap['pid'] and done['ppid']==job['pid']
         and done['qualifier_returncode']==0 and done['wrapper_errors']==[],'terminal original successful exit')
    same(done['command'],parent['numerical_command'],'terminal exact screen command')
    same(ack['command'],done['command'],'ack command'); need(ack['pid']==done['pid'] and ack['owner_pid']==done['ppid']
        and ack['done_sha256']==remember[str(owner/'numerical-done.json')] and ack['final_physical_and_metadata_complete'] is True,'bound terminal ack')
    n=status['physical_samples']; m=status['resource_samples']; need(type(n) is int and n==m and n>=2,'complete final observations')
    for name,path in (('physical_sha256',owner/'resources'/f'physical-{n-1:04d}'/'sample.json'),
                      ('metadata_sha256',owner/'resources'/f'resource-{m-1:04d}.json')):
        get(path); need(ack[name]==remember[str(path)],'ack bound final observation '+name)
    need(not (owner/'terminal-hold-error.json').exists(),'no terminal protocol failure')
    copies=get(execution/'source-copies.json'); need(len(copies)==len(pins) and {c['original'] for c in copies}==set(pins),'complete retained source closure')
    for row in copies:
        path=Path(row['copy']); need(path.resolve().is_relative_to(execution.resolve()/'sources') and path.stat().st_size==row['bytes']<16<<20,'bounded saved source copy')
        need(row['sha256']==pins[row['original']]==sha(path)==sha(row['original']),'saved/current source pin mismatch')
        remember[str(path)]=row['sha256']; remember[row['original']]=row['sha256']
    same(get(execution/'runtime-pins-verified.json'),parent['runtime_pins'],'source-bound runtime payload hash receipt')
    complete=get(child/'complete.json'); need(done['qualifier_complete']==dict(path=str(child/'complete.json'),
        sha256=remember[str(child/'complete.json')],status='PASS_LOCAL_COST_ONLY'),'complete terminal SHA/status')
    need(complete['pid']==done['pid'] and complete['ppid']==done['ppid'],'numeric complete identity')
    plan=get(child/'plan.json'); need(remember[str(child/'plan.json')]==complete['plan_sha256']==parent['numerical_plan_sha256'],'actual child plan SHA')
    same(parent['numerical_command'],[str(R/'.venv/bin/python'),'-I','-B',str(SCREEN),'--run','--plan',str(SCREEN.parent/'prereg-v1/child-plan.json'),
        '--plan-sha256',parent['numerical_plan_sha256'],'--out',str(child)],'actual registered screen invocation')
    need(set(plan)=={'schema','screen_sha256','e042_plan','e042_result','deadline_s'} and plan['schema']=='C1-verifier-cost-plan-v1'
         and plan['screen_sha256']==SOURCE_PINS[str(SCREEN)] and type(plan['deadline_s']) in (int,float)
         and math.isfinite(plan['deadline_s']) and 0<plan['deadline_s']<=1800,'fixed bounded child contract')
    prior=[]
    for key in ('e042_plan','e042_result'):
        ref=plan[key]; need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute() and pins[ref['path']]==ref['sha256'],'parent bound E042 input')
        prior.append(get(Path(ref['path']))); need(remember[ref['path']]==ref['sha256'],'unchanged E042 metadata')
    need(prior[1]['status']=='PASS' and prior[1]['forward_prefix_qualified'] is True and prior[1]['plan_sha256']==plan['e042_plan']['sha256'],'actual E042 qualified predecessor')
    seeds=[s for s in prior[1]['reference_blocks'] if s['used']==1023]
    need(len(seeds)==1 and seeds[0]['status']=='PASS' and len(seeds[0]['tokens'])==8,'one complete original E042 seed')
    same(complete['seed'],seeds[0],'original fixed seed identity'); need(complete['seed']['position']==1022,'actual position1022')
    return parent,plan,complete


def audit(child,parent_path,parent_sha):
    started=time.monotonic(); child=Path(child).resolve(); parent_path=Path(parent_path).resolve(); remember={}
    result=dict(schema='E054-local-cost-independent-v1',status='INCOMPLETE',contract=contract(),child=str(child),parent_plan_sha256=parent_sha,
        audit_source_sha256=sha(__file__),raw_reader_sha256=RAW_SHA,raw_files=0,tensor_pairs=0,pairs=[],timings=[],
        local_cost_audited=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False,
        limits=['All16 saved qualification pairs are checked including finite original bits and complete KV padding; no400 full-tensor timing claim.',
                'Six committed lens/pending/head records and live seed preservation are source-bound producer receipts; saved selected/continuation outputs are independently checked.',
                'Parameter before/after original-byte receipts are compared. Model/runtime payloads are not rehashed; live process absence/restoration and resource-policy compliance are not re-probed.',
                'Break-even is an allT-accepted upper budget for draft+other overhead. It excludes draft cost, rejection/acceptance, scheduler/HTTP and is not engine speedup.'])
    try:
        parent,plan,complete=closure(parent_path,parent_sha,child,remember)
        need(complete['schema']=='C1-verifier-cost-result-v1' and complete['status']=='PASS_LOCAL_COST_ONLY','producer successful terminal')
        for key in ('numerical_screen_qualified','device_cleanup_completed','source_model_pins_rechecked'): need(complete[key] is True,'producer final gate '+key)
        for key in ('http_qualified','performance_qualified','engine_pareto_point'): need(complete[key] is False,'excluded claim')
        for key in ('short_prefix_cost_measured','compiled_qualified','rollback_qualified','full_model_qualified','http_qualified','performance_qualified','engine_pareto_point'):
            need(complete['contract'][key] is False,'excluded contract '+key)
        same(complete['device'],dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3'),'actual device receipt')
        expected_sources={p:s for p,s in SOURCE_PINS.items() if p==str(SCREEN) or p.endswith('/rowwise_native_verify/qualify.py') or p.endswith('/rowwise_native_verify/candidate.py') or p.endswith('/nvfp4_multirow_shared/candidate.py')}
        same(complete['sources'],expected_sources,'producer actual execution sources')
        need(not complete.get('error') and not (child/'failure.json').exists(),'no producer failure')
        elapsed=complete['elapsed_wall_s']; need(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<elapsed<=plan['deadline_s'],'actual bounded elapsed')
        reader=raw_reader(); indices={}; raw_stats={}; seen=set()
        same([r['label'] for r in complete['raw_comparisons']],expected_labels(),'all16 ordered raw pairs')
        for row in complete['raw_comparisons']:
            label=row['label']; width=1 if label.endswith('/continuation') else int(label[1]); need(len(row['files'])==2,'exact reference+actual pair')
            pair=[]
            for arm,desc in zip(('reference','actual'),row['files']):
                relative=label+'/'+arm+'.safetensors'; need(desc['path']==relative,'fixed saved array path')
                path=child/relative; need(not path.is_symlink(),'no snapshot alias'); st=path.stat(); raw_stats[str(path)]=(st.st_size,st.st_mtime_ns,st.st_ino)
                sidecar=path.with_suffix('.json'); remember[str(sidecar)]=sha(sidecar); same(read(sidecar),desc,'exact saved sidecar')
                same(desc['metadata'],snapshot_metadata(),'complete snapshot metadata'); index=reader.snapshot_index(child,desc); check_layout(index,width)
                pair.append(index); seen.add(relative); result['raw_files']+=1
            result['pairs'].append(compare_pair(reader,*pair,label)); result['tensor_pairs']+=len(pair[0]['tensors']); indices[label]=pair[0]
        need({str(p.relative_to(child)) for p in child.rglob('*.safetensors')}==seen and len(seen)==32 and result['tensor_pairs']==2064,'all32 files/2064 tensor pairs; no hidden failure arrays')
        full=reader.tensor_bytes(indices['T8/device'],'root/logits'); ids=greedy(full,8)
        need(ids[:7]==complete['seed']['tokens'][1:] and not set(ids)&set(complete['seed']['original_fixture']['eos_token_ids']),'saved causal outputIDs and normal EOS')
        state_anchors={}
        for label,index in indices.items():
            parts=label.split('/'); width=int(parts[0][1:]); count=int(parts[1][1:]) if parts[1].startswith('k') else width
            prefix=count+int(label.endswith('/continuation'))
            raw=reader.tensor_bytes(index,'root/logits')
            if label.endswith('/continuation'):
                if count<8: need(raw==full[count*248320*2:(count+1)*248320*2],'continuation same oracle row '+label)
                need(not set(greedy(raw,1))&set(complete['seed']['original_fixture']['eos_token_ids']),'continuation normal EOS')
            else: need(raw==full[:width*248320*2],'same original oracle logits '+label)
            states={k:{f:v[f] for f in ('shape','encoding','sha256')} for k,v in index['tensors'].items() if k.startswith('root/state/')}
            if prefix in state_anchors: same(states,state_anchors[prefix],'same full saved oracle state at prefix '+str(prefix))
            else: state_anchors[prefix]=states
        qualification(complete['qualification'],ids); producer_checks(complete)
        values=[]
        for name in ('parameters-before.json','parameters-after.json'):
            path=child/name; remember[str(path)]=sha(path); receipt=read(path); reader.parameter_receipt(receipt); values.append(reader.parameter_values(receipt))
        same(*values,'all1349 original parameter value receipts unchanged')
        timing_path=child/'timings.jsonl'; remember[str(timing_path)]=sha(timing_path)
        result['timings'],total_ns=timing_ledger(timing_path,complete['timings'],ids,plan['deadline_s'])
        need(total_ns<=elapsed*1e9,'nonoverlapping retained intervals bounded by whole run')
        memory=complete['memory_after_cleanup']; need(set(memory)=={'active_bytes','peak_bytes','cache_bytes'} and all(type(v) is int and v>=0 for v in memory.values()),'complete memory receipt')
        need(all(sha(p)==s for p,s in remember.items()),'small source/evidence changed during audit')
        need(all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns,Path(p).stat().st_ino)==value for p,value in raw_stats.items()),'raw file changed during audit')
        result.update(status='PASS',local_cost_audited=True,timing_count=400,group_count=20,selected_prefix_count=6,
            total_retained_ns=total_ns,saved_oracle_ids=ids,complete_sha256=remember[str(child/'complete.json')],
            source_and_small_evidence=remember,parameter_count=1349)
    except BaseException as error:
        result.update(status='FAIL',local_cost_audited=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    result['elapsed_wall_s']=time.monotonic()-started; return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true'); p.add_argument('--child',type=Path)
    p.add_argument('--parent-plan',type=Path); p.add_argument('--parent-plan-sha256'); p.add_argument('--out',type=Path); a=p.parse_args(argv)
    if not a.run:
        if any(v is not None for v in (a.child,a.parent_plan,a.parent_plan_sha256,a.out)): p.error('explicit --run required')
        print(json.dumps(contract(),indent=2)); return 0
    need(all(v is not None for v in (a.child,a.parent_plan,a.parent_plan_sha256,a.out)) and not a.out.exists(),'explicit complete inputs and fresh output')
    result=audit(a.child,a.parent_plan,a.parent_plan_sha256)
    with a.out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','raw_files','tensor_pairs','elapsed_wall_s')}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
