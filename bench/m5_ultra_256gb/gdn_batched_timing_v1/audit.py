"""Independent stdlib-only E048 raw-byte and timing-ledger audit; default OFF.

Read only after owner/lease closure. Does not import the producer/backend, rerun
inference, query services or rehash weights. Actual per-call dispatch and parameter
payloads remain source-bound producer evidence, explicitly separated below.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
import traceback

HERE=Path(__file__).resolve().parent; R=HERE.parents[2]; B=HERE.parent; A=R/'.artifacts/m5-ultra-256gb'
READER=B/'gdn_prework_batched/audit.py'; READER_SHA='911a3008d126ec5a5f3c6f25df17bea54e5333251639f5eee82b66a67f95eb04'
PARENT_SHA='8c1fdfe2bcaa5d45e49b87120650949b879eef616ec1aaf881e49a00a381386a'
SCREEN_SHA='aa8972ddc48f74e45cdd4cdaa6c7c3a81520a3d8dfafb7b401d8f4c7bc02ec70'
CHILD_PLAN_SHA='7df6cff99425698a1fb134df4f25dc8de4e717482f73039e56ec5a0a00930003'
ORIGINAL=A/'gdn-batched-v1-numerical'; ORIGINAL_AUDIT=A/'gdn-batched-v1-independent-numerical.json'
ORIGINAL_AUDIT_SHA='ce6c9c840360f336febe20d84081da2c84b3e93252b1b7733d3a9761358a1bce'
ORIGINAL_COMPLETE_SHA='94e6b0a2dbb9c1beebf3026258709532837219b3aaa69379fa9d7ce9c22dde3b'
CAPTURES_SHA='ea7b87878037ceec085d97958c9b41972bf80a6cff1a3563771fdf14b9d1dc99'

if hashlib.sha256(READER.read_bytes()).hexdigest()!=READER_SHA: raise ValueError('frozen raw reader changed')
_spec=importlib.util.spec_from_file_location('_e048_offline_raw_reader',READER)
raw=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(raw)
need,same,read,sha,local=raw.need,raw.same,raw.read,raw.sha,raw.local


def ledger(path):
    rows=[]
    with Path(path).open() as stream:
        for line in stream:
            need(line.endswith('\n'),'truncated timing ledger line')
            rows.append(json.loads(line,object_pairs_hook=raw.unique))
            need(len(rows)<=110,'unexpected extra timing/warmup row')
    need(len(rows)==110,'five warmups per arm and one hundred actual measured rows')
    expected=[]
    for warmup in range(5):
        for arm in ('native','candidate'): expected.append(dict(phase='warmup',warmup=warmup,arm=arm))
    for group in range(5):
        for pair in range(10):
            arms=('native','candidate') if pair%2==0 else ('candidate','native')
            for slot,arm in enumerate(arms): expected.append(dict(phase='measured',group=group,pair=pair,slot=slot,arm=arm))
    for observed,identity in zip(rows,expected):
        need(set(observed)==set(identity)|{'elapsed_ns'},'exact timing row schema')
        same({k:observed[k] for k in identity},identity,'complete chronological AB/BA identity')
        need(type(observed['elapsed_ns']) is int and 0<observed['elapsed_ns']<=900*10**9,'positive bounded actual elapsed ns')
    groups=[]
    for group in range(5):
        med={arm:statistics.median(r['elapsed_ns'] for r in rows if r['phase']=='measured' and r['group']==group and r['arm']==arm)
             for arm in ('native','candidate')}
        groups.append(dict(group=group,median_ns=med,native_over_candidate=med['native']/med['candidate']))
    return dict(rows=rows,groups=groups,median_group_ratio=statistics.median(g['native_over_candidate'] for g in groups),
        timing_count=sum(r['phase']=='measured' for r in rows),warmup_count=sum(r['phase']=='warmup' for r in rows),
        expected_candidate_calls=1+sum(r['arm']=='candidate' for r in rows))


def checks(rows,labels,batch):
    expected=[]
    for label,kind in labels:
        expected.extend(dict(check=label,name='root/'+name,shape=shape,dtype=dtype,
                             reference_finite=True,candidate_finite=True,bit_equal=True)
            for name,shape,dtype in ((kind,[batch,5120],'mlx.core.bfloat16'),('conv',[batch,3,10240],'mlx.core.bfloat16'),
                                     ('S',[batch,48,128,128],'mlx.core.float32')))
    need(len({(r['check'],r['name']) for r in rows})==len(rows),'duplicate producer comparison receipt')
    order=lambda r:(r['check'],r['name'])
    same(sorted(rows,key=order),sorted(expected,key=order),'exact producer comparison receipts; no missing array checks')


def parameters(rows):
    need(set(rows)=={'0','32','62'},'three actual parameter sets')
    cleaned={}; layout=raw.parameter_layout()
    for layer,values in rows.items():
        need(set(values)==set(layout),'all fourteen parameters per layer'); cleaned[layer]={}
        for name,row in values.items():
            shape,dtype,width=layout[name]
            need(row['shape']==shape and row['dtype']==dtype and row['bytes']==math.prod(shape)*width
                 and row['finite'] is True and type(row['sha256']) is str and len(row['sha256'])==64
                 and all(c in '0123456789abcdef' for c in row['sha256']),'original parameter receipt shape/dtype/hash')
            cleaned[layer][name]={k:v for k,v in row.items() if k!='handle_id'}
    return cleaned


def audit(child,parent_path,parent_sha):
    started=time.monotonic(); child=Path(child).resolve(); parent_path=Path(parent_path).resolve()
    result=dict(schema='GDN-batched-local-timing-independent-v1',status='INCOMPLETE',child_directory=str(child),
        audit_source_sha256=sha(__file__),raw_reader_sha256=READER_SHA,parent_plan_sha256=parent_sha,
        case_count=0,timing_count=0,warmup_count=0,group_count=0,raw_files=0,full_output_state_pairs=0,
        expected_candidate_calls=0,reported_candidate_calls=0,reported_fallback_calls=0,cases=[],
        local_timing_screen_audited=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False,
        scope='Offline original bytes + original timing rows. No inference rerun, live GPU timing, service/resource/restoration audit or engine frontier claim.',
        dispatch_scope='Expected counts recomputed from one qualified call plus raw warmup/measured rows; actual per-call execution/fallback checks remain pinned producer evidence.',
        parameter_scope='Original-byte producer hash receipts compared; parameter payloads not stored/read by this auditor.')
    try:
        need(parent_sha==PARENT_SHA==sha(parent_path),'actual executed parent plan')
        parent=read(parent_path); need(parent['schema']=='E048-gdn-batched-local-timing-plan-v1'
            and parent['outputs']['numerical']==str(child),'exact parent/child directory')
        need(parent['pins'][str(HERE/'screen.py')]==SCREEN_SHA==sha(HERE/'screen.py'),'actual executed screen source')
        lease=Path(parent['outputs']['lease']); execution=Path(parent['outputs']['execution'])
        need((execution/'final.json').is_file() and read(lease/'lease.json')['closed'] is True,'closed owner and lease required before reading payloads')
        complete=read(child/'complete.json')
        need(complete['schema']=='GDN-batched-local-timing-result-v1' and complete['status']=='PASS','producer did not PASS')
        need(type(complete['pid']) is type(complete['ppid']) is int and complete['pid']>1 and complete['ppid']>1,'producer PID/PPID')
        for name in ('device_run','instance_hooks_restored','source_model_pins_rechecked','device_cleanup_completed','local_timing_screen_complete'):
            need(complete[name] is True,'producer terminal scope: '+name)
        for name in ('full_model_qualified','http_qualified','performance_qualified','engine_pareto_point'):
            need(complete[name] is False,'producer overclaimed scope')
        need(complete['source_sha256']==SCREEN_SHA and complete['plan_sha256']==CHILD_PLAN_SHA==sha(child/'plan.json'),'executed source and original child plan')
        need(type(complete['elapsed_wall_s']) in (int,float) and math.isfinite(complete['elapsed_wall_s']) and 0<complete['elapsed_wall_s']<=900,'producer elapsed bound')
        same(complete['runtime'],dict(mlx_version='0.32.3',name='Apple M5 Ultra',architecture='applegpu_g17d'),'actual runtime receipt')
        need(sha(ORIGINAL_AUDIT)==ORIGINAL_AUDIT_SHA and sha(ORIGINAL/'complete.json')==ORIGINAL_COMPLETE_SHA
             and sha(ORIGINAL/'captures.json')==CAPTURES_SHA,'fixed E046 evidence bytes')
        original_audit=read(ORIGINAL_AUDIT); original_complete=read(ORIGINAL/'complete.json'); captures=read(ORIGINAL/'captures.json')
        need(original_audit['status']=='PASS' and original_audit['raw_files']==140 and original_audit['canary_qualified'] is True
             and original_audit['complete_sha256']==ORIGINAL_COMPLETE_SHA and original_audit['captures_sha256']==CAPTURES_SHA,'original independent byte audit')
        ref=original_complete['parameters-before']; need(sha(ORIGINAL/ref['path'])==ref['sha256'],'E046 parameter receipt bytes')
        old_parameters=parameters(read(ORIGINAL/ref['path']))
        same(parameters(read(child/'parameters-before.json')),old_parameters,'selected parameters equal E046')
        same(parameters(read(child/'parameters-after.json')),old_parameters,'selected parameters remain equal E046')
        seen=set(); original_seen=set(); total_ns=0; expected_ids=[]; summary_rows=[]
        need([r['case_id'] for r in captures]==['b2-early','b2-late','b8-early','b8-late'],'four exact original chains')
        for chain in captures:
            batch=chain['batch_size']; start=chain['start']; need(batch in (2,8) and start in (1,129),'actual B2/B8 start')
            for layer in (0,32,62):
                cid=chain['case_id']+f'-layer{layer}'; expected_ids.append(cid); case=local(child,cid)
                source=chain['steps'][0][str(layer)]
                need(source['layer']==layer and source['batch_size']==batch and source['decode_step']==start
                     and source['row_ids']==chain['row_ids'] and len(source['row_ids'])==batch,'first actual capture identity')
                input_spec={'root/x':([batch,5120],{'BF16'}),'root/conv':([batch,3,10240],{'BF16'}),'root/S':([batch,48,128,128],{'F32'})}
                output_spec={'root/output':([batch,5120],{'BF16'}),'root/conv':([batch,3,10240],{'BF16'}),'root/S':([batch,48,128,128],{'F32'})}
                # Recheck both actual selected input and output files, including raw finiteness.
                raw.raw_file(ORIGINAL,source['reference_input'],input_spec,original_seen)
                reference=raw.raw_file(ORIGINAL,source['reference_output'],output_spec,original_seen)
                numerical=read(case/'numerical.json'); need(numerical['status']=='PASS','local numerical gate did not PASS')
                same(numerical['input'],source['reference_input'],'original input reference')
                same(numerical['reference'],source['reference_output'],'original output reference')
                need(set(numerical['saved'])=={'native','candidate'},'both actual arms saved')
                case_seen=set()
                for arm in ('native','candidate'):
                    desc=numerical['saved'][arm]; need(desc['path']==arm+'-qualified.safetensors','exact arm payload path')
                    actual=raw.raw_file(case,desc,output_spec,case_seen)
                    for name in output_spec:
                        raw.pair(reference[name],actual[name],cid+'/'+arm+'/'+name); result['full_output_state_pairs']+=1
                    seen.add(str(case/desc['path'])); del actual
                del reference
                checks(numerical['checks'],[(arm+'/'+name,kind) for arm in ('native','candidate')
                    for name,kind in (('original-output','output'),('seed-preserved','x'))],batch)
                computed=ledger(case/'timings.jsonl'); saved=read(case/'result.json')
                same({k:saved[k] for k in ('case_id','status','batch_size','layer','decode_step','row_ids','timing_count')},
                     dict(case_id=cid,status='PASS',batch_size=batch,layer=layer,decode_step=start,row_ids=source['row_ids'],timing_count=100),'actual local case identity/count')
                same(saved['groups'],computed['groups'],'raw-recomputed group medians/ratios')
                same(saved['median_group_ratio'],computed['median_group_ratio'],'raw-recomputed case ratio')
                same(saved['dispatch'],dict(candidate=computed['expected_candidate_calls'],fallback=0),'expected dispatch counts/source-bound actual receipt')
                checks(saved['final_checks'],[(arm+'/final-seed-preserved','x') for arm in ('native','candidate')],batch)
                result['case_count']+=1; result['timing_count']+=computed['timing_count']; result['warmup_count']+=computed['warmup_count']
                result['group_count']+=len(computed['groups']); result['expected_candidate_calls']+=computed['expected_candidate_calls']
                result['reported_candidate_calls']+=saved['dispatch']['candidate']; result['reported_fallback_calls']+=saved['dispatch']['fallback']
                total_ns+=sum(r['elapsed_ns'] for r in computed['rows']); summary_rows.append(saved)
                result['cases'].append(dict(case_id=cid,batch_size=batch,layer=layer,decode_step=start,
                    timing_sha256=sha(case/'timings.jsonl'),groups=computed['groups'],median_group_ratio=computed['median_group_ratio'],
                    numerical_sha256=sha(case/'numerical.json'),result_sha256=sha(case/'result.json')))
        need(sorted(p.name for p in child.iterdir() if p.is_dir())==sorted(expected_ids),'no extra or missing local cases')
        same(complete['cases'],summary_rows,'complete per-case references agree after independent recomputation')
        for name,expected in (('case_count',12),('numerical_case_count',12),('timing_count',1200),('candidate_calls',672),('candidate_fallback_calls',0)):
            same(complete[name],expected,'exact complete count: '+name)
        need(result['case_count']==12 and result['timing_count']==1200 and result['warmup_count']==120 and result['group_count']==60
             and result['expected_candidate_calls']==result['reported_candidate_calls']==672 and result['reported_fallback_calls']==0
             and result['full_output_state_pairs']==72,'independently counted coverage')
        need(total_ns<=complete['elapsed_wall_s']*10**9,'retained nonoverlapping call intervals exceed whole run')
        result.update(raw_files=len(seen)+len(original_seen),new_raw_files=len(seen),original_raw_files=len(original_seen),
            complete_sha256=sha(child/'complete.json'),total_retained_call_ns=total_ns,local_timing_screen_audited=True,status='PASS')
        need(result['raw_files']==48,'all24 new plus24 original input/output files independently read')
    except BaseException as error:
        result.update(status='FAIL',local_timing_screen_audited=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    result['elapsed_wall_s']=time.monotonic()-started
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--child',type=Path); p.add_argument('--parent-plan',type=Path)
    p.add_argument('--parent-plan-sha256'); p.add_argument('--out',type=Path); a=p.parse_args(argv)
    if a.child is a.parent_plan is a.parent_plan_sha256 is a.out is None:
        print('DESCRIPTION ONLY: after closure, audit12 cases/1200 durations/60 groups and48 original-byte files.'); return 0
    need(a.child and a.parent_plan and a.parent_plan_sha256 and a.out and not a.out.exists(),'explicit fresh audit output')
    result=audit(a.child,a.parent_plan,a.parent_plan_sha256)
    with a.out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','case_count','timing_count','raw_files','full_output_state_pairs')}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
