"""Default-OFF E060 saved generation audit. No model/backend/service execution.

Uses the source-pinned E058 raw parser; independently reconstructs selection,
selection and timing decisions rather than importing the producer helpers.
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

REPO=Path('/Users/ryanli/code/minivllm')
PRODUCER=REPO/'bench/m5_ultra_256gb/e060_short_verify_v1'
RAW=REPO/'bench/m5_ultra_256gb/e058_agent_audit_v2/audit.py'
RAW_SHA='ce9273057d6308411cfedb72e4dfd95d59f759f8dce09556153fc416a1fac44b'
ARMS=('native','fixed2','fixed4','fixed8')
CASES=('math','chinese')
QUAL_ARMS=('fixed2','fixed4','fixed8','stress')
VOCAB=248320

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def unique(pairs):
    d={}
    for k,v in pairs:
        need(k not in d,'duplicate JSON key '+k);d[k]=v
    return d
def read(path):
    def bad(v):raise ValueError('nonfinite JSON '+v)
    p=Path(path);need(p.is_file() and not p.is_symlink() and p.stat().st_size<16<<20,'bounded regular JSON '+str(p))
    return json.loads(p.read_bytes(),object_pairs_hook=unique,parse_constant=bad)
def same(a,b,message):need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)
def integer(x,lo=0,hi=None):return type(x) is int and x>=lo and (hi is None or x<=hi)
def positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0
def digest(x):return type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def ids(values,lo=1,hi=32):
    need(type(values) is list and lo<=len(values)<=hi and all(integer(t,0,VOCAB-1) for t in values),'complete legal token IDs')
    return values

def parser():
    need(sha(RAW)==RAW_SHA,'frozen raw parser SHA')
    spec=importlib.util.spec_from_file_location('_e060_frozen_e058_reader',RAW)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def stamp(path):
    s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def labels():return [case+'-'+arm for case in CASES for arm in QUAL_ARMS]
def array_spec(raw,rows):
    need(integer(rows,1,31),'final computed rows')
    spec=raw.expected_arrays(4,1,'continuation')
    spec['features']=('BF16',(rows,25600))
    return spec

def select_independent(proposals,posterior,remaining,eos):
    """Full prefix match first, then cap/EOS. Final emitted token is pending."""
    matched=next((j for j,t in enumerate(proposals) if t!=posterior[j]),len(proposals))
    produced=(proposals[:matched]+[posterior[matched]])[:remaining]
    stop=next((i for i,t in enumerate(produced) if t in eos),None)
    if stop is not None:produced=produced[:stop+1]
    return matched,produced

def trajectory(rows,output,arm,eos):
    ids(output,2,32);need(not set(output[:-1])&eos,'EOS only at trajectory end')
    need(len(output)==32 or output[-1] in eos,'complete capped or EOS generation')
    if arm=='native':need(rows==[],'native has no speculative rounds');return 0
    need(type(rows) is list and rows,'complete speculative rounds')
    emitted=[output[0]];elapsed=0
    fields={'width','accepted','delivered','draft_ns','verify_commit_ns','draft_ids','posterior','produced','position_after'}
    for row in rows:
        need(set(row)==fields,'exact original round fields')
        need(len(emitted)<32 and emitted[-1] not in eos,'no rounds after termination')
        width=row['width'];need(integer(width) and width in (2,4,8),'admitted verification width')
        need(arm in ('fixed2','fixed4','fixed8'),'fixed verification arm')
        need(width==int(arm[-1]),'fixed verification width')
        draft=ids(row['draft_ids'],7,7);posterior=ids(row['posterior'],width,width)
        accepted,produced=select_independent(draft[:width-1],posterior,32-len(emitted),eos)
        need(integer(row['accepted']) and row['accepted']==accepted,'full raw accepted prefix')
        same(row['produced'],produced,'actual EOS/cap selected tokens')
        need(integer(row['delivered'],1,width) and row['delivered']==len(produced),'delivered/computed prefix count')
        emitted+=produced
        need(row['position_after']==1024+len(emitted)-1 and integer(row['position_after']),'pending-anchor position accounting')
        for key in ('draft_ns','verify_commit_ns'):
            need(integer(row[key],1),'positive original phase timing');elapsed+=row[key]
    same(emitted,output,'round ledger reconstructs entire generated stream')
    return elapsed

def timing_check(rows,outputs,eos):
    schedule=[]
    for group in range(3):
        for case in range(2):
            shift=(group+case)%4
            schedule.extend((group,case,arm) for arm in ARMS[shift:]+ARMS[:shift])
    need(type(rows) is list and len(rows)==24,'all24 generation timings')
    same([(r.get('group'),r.get('case'),r.get('arm')) for r in rows],schedule,'exact rotated24 schedule')
    fields={'group','case','arm','elapsed_ns','output_ids','decoded_tokens','rounds'}
    for r in rows:
        need(set(r)==fields and integer(r['group'],0,2) and integer(r['case'],0,1),'exact timing record')
        same(r['output_ids'],outputs[CASES[r['case']]],'timed IDs equal byte-qualified trajectory')
        count=len(r['output_ids'])-1
        need(integer(r['elapsed_ns'],1) and integer(r['decoded_tokens'],1,31) and r['decoded_tokens']==count,'timed generated-token denominator excludes initial anchor')
        phases=trajectory(r['rounds'],r['output_ids'],r['arm'],eos)
        need(phases<=r['elapsed_ns'],'phase intervals bounded by entire generation')
    summaries=[]
    baseline=sum(r['elapsed_ns'] for r in rows if r['arm']=='native')/sum(r['decoded_tokens'] for r in rows if r['arm']=='native')
    for arm in ARMS:
        samples=[r for r in rows if r['arm']==arm];total=sum(r['elapsed_ns'] for r in samples);tokens=sum(r['decoded_tokens'] for r in samples)
        groups=[]
        for group in range(3):
            a=[r for r in samples if r['group']==group];n=[r for r in rows if r['arm']=='native' and r['group']==group]
            groups.append(dict(group=group,seconds=sum(r['elapsed_ns'] for r in a)/1e9,decoded_tokens=sum(r['decoded_tokens'] for r in a),
                native_over_arm=(sum(r['elapsed_ns'] for r in n)/sum(r['decoded_tokens'] for r in n))/(sum(r['elapsed_ns'] for r in a)/sum(r['decoded_tokens'] for r in a))))
        summaries.append(dict(arm=arm,samples=6,total_ns=total,decoded_tokens=tokens,ns_per_token=total/tokens,
            tokens_per_second=tokens/(total/1e9),native_over_arm=baseline/(total/tokens),groups=groups))
    return summaries,sum(r['elapsed_ns'] for r in rows)

def raw_pairs(raw,root,manifest,result,file_stamps=None):
    file_stamps={} if file_stamps is None else file_stamps
    need(set(manifest)=={'schema','pairs'} and manifest['schema']=='E060-generation-pairs-v1','raw manifest schema')
    same([r.get('label') for r in manifest['pairs']],labels(),'exact eight ordered pair labels')
    outputs={};seen=set()
    for pair in manifest['pairs']:
        label=pair['label'];need(set(pair)=={'label','output_ids','computed_rows','reference','candidate'},'exact pair record')
        token_ids=ids(pair['output_ids'],2,32);rows=pair['computed_rows'];need(integer(rows,1,31) and rows==len(token_ids)-1,'final computed count and pending token')
        case=label.split('-')[0]
        if case in outputs:same(token_ids,outputs[case],'all qualified arms same original native trajectory')
        else:outputs[case]=token_ids
        spec=array_spec(raw,rows);parsed={};pair_bytes=0
        for side in ('reference','candidate'):
            expected='raw/'+label+'-'+side+'.safetensors';need(pair[side]['path']==expected and expected not in seen,'distinct original raw file names')
            path=raw._resolve_member(str(root),expected);before=stamp(path)
            need(before[2]<1<<30,'bounded final tensor file')
            relative,arrays,size=raw._load_member(str(root),side,pair[side]);need(set(arrays)==set(spec),'all130 final original arrays')
            need(stamp(path)==before,'raw file changed while read');file_stamps[path]=before
            parsed[side]=arrays;seen.add(relative);pair_bytes+=size;result['raw_files']+=1
        for name in spec:raw.compare_arrays(spec,name,parsed['reference'][name],parsed['candidate'][name]);result['tensor_pairs']+=1
        # Same bytes were finite-checked by the frozen parser. Check full-vocab
        # greedy output, including stable first-index argmax ties/signed zeros.
        words=struct.unpack('<'+str(VOCAB)+'H',parsed['reference']['logits'][2])
        best=max(range(VOCAB),key=lambda i:struct.unpack('<f',struct.pack('<I',words[i]<<16))[0])
        need(best==token_ids[-1],'last full-vocabulary logits predict final pending token')
        result['pairs'].append(dict(label=label,arrays=130,bytes=pair_bytes,computed_rows=rows))
        result['raw_bytes']+=pair_bytes
        del parsed
    return outputs,seen

def seed_check(raw,root,rows,remember,file_stamps=None):
    file_stamps={} if file_stamps is None else file_stamps
    names=[('math-before','math-after-qualification'),
           ('chinese-before','chinese-after-qualification'),('math-before','test0-after-timing'),('chinese-before','test1-after-timing')]
    need(type(rows) is list and len(rows)==4,'four original seed checks')
    seen=set();total=0
    for row,(before,after) in zip(rows,names):
        need(set(row)=={'before','after'},'seed check fields')
        for side,label in (('before',before),('after',after)):
            ref=row[side];need(set(ref)=={'path','sha256'} and ref['path']=='seed-'+label+'.safetensors' and digest(ref['sha256']),'exact seed receipt')
            path=Path(raw._resolve_member(str(root),ref['path']))
            if str(path) not in remember:
                before=stamp(path);need(before[2]<1<<30,'bounded seed tensor file')
                remember[str(path)]=sha(path);total+=before[2]
                need(stamp(path)==before,'seed file changed while read');file_stamps[str(path)]=before
            need(remember[str(path)]==ref['sha256'],'actual original seed file SHA')
            seen.add(ref['path'])
        need(row['before']['sha256']==row['after']['sha256'],'seed before/after identical file bytes')
    return seen,total

def parameter_values(rows):
    need(type(rows) is dict and rows,'nonempty parameter receipt')
    values={}
    for k,v in rows.items():
        need(type(k) is str and type(v) is dict and set(v)=={'shape','dtype','bytes','sha256','finite','handle_id'},'original parameter receipt fields')
        need(type(v['shape']) is list and all(integer(n,0) for n in v['shape']) and type(v['dtype']) is str and integer(v['bytes']) and digest(v['sha256']) and v['finite'] is True and integer(v['handle_id'],1),'original finite parameter hash receipt')
        values[k]={f:v[f] for f in ('shape','dtype','bytes','sha256','finite')}
    return values

def audit(root,plan_path,plan_sha):
    root=Path(root).resolve();plan_path=Path(plan_path).resolve();started=time.monotonic();remember={};seed_hashes={};file_stamps={}
    result=dict(schema='E060-independent-generation-audit-v1',status='INCOMPLETE',audit_source_sha256=sha(__file__),raw_parser_sha256=RAW_SHA,
        plan_sha256=plan_sha,raw_files=0,tensor_pairs=0,raw_bytes=0,pairs=[],generation_bytes_audited=False,
        http_qualified=False,performance_qualified=False,engine_pareto_point=False,
        scope='Closed saved local greedy generation.16 final snapshots/1040 tensor pairs; no per-round raw state or24 full timed-state comparisons. Prefill excluded. Same full8 drafter; only verification width changes.',
        limits=['Seed files are rehashed and paired for equality; producer full-array comparison is source-bound. They are not counted as final raw tensor pairs.',
                'No calibration or adaptive policy. These two prompts were observed in E059; this is exploratory follow-up, not held-out evidence.',
                'No model/runtime payload rehash, GPU execution or live restoration/resource probe. Owner closure is audited separately.',
                'Only two synthetic test contexts, three samples/arm/context. Native uses the strong no-capture decode_model runner through a host-driven loop, without HTTP lookahead. No serving, HTTP, global engine winner or Pareto qualification.'])
    def get(p):
        p=Path(p);value=read(p);remember[str(p)]=sha(p);return value
    try:
        closed_path=root.parent/(root.name.removesuffix('-numerical')+'-execution')/'final.json'
        closed=get(closed_path);need(closed['status']=='PASS_CAPTURE_AND_RESTORE','actual execution closed and restored before reading raw evidence')
        need(digest(plan_sha) and sha(plan_path)==plan_sha,'explicit registered child plan SHA')
        plan=get(plan_path);same(get(root/'plan.json'),plan,'retained exact child plan')
        need(plan['schema']=='E060-generation-plan-v1' and plan['deadline_s']==1500 and plan['output_cap']==32 and plan['groups']==3 and plan['warmups']==1,'fixed experiment contract')
        same([p['label'] for p in plan['prompts']],list(CASES),'exact two previously observed test labels')
        need(len({(p['context'],p['question']) for p in plan['prompts']})==2,'distinct planned prompt texts')
        required=[PRODUCER/n for n in ('screen.py','selection.py')]
        need(all(str(p) in plan['sources'] for p in required),'complete producer source pins')
        for name,value in plan['sources'].items():
            path=Path(name);need(path.is_absolute() and path.is_file() and path.stat().st_size<16<<20 and digest(value) and sha(path)==value,'current preregistered source SHA '+name);remember[name]=value
        own=str(Path(__file__).resolve());remember[own]=result['audit_source_sha256']
        # Independent post-run audit source is frozen separately, not inserted
        # into or presented as part of the already frozen producer plan.
        prior={}
        for key in ('e058_plan','e058_result','e058_audit','e059_result'):
            ref=plan[key];need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute() and digest(ref['sha256']),'exact prerequisite reference')
            prior[key]=get(ref['path']);need(remember[ref['path']]==ref['sha256'],'actual prerequisite SHA')
        need(prior['e058_result']['status']=='PASS_FEATURES_LOCAL_COST' and prior['e058_result']['plan_sha256']==plan['e058_plan']['sha256'],'same qualified E058 plan/result')
        need(prior['e058_audit']['status']=='PASS' and prior['e058_audit']['schema']=='E058-feature-pairs-v1' and prior['e058_audit']['pairs_checked']==24 and prior['e058_audit']['arrays_checked']==3120,'actual raw E058 prerequisite')
        need(prior['e059_result']['schema']=='E059-generation-result-v1' and prior['e059_result']['status']=='PASS_GENERATION_LOCAL_COST','actual closed E059 predecessor')
        for name,value in prior['e058_plan']['sources'].items():need(sha(name)==value,'frozen E058 source');remember[name]=value
        complete=get(root/'complete.json');need(complete['schema']=='E060-generation-result-v1' and complete['status']=='PASS_GENERATION_LOCAL_COST' and complete['plan_sha256']==plan_sha,'actual producer PASS with same plan')
        need(integer(complete['pid'],1) and integer(complete['ppid'],1) and not complete.get('error'),'completed producer identity/no error')
        for key in ('numerical_screen_qualified','pre_timing_parameters_unchanged','canonical_unchanged','device_cleanup_completed','source_model_pins_rechecked'):need(complete[key] is True,'producer gate '+key)
        for key in ('http_qualified','performance_qualified','engine_pareto_point'):need(complete[key] is False,'excluded claim '+key)
        same(complete['device'],dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx='0.32.3'),'exact device receipt')
        same(complete['native_runner_qualified'],list(CASES),'actual no-capture runner qualification receipts')
        eos=ids(complete['eos_ids'],1,64);need(eos==sorted(set(eos)),'sorted distinct actual engine EOS IDs');eos=set(eos)
        prompts=complete['test_prompt_ids']
        need('calibration' not in complete and 'calibration_prompt_ids' not in complete and not (root/'calibration.json').exists(),'no calibration/adaptive arm in E060')
        need(len(prompts)==2,'two actual prompts')
        for prompt in prompts:ids(prompt,1024,1024)
        need(len({tuple(p) for p in prompts})==2,'distinct actual test token contexts')
        same(prompts,prior['e059_result']['test_prompt_ids'],'exact same previously observed E059 token contexts')
        ref=complete['raw_manifest'];need(ref['path']=='pairs.json' and set(ref)=={'path','sha256'} and digest(ref['sha256']),'raw manifest reference')
        manifest=get(root/ref['path']);need(remember[str(root/ref['path'])]==ref['sha256'],'raw manifest SHA')
        need(complete['completed_pairs']==8,'all eight final pairs')
        raw=parser();remember[str(RAW)]=RAW_SHA
        outputs,raw_seen=raw_pairs(raw,root,manifest,result,file_stamps)
        for output in outputs.values():ids(output,2,32);need(not set(output[:-1])&eos and (len(output)==32 or output[-1] in eos),'qualified EOS/cap ending')
        seed_seen,seed_bytes=seed_check(raw,root,complete['seed_checks'],seed_hashes,file_stamps)
        need({str(p.relative_to(root)) for p in root.rglob('*.safetensors')}==raw_seen|seed_seen,'exact raw/seed snapshot inventory')
        same(get(root/'timings-partial.json'),complete['timings'],'all incremental timing records retained')
        summaries,total_ns=timing_check(complete['timings'],outputs,eos)
        need(positive(complete['elapsed_s']) and complete['elapsed_s']<=plan['deadline_s'] and total_ns<=complete['elapsed_s']*1e9,'timings within complete bounded elapsed')
        values=[]
        for name in ('parameters-before.json','parameters-before-timing.json','parameters-after.json'):values.append(parameter_values(get(root/name)))
        need(len(values[0])==1349,'all original target parameters');same(values[0],values[1],'target parameters unchanged before timing');same(values[0],values[2],'target parameters unchanged after timing')
        before=parameter_values(get(root/'drafter-before.json'));after=parameter_values(get(root/'drafter-after.json'));same(before,after,'drafter parameter bytes unchanged')
        memory=complete['memory_after_cleanup'];need(set(memory)=={'active','peak','cache'} and all(integer(v) for v in memory.values()) and memory['peak']>=memory['active'],'original memory cleanup receipt')
        need(complete['independent_raw_audit_pending'] is True,'producer independent audit pending receipt')
        need(all(sha(p)==value for p,value in remember.items()),'source/small evidence changed during audit')
        need(all(stamp(p)==value for p,value in file_stamps.items()),'saved tensor files changed during audit')
        result.update(status='PASS',generation_bytes_audited=True,pairs_checked=8,arrays_checked=1040,timing_count=24,seed_checks=4,seed_files_hashed=len(seed_seen),seed_bytes_hashed=seed_bytes,
            target_parameter_count=1349,drafter_parameter_count=len(before),timing_summary=summaries,total_timed_ns=total_ns,
            complete_sha256=remember[str(root/'complete.json')],closed_execution_sha256=remember[str(closed_path)],source_and_small_evidence=remember,seed_file_hashes=seed_hashes,snapshot_file_stats=file_stamps)
    except BaseException as e:result.update(status='FAIL',generation_bytes_audited=False,error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
    result['audit_elapsed_s']=time.monotonic()-started;return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--root',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256');p.add_argument('--out',type=Path);a=p.parse_args(argv)
    if not a.run:
        need(not any((a.root,a.plan,a.plan_sha256,a.out)),'explicit --run required');print(json.dumps(dict(enabled_by_default=False,raw_pairs=8,tensor_pairs=1040,timings=24,scope='local greedy generation only')));return 0
    need(all((a.root,a.plan,a.plan_sha256,a.out)) and not a.out.exists(),'complete arguments and fresh output')
    report=audit(a.root,a.plan,a.plan_sha256)
    with a.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','raw_files','tensor_pairs','audit_elapsed_s')}));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
