"""Independent closed GDN37 serving receipt audit; stdlib, no GPU/model imports.

The fixed E023 CaseAudit retains every schedule/logit/state/ahead/normal-EOS
check. Its original stream function is rebound to an isolated globals mapping
with one explicit CaseAudit dependency; no imported module globals are changed.
Added logic copies the independently drained C8 slot stack into lifecycle and
joins each actual candidate model invocation to its dispatch row. Consumed
prior-ahead calls are not executed twice. Successful tensor payloads are not
stored by this producer: this audits source-pinned device bit/finite receipts,
not an offline recomputation. Resource/reap/restoration remain separate.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'pareto_realistic/audit_qualification.py'
BASE_SHA='b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
NUMERIC_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SERVING_SHA='3f493752d95996bd08cc1ea7389fc3159ab615e1ac162b36332eecc47dab862c'
QUALIFIER_SHA='26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'
CANDIDATE_SHA='1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260'
CANARY_SHA='f3088a98d2aca6b4f267e7d4c258efbbf9f83e1beccb6745c88ae42343f88e89'
CANARY_AUDIT_SHA='a3e97dc368da47735d7c8e9fd2bcb40d7decaccb9c5926084d53cffb10495d3d'
FIXED={str(HERE/'serving.py'):SERVING_SHA,str(HERE/'candidate.py'):CANDIDATE_SHA,
       str(HERE/'qualify.py'):CANARY_SHA,str(HERE/'audit.py'):CANARY_AUDIT_SHA,
       str(HERE.parent/'pareto_realistic/qualify_native.py'):QUALIFIER_SHA}
CASES=('D-C1','D-C8','lifecycle')

class Incomplete(Exception):pass
def need(ok,message):
    if not ok:raise ValueError(message)
def same(a,b,message):
    need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)
def read(path):
    need(Path(path).suffix!='.jsonl','JSONL must be streamed')
    return json.loads(Path(path).read_bytes())
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()
def pinned(ref):
    need(type(ref) is dict and set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute(),'exact absolute pinned ref')
    need(sha(ref['path'])==ref['sha256'],'pinned file changed: '+ref['path']);return read(ref['path'])
def base_auditor():
    need(sha(BASE)==BASE_SHA,'fixed original numerical auditor')
    spec=importlib.util.spec_from_file_location('_gdn_serving_independent_base',BASE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def stream_with(base,case_type,*args,**kwargs):
    """Same frozen stream code/defaults, explicit local CaseAudit injection."""
    fn=base.stream_ledger
    need(type(fn) is FunctionType and fn.__closure__ is None and fn.__globals__ is vars(base),'original stream callable')
    namespace=dict(fn.__globals__);namespace['CaseAudit']=case_type
    bound=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__)
    bound.__kwdefaults__=fn.__kwdefaults__
    return bound(*args,**kwargs)

class Dispatch:
    """One forward cursor, matched against independently reconstructed calls."""
    def __init__(self,stream):
        self.stream=stream;self.digest=hashlib.sha256();self.bytes=self.lines=0;self.per_case={}
    def call(self,case,step,key):
        need(case in CASES and type(step) is int and step>0,'actual case/step')
        mode=key[0]
        if mode=='decode':
            B=len(key[1]);used=max(key[2])+1
            need(B>0 and len(key[2])==len(key[3])==B,'decode semantic geometry')
            path='short-packed-bypass' if used<=256 else 'B1-candidate' if B==1 else 'B>1-fallback'
        else:
            need(mode=='prefill' and type(key[3]) is int and key[3]>0,'ordinary prefill semantic key')
            B=key[3];used=0;path='prefill-untouched'
        candidate=48 if path=='B1-candidate' else 0;fallback=48 if path=='B>1-fallback' else 0
        expected=dict(event='prework-model-call',status='SUBMITTED',case=case,step=step,mode=mode,
                      B=B,used=used,semantic_key=key,path=path,candidate=candidate,fallback=fallback)
        raw=self.stream.readline();need(raw and raw.endswith(b'\n'),'missing/truncated candidate dispatch')
        self.digest.update(raw);self.bytes+=len(raw);self.lines+=1
        same(json.loads(raw),expected,'actual candidate dispatch/key/domain/count mismatch at line '+str(self.lines))
        counts=self.per_case.setdefault(case,dict(candidate=0,fallback=0,b1_calls=0,multirow_calls=0,prefill_calls=0,short_bypass_calls=0))
        counts['candidate']+=candidate;counts['fallback']+=fallback
        counts[{'B1-candidate':'b1_calls','B>1-fallback':'multirow_calls','prefill-untouched':'prefill_calls',
                'short-packed-bypass':'short_bypass_calls'}[path]]+=1
    def finish(self):
        need(self.stream.read(1)==b'','extra/unmatched candidate dispatch')
        return dict(sha256=self.digest.hexdigest(),bytes=self.bytes,lines=self.lines,per_case=self.per_case,
                    scope='Actual evaluated candidate model-call submissions; not profiler Metal kernel completion counts.')
    def expected_coverage(self):
        need(set(self.per_case)==set(CASES),'all original cases dispatched')
        rows=self.per_case
        need(rows['D-C1']['b1_calls']>0 and rows['D-C1']['multirow_calls']==0 and rows['D-C8']['multirow_calls']>0,
             'actual long B1 hit / C8 multirow fallback coverage')
        def group(names):
            # Every long call executes all48 GDN instances once; source-pinned
            # dispatch_delta verifies equal per-layer deltas before aggregation.
            totals={k:sum(rows[name][k] for name in names)//48 for k in ('candidate','fallback')}
            return [dict(totals) for _ in range(48)]
        return dict(enabled=True,layer_installs=96,layer_restores=96,observer_restores=4,calls=self.lines,
                    per_case=rows,final_counters=[group(('D-C1',)),group(('D-C8','lifecycle'))],
                    identity_refill_enabled=False,route_graph_enabled=False)

def extended_case(base,dispatch):
    previous=None;original=base.CaseAudit
    class PreworkCase(original):
        def __init__(self,*args,**kwargs):
            nonlocal previous
            super().__init__(*args,**kwargs)
            if self.name=='lifecycle':
                need(previous is not None and previous.name=='D-C8' and previous.C==self.C==8,'actual reused C8 predecessor')
                need(previous.next==len(previous.jobs) and not previous.requests and not previous.waiting
                     and not previous.running and not previous.staging and previous.previous_ahead is None
                     and not previous.previous_arrays and previous.previous_batch==0,'fully drained reused C8')
                need(all(type(x) is int for x in previous.free) and sorted(previous.free)==list(range(self.C)),
                     'complete previous free-slot permutation')
                base.same(self.finished,previous.finished,'actual pending retirement order')
                self.free=list(previous.free)
            previous=self
        def check_arrays(self,row,calls,pre_staging,has_future,B):
            # Parent accept has already reconstructed/verified call keys,
            # consumed/discarded ahead, schedule, output retirement and arrays.
            super().check_arrays(row,calls,pre_staging,has_future,B)
            consumed=row['ahead']['candidate']['consumed_prior_future']
            for key in calls:
                if key!=consumed:dispatch.call(self.name,self.step,key)
    return PreworkCase

def canary_binding(refs):
    need(type(refs) is dict and set(refs)=={'audit','plan','complete'},'canary exact three refs')
    audit,plan,complete=(pinned(refs[k]) for k in ('audit','plan','complete'))
    child=Path(refs['complete']['path']).parent.resolve()
    need(Path(refs['complete']['path'])==child/'complete.json' and Path(refs['plan']['path'])==child/'plan.json'
         and audit['child_directory']==str(child),'same actual canary child')
    need(audit['schema']=='GDN-prework-real-layer-independent-v1' and audit['status']=='PASS'
         and audit['canary_qualified'] is True and audit['audit_source_sha256']==CANARY_AUDIT_SHA,'independent canary PASS')
    for key,value in dict(raw_files=60,full_output_state_pairs=54,causal_state_pairs=24,
                          full_vocabulary_greedy_checks=6,device_comparison_arrays=162).items():same(audit[key],value,'canary coverage '+key)
    need(complete['schema']=='GDN-prework-real-layer-result-v1' and complete['status']=='PASS' and complete['canary_qualified'] is True,'actual canary complete')
    for key,value in dict(candidate_calls=18,candidate_compared_calls=18,reference_chains=2,candidate_chains=6,
                          device_run=True,instance_hooks_restored=True,source_model_pins_rechecked=True).items():same(complete[key],value,'canary complete '+key)
    same(complete['dispatch_counters'],{str(i):dict(candidate=6,fallback=0) for i in (0,32,62)},'actual canary dispatch')
    for value in (audit,complete):
        for key in ('full_model_qualified','http_qualified','performance_qualified'):need(value[key] is False,'limited canary scope')
    need(audit['plan_sha256']==complete['plan_sha256']==refs['plan']['sha256'] and audit['complete_sha256']==refs['complete']['sha256'],'canary terminal binding')
    need(plan['schema']=='GDN-prework-real-layer-plan-v1' and plan['qualifier_sha256']==CANARY_SHA,'canary producer source')
    same(plan['candidate'],dict(path=str(HERE/'candidate.py'),sha256=CANDIDATE_SHA),'same qualified candidate')
    for name,field in (('events.jsonl','ledger_sha256'),('captures.json','captures_sha256')):need(sha(child/name)==audit[field],'canary raw receipt '+name)
    same(audit['source_pins'],complete['sources'],'canary source identities')
    need(complete['sources'][str(HERE/'candidate.py')]==CANDIDATE_SHA and complete['sources'][str(HERE/'qualify.py')]==CANARY_SHA,'canary exact candidate/qualifier')
    inherited=pinned(plan['e024_plan'])
    need(inherited['e023_plan']['sha256']==NUMERIC_SHA and sha(inherited['e023_plan']['path'])==NUMERIC_SHA,'same canary model/workload plan')
    return dict(evidence=refs,child_directory=str(child),raw_files=60,full_output_state_pairs=54,candidate_calls=18,
                audit_source_sha256=CANARY_AUDIT_SHA,scope='Consumes SHA-pinned independent raw canary proof; does not rerun its raw audit.')

def binding(child,digest,base):
    plan,top,claimed=[read(child/p) for p in ('plan.json','complete.json','base/complete.json')]
    if any(x.get('status')=='INCOMPLETE' for x in (top,claimed)):raise Incomplete('numerical result not terminal')
    need(top['status']==claimed['status']=='PASS','producer numerical failure')
    need(sha(child/'plan.json')==digest==top['plan_sha256'],'registered serving plan SHA')
    need(set(plan)=={'schema','qualifier_sha256','qualification','canary','sources','deadline_s'}
         and plan['schema']=='GDN-prework-serving-plan-v1' and type(plan['deadline_s']) is int
         and plan['deadline_s']==1800,'exact serving plan/deadline')
    same(plan['sources'],FIXED,'exact serving/base/canary/candidate/auditor source set')
    need(plan['qualifier_sha256']==SERVING_SHA,'frozen serving source')
    for path,digest in FIXED.items():need(sha(path)==digest,'serving source changed '+path)
    same(top['sources'],FIXED,'terminal sources');same(top['canary'],canary_binding(plan['canary']),'independent prerequisite binding')
    need(top['schema']=='GDN-prework-serving-result-v1' and top['same_schedule_checks_passed'] is True
         and top['source_pins_rechecked'] is True and top['independent_audit_required'] is True,'serving terminal gates')
    for key in ('full_model_qualified','http_qualified','performance_qualified'):need(top[key] is False,'producer scope inflation')
    same(top['total_requests'],37,'original37')
    same(top['base_result'],dict(path='base/complete.json',sha256=sha(child/'base/complete.json'),status='PASS',returncode=0),'base terminal/returncode')
    q=plan['qualification'];numeric=pinned(q)
    need(q['sha256']==sha(child/'base/plan.json')==NUMERIC_SHA,'original numerical plan')
    need(claimed['plan_sha256']==NUMERIC_SHA and claimed['qualifier_sha256']==QUALIFIER_SHA
         and claimed['numeric_only_not_http'] is True and claimed['source_model_pins_rechecked'] is True,'original terminal/source recheck')
    same(claimed['total_requests'],37,'original total');same(claimed['candidate'],numeric['candidate'],'original candidate flags')
    same(numeric['candidate'],dict(compile_decode=True,fuse_decode=True,gates_compile=True,norm_reuse=False),'fixed flags')
    same(numeric['deadline_s'],1800,'original1800')
    manifests={field:pinned(numeric[field]) for field in ('fixtures','source_manifest','model_manifest')}
    root=Path(numeric['package_root'])/'tutorial_code/miniengine';core=manifests['source_manifest']['files']
    need(set(core)=={str(p.relative_to(root)) for p in root.rglob('*.py')},'complete original core set')
    for rel,digest in core.items():need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(root/rel)==digest,'original core source '+rel)
    model=manifests['model_manifest'];config_path=Path(numeric['model'])/'config.json'
    need(model['schema']=='E023-model-v1' and model['model']==numeric['model'] and model['revision']==Path(numeric['model']).name
         and sha(config_path)==model['files']['config.json'],'model/config identity')
    config=read(config_path);config=config.get('text_config',config)
    need(config['layer_types']==['full_attention' if i%4==3 else 'linear_attention' for i in range(64)]
         and config['vocab_size']==248320 and config['hidden_size']==5120,'actual64/48 model geometry')
    fixtures=manifests['fixtures'];need(numeric['fixtures']['sha256']==base.FIXTURE_SHA and fixtures['model']==numeric['model'],'fixture identity')
    fixtures=fixtures['fixtures'];need(len(fixtures)==8 and len({f['fixture_id'] for f in fixtures})==8,'eight fixed fixtures')
    eos=fixtures[0]['eos_token_ids']
    need(all(f['eos_token_ids']==eos and f['normal_eos'] is True and f['max_output_tokens']==512 for f in fixtures),'normal EOS/cap512')
    return plan,top,claimed,fixtures,config,eos

def audit(child_dir,expected_plan_sha256):
    child=Path(child_dir).resolve()
    result=dict(schema='GDN-prework-serving-independent-v1',status='INCOMPLETE',qualified=False,same_schedule_qualified=False,
      ledger_opened=False,http_qualified=False,performance_qualified=False,combined_candidates_qualified=False,
      parameter_replacement_qualified=False,owner_resources_restore_qualified=False,offline_tensor_bytes_rechecked=False,
      audit_source_sha256=sha(__file__),base_auditor_sha256=BASE_SHA,child_directory=str(child),
      scope='Original37 independent schedule/full-state/ahead/normal-EOS receipt gate plus actual48-layer prework dispatch accounting. Success tensors are source-pinned device bit/finite receipts, not offline arithmetic. No HTTP/performance/composition/parameter-replacement or lifecycle resource qualification.')
    try:
        for name in ('complete.json','plan.json','base/complete.json','base/plan.json','base/steps.jsonl','dispatch.jsonl'):
            if not (child/name).is_file():raise Incomplete('missing '+name)
        base=base_auditor();plan,top,claimed,fixtures,config,eos=binding(child,expected_plan_sha256,base)
        stable={name:sha(child/name) for name in ('plan.json','complete.json','base/complete.json','base/plan.json')}
        paths=[child/'base/steps.jsonl',child/'dispatch.jsonl'];before=[p.stat() for p in paths]
        with paths[1].open('rb') as stream:
            dispatch=Dispatch(stream);case=extended_case(base,dispatch);result['ledger_opened']=True
            ledger=stream_with(base,case,paths[0],fixtures,config,eos,claimed['cases'],partial=False)
            result['ledger']=ledger;result['dispatch']=dispatch.finish()
        need([len(c['completed']) for c in ledger['cases']]==[8,16,13] and not ledger['partial_cases']
             and ledger['truncated_final_line'] is None,'all37 complete untruncated requests')
        coverage=dispatch.expected_coverage();same(top['coverage'],coverage,'independently reconstructed all-layer/pair coverage')
        need(top['dispatch_sha256']==result['dispatch']['sha256'],'dispatch terminal digest')
        for path,st,size in zip(paths,before,(ledger['bytes'],dispatch.bytes)):
            after=path.stat();need((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
                                  and size==st.st_size,'closed evidence changed/truncated')
        need(all(sha(child/name)==digest for name,digest in stable.items()),'terminal changed during audit')
        need(all(sha(path)==digest for path,digest in FIXED.items()) and sha(BASE)==BASE_SHA,'audited sources changed')
        same(canary_binding(plan['canary']),top['canary'],'canary prerequisite changed during audit')
        result.update(status='PASS',qualified=True,same_schedule_qualified=True,plan_sha256=expected_plan_sha256,
                      base_plan_sha256=NUMERIC_SHA,terminal_sha256=stable,coverage=coverage,source_pins=FIXED,
                      observed_eos_requests=sum(bool(r['raw_ids']) and r['raw_ids'][-1] in eos for c in ledger['cases'] for r in c['completed']),
                      weight_shards_rehashed=False,uncovered='Unobserved cancellation/sampling-key invalidation, parameter replacement, full payload recomputation, HTTP and any E024/E028 combination remain unqualified.')
    except (Incomplete,FileNotFoundError) as exc:result.update(status='INCOMPLETE',reason=str(exc))
    except Exception as exc:result.update(status='FAIL',first_failure=dict(type=type(exc).__name__,message=str(exc)))
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--child',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args(argv)
    need(args.out.parent.is_dir() and not args.out.exists(),'new output with existing parent required')
    result=audit(args.child,args.plan_sha256)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],qualified=result['qualified'])))
    return int(not result['qualified'])
if __name__=='__main__':raise SystemExit(main())
