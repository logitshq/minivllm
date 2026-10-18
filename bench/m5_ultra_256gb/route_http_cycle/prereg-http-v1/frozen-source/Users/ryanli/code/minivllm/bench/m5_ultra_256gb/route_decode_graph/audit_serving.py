"""Offline E024 serving receipts: independent schedules, not GPU recomputation.

audit(child_dir, expected_plan_sha256) returns a dict without writing evidence.
The frozen E023 checker owns all original 37-request numerical/state gates.
Only drained C8 slot inheritance and ordered route-event accounting are added.
Success tensor bytes are absent: parameter and serving arithmetic remain
source-pinned device bit/finite receipts. Resource/reap/restore are external.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'pareto_realistic/audit_qualification.py'
BASE_SHA='b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
NUMERIC_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
FIXED={
    HERE.parent/'pareto_realistic/qualify_native.py':'26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    HERE.parent/'long_decode_graph/qualify.py':'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
    HERE/'candidate.py':'6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3',
    HERE.parent/'long_decode_graph/candidate.py':'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
    HERE.parent/'long_decode_graph/model-source-pins.json':'7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
}
PARAMETERS=(('norm.weight',[5120],'mlx.core.bfloat16'),
    ('layers.0.linear_attn.in_proj_a.weight',[48,640],'mlx.core.uint32'),
    ('layers.0.linear_attn.in_proj_a.scales',[48,320],'mlx.core.uint8'))
BF16='mlx.core.bfloat16'


class Incomplete(Exception):pass
def need(ok,message):
    if not ok:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,allow_nan=False)
def same(a,b,message):need(canonical(a)==canonical(b),message)
def read(path):
    need(Path(path).suffix!='.jsonl','stream JSONL only')
    with Path(path).open() as f:return json.load(f)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def base_auditor():
    need(sha(BASE)==BASE_SHA,'frozen E023 auditor SHA')
    spec=importlib.util.spec_from_file_location('_e024_serving_audit_base',BASE)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def array_row(label,name,shape,dtype=BF16):
    return dict(check=label,name=name,shape=shape,dtype=dtype,
        bit_equal=True,reference_finite=True,candidate_finite=True)


def state_rows(label,layout,B,L,prefix='root',single=False):
    rows=[]
    for i,kind in enumerate(layout):
        if kind=='full_attention':
            fields=[(name,[1 if single else B,4,L,256],BF16) for name in (('keys','values') if single else ('K','V'))]
        else:
            fields=[('conv_state' if single else 'conv',([3,10240] if single else [B,3,10240]),BF16),
                    ('S',([48,128,128] if single else [B,48,128,128]),'mlx.core.float32')]
        rows.extend(array_row(label,f'{prefix}/{i}/{name}',shape,dtype) for name,shape,dtype in fields)
    return rows


class RouteEvents:
    """Single-pass cursor; every event consumed by a fixed diagnostic or step."""
    def __init__(self,stream):
        self.stream=stream;self.digest=hashlib.sha256();self.bytes=self.lines=0
        self.families=set();self.hits={};self.fallbacks=0;self.comparisons=0;self.direct_calls=0;self.changed=0

    def take(self,event):
        raw=self.stream.readline()
        need(raw and raw.endswith(b'\n'),'missing/truncated route event: '+event)
        self.digest.update(raw);self.bytes+=len(raw);self.lines+=1
        row=json.loads(raw);need(row.get('event')==event,'route event order at line '+str(self.lines))
        return row

    def route(self,phase,step,label,B,used,L):
        need(257<=used<=1536,'unexpected route event outside dispatch domain')
        graph=1<=B<=8 and L>=1536
        if graph:
            need(L==1536,'unqualified serving physical capacity')
            bound=1023 if used<1024 else 1536;self.families.add(bound)
            hit=self.hits.setdefault(phase,{'1023':0,'1536':0});hit[str(bound)]+=1
        else:self.fallbacks+=1
        expected=dict(event='route-call',phase=phase,step=step,direct_label=label,status='SUBMITTED',B=B,used=used,
            capacities=[[L,L] for _ in range(16)],expected_graph=graph,actual_invocations=int(graph),functions=len(self.families))
        same(self.take('route-call'),expected,'actual callable/domain/step/label/backing/family mismatch')

    def compare(self,label,expected):
        row=self.take('parameter-comparison')
        need(set(row)=={'event','label','arrays'} and row['label']==label,'parameter comparison order/label')
        # Retained caches deliberately compare both arms under the same labels;
        # exact multiplicities preserve those checks without inventing names.
        same(dict(Counter(map(canonical,row['arrays']))),dict(Counter(map(canonical,expected))),
             'parameter full original-dtype/shape/bit/finite comparison: '+label)
        self.comparisons+=len(expected)

    def direct(self,label,B,used,layout,extra=()):
        self.route('parameter-diagnostic',None,label,B,used,1536)
        rows=list(extra)+state_rows(label+'/input',layout,B,1536,'root/caches')
        rows += [array_row(label+'/input','root/'+n,[B],'mlx.core.int32') for n in ('positions','lens')]
        rows += [array_row(label+'/logits','root',[B,248320])]
        rows += state_rows(label+'/state',layout,B,1536)
        self.compare(label,rows);self.direct_calls+=1

    def parameters(self,fixtures,layout):
        prompts=[]
        for f in fixtures:
            ids=f['prompt_ids'];need(ids,'empty diagnostic source prompt')
            prompts.append({**f,'fixture_id':'diagnostic-repeat-'+f['fixture_id'],
                'prompt_ids':(ids*((1023+len(ids)-1)//len(ids)))[:1023],
                'source_fixture_id':f['fixture_id'],'diagnostic_only':True,'normal_eos':True,'max_output_tokens':4})
        same(self.take('parameter-inputs'),dict(event='parameter-inputs',prompts=prompts,
            history='diagnostic repeated D IDs, not generated history'),'diagnostic input provenance')
        for used in (1023,1024):
            end=used-1;L=((end+255)//256)*256
            for i in range(8):
                label=f'parameter-prefill-{end}-{i}'
                self.compare(label,[array_row(label+'/logits','root',[1,248320])]
                    +state_rows(label+'/state',layout,1,L,single=True))
            for n,B in enumerate((1,8,1)):
                self.direct(f'parameter-baseline-{used}-{n}-B{B}',B,used,layout)
            for path,shape,dtype in PARAMETERS:
                for n,B in enumerate((1,8,1)):
                    label=f'parameter-changed-{path}-{used}-{n}-B{B}'
                    extra=[array_row(path+'/'+suffix,'root',shape,dtype) for suffix in ('original-values','independent-new-values')] if n==0 else []
                    self.direct(label,B,used,layout,extra)
                    same(self.take('parameter-eager-change'),dict(event='parameter-eager-change',label=label,
                        parameter=path,used=used,B=B,changed_eager_logits=True),'meaningful eager parameter change receipt')
                    self.changed+=1
                same(self.take('parameter-original-handles-restored'),dict(event='parameter-original-handles-restored',
                    parameter=path,both=True),'both original parameter handles restored')
                for n,B in enumerate((1,8,1)):
                    self.direct(f'parameter-restored-{path}-{used}-{n}-B{B}',B,used,layout)
                    self.compare('restored-original',[array_row(f'parameter-restored-original-{path}-{used}-{n}','root',[B,248320])])
            retained=[]
            for _ in range(2):
                for i in range(8):retained+=state_rows('parameter-prefill-retained',layout,1,L,f'root/{i}',single=True)
            self.compare('parameter-prefill-retained',retained)
        need(self.direct_calls==42 and self.changed==18,'full fixed parameter diagnostic')

    def finish(self):
        need(self.stream.read(1)==b'','unmatched extra route event')
        return dict(sha256=self.digest.hexdigest(),bytes=self.bytes,lines=self.lines,
            parameter_direct_calls=self.direct_calls,meaningful_eager_change_receipts=self.changed,
            parameter_array_comparisons=self.comparisons,route_hits=self.hits,long_fallback_calls=self.fallbacks,
            call_binding='CPU case-step/direct label plus independently inferred within-step order; no explicit token-key receipt')


def extended_case(base,routes,counts):
    original=base.CaseAudit;previous=None
    class ServingCase(original):
        def __init__(self,*args,**kwargs):
            nonlocal previous
            super().__init__(*args,**kwargs)
            self.old_ids=[];self.capacity=0
            if self.name=='lifecycle':
                need(previous is not None and previous.name=='D-C8' and previous.C==self.C==8,'reused C8 predecessor')
                need(previous.next==len(previous.jobs) and not previous.requests and not previous.waiting
                     and not previous.running and not previous.staging and previous.previous_ahead is None
                     and not previous.previous_arrays and previous.previous_batch==0 and not previous.old_ids
                     and previous.capacity==0,'fully drained reused predecessor')
                need(all(type(x) is int for x in previous.free) and sorted(previous.free)==list(range(self.C)),
                     'complete reused free-slot permutation')
                base.same(self.finished,previous.finished,'reused pending retire IDs')
                self.free=list(previous.free)  # Never alias the preceding final summary.
            elif self.name=='D-C8':routes.families=set()  # New actual model pair.
            previous=self

        def check_arrays(self,row,calls,pre_staging,has_future,B):
            super().check_arrays(row,calls,pre_staging,has_future,B)
            decode=row['schedule']['decode'];requests={x['rid']:x for x in row['schedule']['requests']}
            new=[rid for rid in decode if rid not in self.old_ids]
            keep=[rid for rid in decode if rid in self.old_ids];L=self.capacity
            if decode and decode!=self.old_ids:
                required=max([len(requests[r]['prompt_ids'])+requests[r]['sampling']['max_tokens'] for r in new]
                    +[self.capacity if keep else 0]);L=((required+255)//256)*256
            # Bind actual padded arrays to reconstructed rebuild history, not an
            # event's self-reported capacity. The inherited checker owns all
            # other shape/batch/name/bit/finite and semantic-key comparisons.
            for item in row['arrays']:
                if item['name'].rsplit('/',1)[-1] not in ('K','V'):continue
                label=item['check'];wanted=None
                if label=='before-step-state' and item['name'].startswith(('root/committed/','root/ahead/')):wanted=self.capacity
                elif label.startswith("call-state ('decode',") or label.endswith('/committed-not-rewritten-by-ahead'):wanted=L
                elif label in ('before-retirement-state','after-retirement-state') and item['name'].startswith(('root/committed/','root/ahead/')):wanted=L
                if wanted is not None:need(item['shape'][2]==wanted,'reconstructed padded KV capacity')
            consumed=row['ahead']['candidate']['consumed_prior_future']
            for key in calls:
                if key[0]!='decode' or key==consumed:continue
                used=max(key[2])+1
                if 257<=used<=1536:routes.route(self.name,self.step,None,len(key[1]),used,L)
            bucket=counts.setdefault(self.name,dict(consumed=0,discarded_prior=0,partial_retirement=0,refills=0))
            bucket['consumed']+=int(consumed is not None)
            bucket['discarded_prior']+=int(row['ahead']['candidate']['discarded_prior_future'] is not None)
            after=row['after']['engine']['batch_rids']
            bucket['partial_retirement']+=int(bool(after) and bool(set(decode)-set(after)))
            bucket['refills']+=int(bool(self.old_ids) and all(r in decode for r in self.old_ids) and bool(new))
            self.old_ids=list(after);self.capacity=L if after else 0
    return ServingCase


def binding(child,digest,base):
    plan,top,claimed=[read(child/p) for p in ('plan.json','complete.json','base/complete.json')]
    if any(x.get('status')=='INCOMPLETE' for x in (top,claimed)):raise Incomplete('numerical result not terminal')
    need(top['status']==claimed['status']=='PASS','producer numerical failure')
    need(sha(child/'plan.json')==digest==top['plan_sha256'],'registered serving plan SHA')
    need(set(plan)=={'schema','qualifier_sha256','qualification','sources','deadline_s'}
         and plan['schema']=='E024-route-serving-plan-v1' and type(plan['deadline_s']) is int
         and plan['deadline_s']==1800,'exact five-field serving plan')
    sources=plan['sources'];fixed={str(p):h for p,h in FIXED.items()}
    need(set(sources)==set(fixed)|{str(HERE/'integration.py'),str(HERE/'serving_qualification.py')},'exact serving source set')
    for p,h in sources.items():need(h==sha(p) and (p not in fixed or h==fixed[p]),'serving source SHA: '+p)
    need(plan['qualifier_sha256']==sources[str(HERE/'serving_qualification.py')],'actual qualifier SHA')
    same(top['sources'],sources,'result source pins')
    need(top['schema']=='E024-route-serving-result-v1' and top['same_schedule_qualified'] is True
         and top['source_pins_rechecked'] is True and top['http_qualified'] is False
         and top['performance_qualified'] is False and top['identity_refill_enabled'] is False
         and type(top['total_requests']) is int and top['total_requests']==37,'serving final scope')
    same(top['base_result'],dict(path='base/complete.json',sha256=sha(child/'base/complete.json'),status='PASS',returncode=0),'base terminal SHA/exit')
    q=plan['qualification']
    need(set(q)=={'path','sha256'} and Path(q['path']).is_absolute() and q['sha256']==NUMERIC_SHA
         and sha(q['path'])==sha(child/'base/plan.json')==NUMERIC_SHA,'original numeric plan')
    numeric=read(child/'base/plan.json')
    need(claimed['plan_sha256']==NUMERIC_SHA and claimed['qualifier_sha256']==fixed[str(HERE.parent/'pareto_realistic/qualify_native.py')]
         and claimed['numeric_only_not_http'] is True and claimed['source_model_pins_rechecked'] is True
         and type(claimed['total_requests']) is int and claimed['total_requests']==37,'original terminal/source/model receipt')
    same(claimed['candidate'],numeric['candidate'],'inherited candidate flags')
    need(numeric['candidate']==dict(compile_decode=True,fuse_decode=True,gates_compile=True,norm_reuse=False)
         and type(numeric['deadline_s']) is int and numeric['deadline_s']==1800,'fixed original flags/deadline')
    manifests={}
    for field in ('fixtures','source_manifest','model_manifest'):
        item=numeric[field];need(sha(item['path'])==item['sha256'],'manifest SHA: '+field);manifests[field]=read(item['path'])
    root=Path(numeric['package_root'])/'tutorial_code/miniengine';core=manifests['source_manifest']['files']
    need(set(core)=={str(p.relative_to(root)) for p in root.rglob('*.py')},'complete frozen core set')
    for rel,h in core.items():need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(root/rel)==h,'frozen core source')
    model=manifests['model_manifest'];config_path=Path(numeric['model'])/'config.json'
    need(model['schema']=='E023-model-v1' and model['model']==numeric['model']
         and model['revision']==Path(numeric['model']).name and sha(config_path)==model['files']['config.json'],'model identity/config')
    config=read(config_path);config=config.get('text_config',config)
    need(config['layer_types']==['full_attention' if i%4==3 else 'linear_attention' for i in range(64)]
         and config['vocab_size']==248320 and config['hidden_size']==5120,'fixed 64-layer model dimensions')
    fixtures=manifests['fixtures'];need(numeric['fixtures']['sha256']==base.FIXTURE_SHA and fixtures['model']==numeric['model'],'fixture identity')
    fixtures=fixtures['fixtures'];need(len(fixtures)==8 and len({f['fixture_id'] for f in fixtures})==8,'eight unique D fixtures')
    eos=fixtures[0]['eos_token_ids']
    need(all(f['eos_token_ids']==eos and f['normal_eos'] is True and f['max_output_tokens']==512 for f in fixtures),'normal EOS/cap512')
    return plan,top,claimed,fixtures,config,eos


def audit(child_dir,expected_plan_sha256):
    child=Path(child_dir)
    result=dict(schema='E024-serving-independent-audit-v1',status='INCOMPLETE',qualified=False,ledger_opened=False,
        performance_qualified=False,http_qualified=False,offline_tensor_bytes_rechecked=False,
        base_auditor_sha256=BASE_SHA,audit_source_sha256=sha(__file__),
        scope='Independent 37-request schedule reconstruction and complete source-pinned device comparison receipts; no offline GPU arithmetic/tensor-byte recomputation or owner/resource/restoration claim.')
    try:
        for name in ('complete.json','plan.json','base/complete.json','base/plan.json','base/steps.jsonl','route-events.jsonl'):
            if not (child/name).is_file():raise Incomplete('missing '+name)
        base=base_auditor();plan,top,claimed,fixtures,config,eos=binding(child,expected_plan_sha256,base)
        stable={p:sha(child/p) for p in ('plan.json','complete.json','base/complete.json','base/plan.json')}
        ledger_path=child/'base/steps.jsonl';events_path=child/'route-events.jsonl'
        before=[p.stat() for p in (ledger_path,events_path)];counts={}
        with events_path.open('rb') as stream:
            routes=RouteEvents(stream);routes.parameters(fixtures,config['layer_types'])
            base.CaseAudit=extended_case(base,routes,counts);result['ledger_opened']=True
            ledger=base.stream_ledger(ledger_path,fixtures,config,eos,claimed['cases'],partial=False)
            result['ledger']=ledger;result['route_events']=routes.finish()
        need([len(c['completed']) for c in ledger['cases']]==[8,16,13] and not ledger['partial_cases']
             and ledger['truncated_final_line'] is None,'complete untruncated37')
        expected=dict(installs=2,restores=4,parameter_checks_complete=True,parameter_direct_calls=42,
            route_hits=routes.hits,long_fallback_calls=routes.fallbacks,cases=counts,identity_refill_enabled=False)
        same(top['coverage'],expected,'independently reconstructed coverage')
        for case in ('D-C1','D-C8'):
            need(all(routes.hits.get(case,{}).get(b,0)>0 for b in ('1023','1536')) and counts[case]['consumed']>0,'both actual graph families and consumed ahead: '+case)
        need(sum(c['partial_retirement'] for c in counts.values())>0 and counts['D-C8']['refills']>0,'actual partial retirement and C8 refill')
        discarded=sum(c['discarded_prior'] for c in counts.values())
        eos_count=sum(bool(r['raw_ids']) and r['raw_ids'][-1] in eos for c in ledger['cases'] for r in c['completed'])
        for name,value in dict(discarded_ahead_observed=discarded>0,discarded_prior_ahead_count=discarded,
                               observed_eos_requests=eos_count,eos_retirement_observed=eos_count>0).items():
            same(top[name],value,'actual coverage report: '+name)
        need(all(sha(child/p)==h for p,h in stable.items()),'terminal file changed during audit')
        for p,s,n in zip((ledger_path,events_path),before,(ledger['bytes'],routes.bytes)):
            after=p.stat();need((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
                and n==s.st_size,'closed ledger changed/truncated during audit')
        need(all(sha(p)==h for p,h in plan['sources'].items()),'serving source changed during audit')
        result.update(status='PASS',qualified=True,plan_sha256=expected_plan_sha256,base_plan_sha256=NUMERIC_SHA,
            terminal_sha256=stable,coverage=expected,discarded_ahead_observed=discarded>0,
            observed_eos_requests=eos_count,weight_shards_rehashed=False,
            uncovered='Unobserved ahead invalidation, cancellation/sampling-key updates, HTTP/performance and combined E028 remain outside this result.')
    except (Incomplete,FileNotFoundError) as exc:result.update(status='INCOMPLETE',reason=str(exc))
    except Exception as exc:result.update(status='FAIL',first_failure=dict(type=type(exc).__name__,message=str(exc)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);need(a.out.parent.is_dir() and not a.out.exists(),'new output path required')
    result=audit(a.child,a.plan_sha256)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],qualified=result['qualified'])))
    return int(not result['qualified'])


if __name__=='__main__':raise SystemExit(main())
