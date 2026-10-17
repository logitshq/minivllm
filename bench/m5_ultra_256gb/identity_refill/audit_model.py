"""Independent E028 child numerical audit; stdlib only, no measurement imports.

audit(child_dir) returns PASS/FAIL/INCOMPLETE and qualified. The frozen E023
auditor still owns all 37-request scheduling/logits/cache/ahead/retirement gates.
Only its private CaseAudit is extended for E028's additional predecode groups.
Both JSONL files are streamed, without rewriting evidence or fabricating an old
owner layout. Runtime/resource/reap/controller proofs belong to the owner audit.
Recorded GPU comparison flags are audited; GPU arithmetic is not rerun here.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDITOR = HERE.parent / 'pareto_realistic/audit_qualification.py'
AUDITOR_SHA = 'b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
PLAN_SHA = 'bc4890e6d302141597eae8e0e9347c093307a8492c09f958f42bdb8d1c7aa73c'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SOURCES = {
    HERE/'model_qualification.py': 'b8b84a57b63c6f46ea1449ec9eeed8310d1b4dbe074a81082e5a665371061042',
    HERE.parent/'pareto_realistic/qualify_native.py': '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    HERE/'integration.py': 'ea0d6cb6af7c80d7580b4ab1c8319a4b76110fd1d73cf660c048c8dd45cf0c52',
    HERE/'candidate.py': 'bb9f8a461625e97c80e75140c9d9b7b8718a633595ab4f3a422267e752ded8f5',
}
PREFIX = 'E028-refill-predecode-state '


class Incomplete(Exception):
    pass


def need(ok, message):
    if not ok: raise ValueError(message)


def read(path):
    need(Path(path).suffix != '.jsonl', 'JSONL must be streamed')
    with Path(path).open() as f: return json.load(f)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8 << 20): h.update(chunk)
    return h.hexdigest()


def source_auditor():
    need(sha(AUDITOR)==AUDITOR_SHA, 'frozen numerical auditor changed')
    spec=importlib.util.spec_from_file_location('_e028_independent_base_audit',AUDITOR)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    return base


class Refills:
    """One event consumed for each independently inferred eligible transition."""
    def __init__(self, stream):
        self.stream,self.digest,self.bytes,self.lines=stream,hashlib.sha256(),0,0
        self.per_case=Counter()
        self.counters=[dict(identity_refill=0,fallback=0) for _ in range(2)]

    def consume(self, expected, same):
        raw=self.stream.readline()
        need(raw and raw.endswith(b'\n'), 'missing or truncated paired refill event')
        self.digest.update(raw);self.bytes+=len(raw);self.lines+=1
        same(json.loads(raw),expected,'refill event/step/transition/counter mismatch')
        self.per_case[expected['transition']['case']]+=1

    def finish(self, coverage, same):
        need(self.stream.read(1)==b'', 'unmatched extra refill event')
        need(self.per_case['D-C8']>=1 and self.per_case['D-C1']==0, 'D-C8 refill required; D-C1 must have none')
        expected=dict(candidate_installs=2,paired_identity_refills=self.lines,compared_arrays=128*self.lines,
                      per_case=dict(self.per_case),restored_observers=4,candidate_final_counters=self.counters)
        same(coverage,expected,'coverage must match independently counted branches and full states')
        return dict(sha256=self.digest.hexdigest(),bytes=self.bytes,events=self.lines,
                    arrays=128*self.lines,per_case=dict(self.per_case),candidate_final_counters=self.counters)


def extended_case(base, refills, c8_steps):
    original=base.CaseAudit

    class RefillCase(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.old_ids,self.old_lens,self.capacity=[],[],0

        def check_arrays(self,row,calls,pre_staging,has_future,B):
            decode=row['schedule']['decode']
            requests={x['rid']:x for x in row['schedule']['requests']}
            positions={rid:i for i,rid in enumerate(self.old_ids)}
            keep=[positions[rid] for rid in decode if rid in positions]
            new=[rid for rid in decode if rid not in positions]
            L=self.capacity
            if decode and decode!=self.old_ids:
                required=max([len(requests[r]['prompt_ids'])+requests[r]['sampling']['max_tokens'] for r in new]
                             +[self.capacity if keep else 0])
                L=(required+255)//256*256
            eligible=bool(decode and self.old_ids and new and keep==list(range(len(self.old_ids))) and L==self.capacity)
            original_rows=[];extra=[]
            for item in row['arrays']:
                (extra if item['check'].startswith(PREFIX) else original_rows).append(item)
            counter=refills.counters[0 if self.name=='D-C1' else 1]
            before=dict(counter)
            if decode and decode!=self.old_ids: counter['identity_refill' if eligible else 'fallback']+=1
            # Bind the retained pre-step attention capacity to the independently
            # tracked rebuild/clear history, not to the event's claimed capacity.
            for item in original_rows:
                if (item['check']=='before-step-state' and item['name'].startswith('root/committed/')
                        and item['name'].rsplit('/',1)[-1] in ('K','V')):
                    need(item['shape']==[len(self.old_ids),4,self.capacity,256], 'old batch capacity/shape drift')
            if eligible:
                label=PREFIX+repr(decode)
                expected={}
                for i,kind in enumerate(self.layout):
                    fields={'K':[B,4,L,256],'V':[B,4,L,256]} if kind=='full_attention' else {'conv':[B,3,10240],'S':[B,48,128,128]}
                    for field,shape in fields.items():
                        expected[f'root/layers/{i}/{field}']=(shape,'mlx.core.float32' if field=='S' else 'mlx.core.bfloat16')
                need(len(extra)==len(expected)==128 and len({x['name'] for x in extra})==128, 'exact 64-layer/128-array refill required')
                for item in extra:
                    need(item['check']==label and item['name'] in expected, 'refill label/layer mismatch')
                    shape,dtype=expected[item['name']]
                    base.same(item['shape'],shape,'refill full shape/batch/capacity mismatch')
                    need(item['dtype']==dtype and all(item[k] is True for k in ('bit_equal','reference_finite','candidate_finite')),
                         'refill original dtype/bit/finite failure')
                transition=dict(case=self.name,old_ids=self.old_ids,old_lens=self.old_lens,keep=keep,new_ids=new,decode_ids=decode,capacity=L)
                refills.consume(dict(event='identity-refill',status='PASS',
                    observer_step=self.step+(c8_steps if self.name=='lifecycle' else 0),transition=transition,
                    candidate_witness=dict(before=before,after=dict(counter),path='identity_refill',reason='identity_keep_unchanged_capacity'),arrays=128),base.same)
            else: need(not extra, 'refill group without eligible actual transition')
            # A shallow in-memory projection only: the original checker receives
            # every original group unchanged; the evidence row/file is untouched.
            super().check_arrays({**row,'arrays':original_rows},calls,pre_staging,has_future,B)
            self.array_counts.update(all_arrays=len(extra),state_arrays=len(extra),e028_refill_arrays=len(extra))
            retired={rid for rid,token,done in row['reference_delivered'] if done}
            if retired.intersection(decode) and any(r not in retired for r in decode): counter['fallback']+=1
            self.old_ids=list(row['after']['engine']['batch_rids'])
            self.old_lens=list(row['after']['engine']['batch_lens'])
            self.capacity=L if self.old_ids else 0

    return RefillCase


def binding(child,base):
    top,plan,result=[read(child/p) for p in ('complete.json','plan.json','base/complete.json')]
    if top.get('status')=='INCOMPLETE' or result.get('status')=='INCOMPLETE': raise Incomplete('child numerical result not terminal')
    need(top['status']==result['status']=='PASS','child/base numerical failure')
    need(sha(child/'plan.json')==PLAN_SHA and top['plan_sha256']==PLAN_SHA,'fixed E028 plan SHA')
    need(set(plan)=={'schema','qualification','sources'} and plan['schema']=='E028-model-qualification-plan-v1','E028 plan schema')
    expected={str(p):h for p,h in SOURCES.items()}
    base.same(plan['sources'],expected,'four exact source pins');base.same(top['sources'],expected,'result source pins')
    for path,h in expected.items(): need(sha(path)==h,'current small source SHA: '+path)
    need(top['schema']=='E028-model-qualification-result-v1' and top['source_pins_rechecked'] is True
         and top['numerical_only_not_http'] is True and top['performance_qualified'] is False and type(top['total_requests']) is int
         and top['total_requests']==37,'E028 final scope/pins/request count')
    base.same(top['base_result'],dict(path='base/complete.json',sha256=sha(child/'base/complete.json'),status='PASS',returncode=0),'base result SHA/exit binding')
    q=plan['qualification']
    need(set(q)=={'path','sha256'} and q['sha256']==BASE_PLAN_SHA and sha(q['path'])==BASE_PLAN_SHA
         and sha(child/'base/plan.json')==BASE_PLAN_SHA,'original E023 plans')
    numeric=read(child/'base/plan.json')
    need(result['plan_sha256']==BASE_PLAN_SHA and result['qualifier_sha256']==expected[str(HERE.parent/'pareto_realistic/qualify_native.py')]
         and result['numeric_only_not_http'] is True and result['source_model_pins_rechecked'] is True
         and type(result['total_requests']) is int and result['total_requests']==37,'original complete37/source/qualifier gate')
    base.same(result['candidate'],numeric['candidate'],'original candidate flags')
    manifests={}
    for name in ('fixtures','source_manifest','model_manifest'):
        item=numeric[name];need(sha(item['path'])==item['sha256'],'manifest SHA '+name);manifests[name]=read(item['path'])
    root=Path(numeric['package_root'])/'tutorial_code/miniengine'
    sources=manifests['source_manifest']['files']
    need(set(sources)=={str(p.relative_to(root)) for p in root.rglob('*.py')},'complete frozen core source set')
    for rel,h in sources.items():
        need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(root/rel)==h,'frozen core source SHA')
    model=manifests['model_manifest'];config_path=Path(numeric['model'])/'config.json'
    need(model['schema']=='E023-model-v1' and model['model']==numeric['model'] and model['revision']==Path(numeric['model']).name
         and sha(config_path)==model['files']['config.json'],'model identity/config SHA')
    config=read(config_path);config=config.get('text_config',config)
    need(len(config['layer_types'])==64 and config['layer_types'].count('full_attention')==16 and config['vocab_size']==248320,'fixed64-layer model')
    fixtures=manifests['fixtures']
    need(numeric['fixtures']['sha256']==base.FIXTURE_SHA and fixtures['model']==numeric['model'] and len(fixtures['fixtures'])==8,'fixed eight fixtures')
    fixtures=fixtures['fixtures'];eos=fixtures[0]['eos_token_ids']
    need(all(f['eos_token_ids']==eos and f['normal_eos'] is True and f['max_output_tokens']==512 for f in fixtures),'normal EOS/cap fixtures')
    return top,result,fixtures,config,eos


def audit(child_dir):
    child=Path(child_dir)
    result=dict(schema='E028-model-independent-audit-v1',status='INCOMPLETE',qualified=False,ledger_opened=False,
                performance_qualified=False,scope='Recorded full-model same-schedule numerical checks only; no HTTP/performance or owner/service qualification.',
                audit_source_sha256=sha(__file__),base_auditor_sha256=AUDITOR_SHA)
    try:
        for name in ('complete.json','plan.json','base/complete.json','base/plan.json','base/steps.jsonl','refills.jsonl'):
            if not (child/name).is_file(): raise Incomplete('missing '+name)
        base=source_auditor();top,claimed,fixtures,config,eos=binding(child,base)
        need([c['name'] for c in claimed['cases']]==list(base.CASES),'all three completed cases required')
        with (child/'refills.jsonl').open('rb') as stream:
            refills=Refills(stream)
            base.CaseAudit=extended_case(base,refills,claimed['cases'][1]['steps'])
            result['ledger_opened']=True
            ledger=base.stream_ledger(child/'base/steps.jsonl',fixtures,config,eos,claimed['cases'],partial=False)
            result['ledger']=ledger
            result['refills']=refills.finish(top['coverage'],base.same)
        need([len(c['completed']) for c in ledger['cases']]==[8,16,13] and not ledger['partial_cases']
             and ledger['truncated_final_line'] is None,'complete37/untruncated ledger')
        result.update(status='PASS',qualified=True,plan_sha256=PLAN_SHA,base_plan_sha256=BASE_PLAN_SHA,
                      child_complete_sha256=sha(child/'complete.json'),base_complete_sha256=sha(child/'base/complete.json'),
                      weight_shards_rehashed=False,runtime_and_owner_evidence='Must be checked by the separate owner audit')
    except (Incomplete,FileNotFoundError) as error: result.update(status='INCOMPLETE',reason=str(error))
    except Exception as error: result.update(status='FAIL',first_failure=dict(type=type(error).__name__,message=str(error)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(argv);need(a.out.parent.is_dir() and not a.out.exists(),'new output path required')
    result=audit(a.child)
    with a.out.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],qualified=result['qualified'])))
    return int(not result['qualified'])


if __name__=='__main__': raise SystemExit(main())
