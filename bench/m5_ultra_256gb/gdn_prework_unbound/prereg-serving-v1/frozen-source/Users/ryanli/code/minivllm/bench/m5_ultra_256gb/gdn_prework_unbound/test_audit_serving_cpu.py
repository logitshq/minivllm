"""Focused fake receipt/source tests. No real model/device/backend qualification."""
import builtins
import copy
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_prework_unbound import audit_serving as a
from bench.m5_ultra_256gb.identity_refill import test_audit_model_v2_cpu as slots


def cursor(rows):return a.Dispatch(io.BytesIO(b''.join((json.dumps(row)+'\n').encode() for row in rows)))
def event(case,step,key):
    # Fixture records are synthetic device claims, not qualification evidence.
    mode=key[0];B=len(key[1]) if mode=='decode' else key[3];used=max(key[2])+1 if mode=='decode' else 0
    path='prefill-untouched' if mode=='prefill' else 'short-packed-bypass' if used<=256 else 'B1-candidate' if B==1 else 'B>1-fallback'
    return dict(event='prework-model-call',status='SUBMITTED',case=case,step=step,mode=mode,B=B,used=used,
                semantic_key=key,path=path,candidate=48 if path=='B1-candidate' else 0,fallback=48 if path=='B>1-fallback' else 0)

def toy_dispatch(rows):
    return [event(row['case'],row['step'],key) for row in rows for key in row['call_keys']
            if key!=row['ahead']['candidate']['consumed_prior_future']]

def canary_fixture():
    child=Path('/fake/canary');refs={k:dict(path=str(child/filename),sha256=k+'-sha') for k,filename in
                                   [('audit','audit.json'),('plan','plan.json'),('complete','complete.json')]}
    scope=dict(full_model_qualified=False,http_qualified=False,performance_qualified=False)
    sources={str(a.HERE/'candidate.py'):a.CANDIDATE_SHA,str(a.HERE/'qualify.py'):a.CANARY_SHA}
    audit=dict(schema='GDN-prework-real-layer-independent-v1',status='PASS',canary_qualified=True,
        audit_source_sha256=a.CANARY_AUDIT_SHA,child_directory=str(child),plan_sha256='plan-sha',complete_sha256='complete-sha',
        raw_files=60,full_output_state_pairs=54,causal_state_pairs=24,full_vocabulary_greedy_checks=6,
        device_comparison_arrays=162,ledger_sha256='events',captures_sha256='captures',source_pins=sources,**scope)
    complete=dict(schema='GDN-prework-real-layer-result-v1',status='PASS',canary_qualified=True,
        candidate_calls=18,candidate_compared_calls=18,reference_chains=2,candidate_chains=6,
        device_run=True,instance_hooks_restored=True,source_model_pins_rechecked=True,plan_sha256='plan-sha',
        dispatch_counters={str(i):dict(candidate=6,fallback=0) for i in (0,32,62)},sources=sources,**scope)
    inherited=dict(path='/fake/inherited.json',sha256='inherited')
    plan=dict(schema='GDN-prework-real-layer-plan-v1',qualifier_sha256=a.CANARY_SHA,
              candidate=dict(path=str(a.HERE/'candidate.py'),sha256=a.CANDIDATE_SHA),e024_plan=inherited)
    values={refs['audit']['path']:audit,refs['plan']['path']:plan,refs['complete']['path']:complete,
            inherited['path']:dict(e023_plan=dict(path='/fake/base.json',sha256=a.NUMERIC_SHA))}
    hashes={str(child/'events.jsonl'):'events',str(child/'captures.json'):'captures','/fake/base.json':a.NUMERIC_SHA}
    return refs,values,hashes

class Tests(unittest.TestCase):
    def test_import_and_missing_terminal_never_load_backend_or_big_ledger(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='mlx' or name.startswith(('mlx.','numpy','omlx','mlx_lm','mlx_vlm')):raise AssertionError('backend import')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):runpy.run_path(a.__file__)
        with tempfile.TemporaryDirectory() as root:
            result=a.audit(root,'unqualified')
        self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['qualified'] or result['ledger_opened'])
        self.assertFalse(result['http_qualified'] or result['performance_qualified'] or result['offline_tensor_bytes_rechecked'])

    def test_failed_terminal_cannot_be_promoted_and_does_not_open_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'base').mkdir()
            for name,data in [('plan.json',{}),('complete.json',{'status':'FAIL'}),('base/complete.json',{'status':'FAIL'}),('base/plan.json',{})]:
                (root/name).write_text(json.dumps(data))
            for name in ('dispatch.jsonl','base/steps.jsonl'):(root/name).write_bytes(b'must not be parsed\n')
            result=a.audit(root,'not-authorized')
        self.assertEqual(result['status'],'FAIL');self.assertFalse(result['qualified'] or result['ledger_opened'])

    def test_dispatch_all_domains_prefill_token_dimension_and_two_calls_same_step(self):
        keys=[['prefill','r',0,19],['decode',['r'],[255],[7]],['decode',['r'],[256],[8]],
              ['decode',['r'],[257],[9]],['decode',['a','b'],[900,901],[4,5]]]
        rows=[event('D-C1' if i<4 else 'D-C8',1,k) for i,k in enumerate(keys)]
        c=cursor(rows)
        for i,k in enumerate(keys):c.call('D-C1' if i<4 else 'D-C8',1,k)
        result=c.finish();self.assertEqual(result['lines'],5)
        self.assertEqual(rows[0]['B'],19);self.assertEqual(rows[0]['used'],0)
        self.assertEqual(c.per_case['D-C1'],dict(candidate=96,fallback=0,b1_calls=2,multirow_calls=0,prefill_calls=1,short_bypass_calls=1))
        self.assertEqual(c.per_case['D-C8']['fallback'],48)

    def test_dispatch_corrupt_call_types_keys_status_or_48_layer_count_rejected(self):
        key=['decode',['r'],[977],[8]];row=event('D-C1',2,key)
        for field,value in [('candidate',47),('fallback',48),('case','D-C8'),('step',True),('used',977),('B',2),
                            ('status','FAILED'),('mode','prefill'),('path','short-packed-bypass'),
                            ('semantic_key',['decode',['wrong'],[977],[8]])]:
            bad=copy.deepcopy(row);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):cursor([bad]).call('D-C1',2,key)
        pre=['prefill','r',0,19];bad=event('D-C1',1,pre);bad['used']=None
        with self.assertRaises(ValueError):cursor([bad]).call('D-C1',1,pre)

    def test_missing_truncated_extra_and_reordered_dispatch_fail(self):
        first=['decode',['r'],[900],[7]];second=['decode',['r'],[901],[8]]
        for rows in ([],[event('D-C1',3,second)]):
            with self.assertRaises(ValueError):cursor(rows).call('D-C1',3,first)
        raw=json.dumps(event('D-C1',3,first)).encode();c=a.Dispatch(io.BytesIO(raw))
        with self.assertRaisesRegex(ValueError,'truncated'):c.call('D-C1',3,first)
        c=cursor([event('D-C1',3,first),event('D-C1',3,second)]);c.call('D-C1',3,first)
        with self.assertRaisesRegex(ValueError,'extra'):c.finish()

    def test_consumed_ahead_is_not_a_second_executed_model_call(self):
        base=a.base_auditor();current=['decode',['x'],[977],[7]];future=['decode',['x'],[978],[8]]
        row=dict(ahead={'candidate':{'consumed_prior_future':current}})
        for extra in (False,True):
            events=([event('D-C1',9,current)] if extra else [])+[event('D-C1',9,future)]
            d=cursor(events);cls=a.extended_case(base,d);case=cls('D-C1',slots.FIXTURES,slots.CONFIG,[99]);case.step=9
            with patch.object(base.CaseAudit,'check_arrays') as inherited:
                if extra:
                    with self.assertRaises(ValueError):case.check_arrays(row,[current,future],set(),True,1)
                else:case.check_arrays(row,[current,future],set(),True,1);d.finish();self.assertEqual(d.lines,1)
                inherited.assert_called_once()

    def test_final_counters_bind_each_layer_and_actual_pair_lifetime(self):
        calls=[('D-C1',['decode',['r'],[900],[7]]),('D-C8',['decode',['a','b'],[900,900],[8,9]]),
               ('lifecycle',['decode',['z'],[901],[10]])]
        d=cursor([event(c,1,k) for c,k in calls])
        for c,k in calls:d.call(c,1,k)
        d.finish();coverage=d.expected_coverage()
        self.assertEqual(coverage['final_counters'][0],[dict(candidate=1,fallback=0)]*48)
        self.assertEqual(coverage['final_counters'][1],[dict(candidate=1,fallback=1)]*48)
        for corruption in ('redistributed','missing-layer','wrong-pair','unrestored','unobserved','extra-case'):
            bad=copy.deepcopy(coverage)
            if corruption=='redistributed':bad['final_counters'][0][0]['candidate']+=1;bad['final_counters'][0][1]['candidate']-=1
            elif corruption=='missing-layer':bad['final_counters'][1].pop()
            elif corruption=='wrong-pair':bad['final_counters'].reverse()
            elif corruption=='unrestored':bad['layer_restores']-=1
            elif corruption=='unobserved':bad['calls']+=1
            else:bad['per_case']['extra']={}
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):a.same(bad,d.expected_coverage(),'coverage')

    def pair(self):
        base=a.base_auditor();rows=slots.toy_rows();d=cursor(toy_dispatch(rows));cls=a.extended_case(base,d)
        c8=cls('D-C8',slots.FIXTURES,slots.CONFIG,[99])
        for row in rows[:3]:c8.accept(row)
        return base,cls,c8,d,rows

    def test_real_cpu_scheduler_reused_slots_and_pending_retirement_order(self):
        base,cls,c8,d,rows=self.pair();saved=list(c8.free)
        self.assertEqual(saved,[1,3,5,7,0,2,4,6]);life=cls('lifecycle',slots.FIXTURES,slots.CONFIG,[99],c8.finished)
        self.assertIsNot(life.free,c8.free);life.accept(rows[3]);d.finish();self.assertEqual(c8.free,saved)
        self.assertIs(life.accept.__func__,base.CaseAudit.accept);self.assertIs(life.finish.__func__,base.CaseAudit.finish)
        base,cls,c8,d,rows=self.pair();life=cls('lifecycle',slots.FIXTURES,slots.CONFIG,[99],c8.finished)
        bad=copy.deepcopy(rows[3]);bad['schedule']['requests'][0]['slot']=0
        with self.assertRaisesRegex(ValueError,'FCFS'):life.accept(bad)

    def test_bad_predecessor_not_seeded_from_lifecycle_row_and_c8_is_fresh(self):
        for mutation in ('live','duplicate-slot','bool-slot','pending'):
            base,cls,c8,_,_=self.pair();pending=list(c8.finished)
            if mutation=='live':c8.staging.add('leak')
            elif mutation=='duplicate-slot':c8.free[0]=c8.free[1]
            elif mutation=='bool-slot':c8.free[c8.free.index(1)]=True
            else:pending.reverse()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):cls('lifecycle',slots.FIXTURES,slots.CONFIG,[99],pending)
        base=a.base_auditor();cls=a.extended_case(base,cursor([]));c1=cls('D-C1',slots.FIXTURES,slots.CONFIG,[99]);c1.finished=['old']
        c8=cls('D-C8',slots.FIXTURES,slots.CONFIG,[99]);self.assertEqual(c8.finished,[]);self.assertEqual(c8.free,list(range(7,-1,-1)))

    def test_original_full64_dtype_finite_bit_and_complete_state_gate_runs_first(self):
        base=a.base_auditor();layout=['full_attention' if i%4==3 else 'linear_attention' for i in range(64)]
        config=dict(vocab_size=248320,layer_types=layout);key=['prefill','r',0,19];literal=repr(tuple(key))
        def arrays(prefix,label):
            result=[]
            for i,kind in enumerate(layout):
                fields=[('keys',[1,4,256,256]),('values',[1,4,256,256])] if kind=='full_attention' else [('conv_state',[3,10240]),('S',[48,128,128])]
                for field,shape in fields:result.append(dict(check=label,name=f'{prefix}/{i}/{field}',shape=shape,
                    dtype='mlx.core.float32' if field=='S' else 'mlx.core.bfloat16',bit_equal=True,reference_finite=True,candidate_finite=True))
            return result
        original=dict(arrays=[dict(check='logits '+literal,name='root',shape=[1,248320],dtype='mlx.core.bfloat16',bit_equal=True,reference_finite=True,candidate_finite=True)]
             +arrays('root','call-state '+literal)+arrays('root/staging/r','before-retirement-state'),
             ahead={'candidate':{'consumed_prior_future':None}},after={'state':{},'engine':{'batch_rids':[]}})
        for mutation in (None,'bit','finite','dtype','rank','missing'):
            d=cursor([event('D-C1',1,key)]);case=a.extended_case(base,d)('D-C1',slots.FIXTURES,config,[99]);case.step=1
            row=copy.deepcopy(original)
            if mutation=='bit':row['arrays'][0]['bit_equal']=False
            elif mutation=='finite':row['arrays'][0]['candidate_finite']=False
            elif mutation=='dtype':next(x for x in row['arrays'] if x['name'].endswith('/S'))['dtype']='mlx.core.bfloat16'
            elif mutation=='rank':next(x for x in row['arrays'] if x['name'].endswith('/S'))['shape']=[1,48,128,128]
            elif mutation=='missing':row['arrays'].pop()
            if mutation is None:case.check_arrays(row,[key],{'r'},False,0);d.finish()
            else:
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):case.check_arrays(row,[key],{'r'},False,0)
                self.assertEqual(d.lines,0)

    def test_canary_proof_binding_is_not_just_claimed_pass_and_real_pin_rejects_changed_bytes(self):
        refs,values,hashes=canary_fixture()
        def check(v,h):
            with patch.object(a,'pinned',side_effect=lambda r:v[r['path']]),patch.object(a,'sha',side_effect=lambda p:h[str(p)]):return a.canary_binding(refs)
        self.assertEqual(check(values,hashes)['candidate_calls'],18)
        for mutation in ('plan','raw','counts','source','scope','dispatch'):
            v,h=copy.deepcopy(values),dict(hashes);audit=v[refs['audit']['path']]
            if mutation=='plan':audit['plan_sha256']='other'
            elif mutation=='raw':h['/fake/canary/events.jsonl']='changed'
            elif mutation=='counts':audit['raw_files']=59
            elif mutation=='source':audit['audit_source_sha256']='wrong'
            elif mutation=='scope':audit['performance_qualified']=True
            else:v[refs['complete']['path']]['dispatch_counters']['32']['candidate']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):check(v,h)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory).resolve()/'proof.json';p.write_text('{"status":"PASS"}')
            ref=dict(path=str(p),sha256=a.sha(p));self.assertEqual(a.pinned(ref)['status'],'PASS');p.write_text('{"status":"FAIL"}')
            with self.assertRaisesRegex(ValueError,'changed'):a.pinned(ref)

    def test_stream_reuses_frozen_code_without_module_global_mutation_even_on_error(self):
        base=a.base_auditor();original=dict(vars(base));cls=a.extended_case(base,cursor([]));executed=[]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'empty.jsonl';path.write_bytes(b'')
            try:a.stream_with(base,cls,path,slots.FIXTURES,slots.CONFIG,[99],[],partial=False)
            except ValueError as exc:
                self.assertIn('missing case',str(exc));tb=exc.__traceback__
                while tb is not None:
                    if tb.tb_frame.f_code is base.stream_ledger.__code__:executed.append(tb.tb_frame.f_globals)
                    tb=tb.tb_next
            else:self.fail('empty original37 ledger was accepted')
        self.assertEqual(vars(base),original);self.assertIs(base.CaseAudit,original['CaseAudit'])
        self.assertEqual(len(executed),1);self.assertIsNot(executed[0],vars(base));self.assertIs(executed[0]['CaseAudit'],cls)

if __name__=='__main__':unittest.main()
