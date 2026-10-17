"""Synthetic audit mechanisms only; no device, models, services or real ledger."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e028_audit_cpu',HERE/'audit_model.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
CONFIG={'vocab_size':248320,'layer_types':['full_attention' if i%4==3 else 'linear_attention' for i in range(64)]}
FIXTURES=[{'fixture_id':'f'+str(i),'prompt_ids':[1]*10} for i in range(8)]


def dumps(x):return (json.dumps(x)+'\n').encode()


def arrays(label,prefix,B,L=768):
    out=[]
    for i,kind in enumerate(CONFIG['layer_types']):
        fields={'K':[B,4,L,256],'V':[B,4,L,256]} if kind=='full_attention' else {'conv':[B,3,10240],'S':[B,48,128,128]}
        for field,shape in fields.items():
            out.append(dict(check=label,name=f'{prefix}/{i}/{field}',shape=shape,
                            dtype='mlx.core.float32' if field=='S' else 'mlx.core.bfloat16',
                            bit_equal=True,reference_finite=True,candidate_finite=True))
    return out


def fixture(case='D-C8',step=3,c8_steps=20,retire=False):
    base=a.source_auditor();prefix='probe-' if case=='lifecycle' else 'c8-'
    old,new=prefix+'0',prefix+'1';decode=[old,new]
    event=dict(event='identity-refill',status='PASS',observer_step=step+(c8_steps if case=='lifecycle' else 0),
               transition=dict(case=case,old_ids=[old],old_lens=[11],keep=[0],new_ids=[new],decode_ids=decode,capacity=768),
               candidate_witness=dict(before=dict(identity_refill=0,fallback=1),after=dict(identity_refill=1,fallback=1),
                                      path='identity_refill',reason='identity_keep_unchanged_capacity'),arrays=128)
    source=arrays('before-step-state','root/committed',1)
    for label in ('reference/committed-not-rewritten-by-ahead','candidate/committed-not-rewritten-by-ahead','before-retirement-state'):
        source+=arrays(label,'root' if label.startswith(('reference/','candidate/')) else 'root/committed',2)
    source+=arrays('after-retirement-state','root/committed',1 if retire else 2)
    extra=arrays(a.PREFIX+repr(decode),'root/layers',2)
    row=dict(schedule=dict(decode=decode,requests=[dict(rid=r,prompt_ids=[1]*10,sampling=dict(max_tokens=512)) for r in decode]),
             arrays=source+extra,reference_delivered=[[old,99,True]] if retire else [],
             after=dict(engine=dict(batch_rids=[new] if retire else decode,batch_lens=[11] if retire else [12,11]),
                        state={'arrays':[{'array':x['name']} for x in arrays('', 'root/committed',1 if retire else 2)]}))
    return base,event,row


class Tests(unittest.TestCase):
    def checker(self,base,event,row,case='D-C8',step=3,c8_steps=20):
        stream=a.Refills(io.BytesIO(dumps(event)))
        stream.counters[1]['fallback']=1
        cls=a.extended_case(base,stream,c8_steps);state=cls(case,FIXTURES,CONFIG,[99])
        state.old_ids=event['transition']['old_ids'].copy();state.old_lens=[11];state.capacity=768;state.step=step
        state.previous_batch=1;state.previous_arrays=state.cache_names('root/committed')
        return stream,state

    def test_all128_extra_arrays_and_original_groups_are_checked_without_mutation(self):
        base,event,row=fixture();original=copy.deepcopy(row)
        stream,state=self.checker(base,event,row)
        state.check_arrays(row,[],set(),False,2)
        self.assertEqual(row,original)
        self.assertEqual(stream.lines,1);self.assertEqual(state.array_counts['e028_refill_arrays'],128)
        self.assertEqual(state.array_counts['all_arrays'],768)
        self.assertEqual(stream.counters[1],dict(identity_refill=1,fallback=1))
        self.assertEqual(stream.digest.hexdigest(),hashlib.sha256(dumps(event)).hexdigest())

    def test_shape_dtype_flags_names_and_group_coverage_corruptions_fail(self):
        for mutation in ('batch','capacity','s-rank','dtype','bit','finite','missing','duplicate','label','unexpected'):
            base,event,row=fixture();extra=[x for x in row['arrays'] if x['check'].startswith(a.PREFIX)]
            if mutation=='batch':extra[0]['shape'][0]=1
            if mutation=='capacity':next(x for x in extra if x['name'].endswith('/K'))['shape'][2]=1024
            if mutation=='s-rank':next(x for x in extra if x['name'].endswith('/S'))['shape'].pop(0)
            if mutation=='dtype':extra[0]['dtype']='mlx.core.float32'
            if mutation=='bit':extra[0]['bit_equal']=False
            if mutation=='finite':extra[0]['candidate_finite']=1
            if mutation=='missing':row['arrays'].pop()
            if mutation=='duplicate':extra[0]['name']=extra[1]['name']
            if mutation=='label':extra[0]['check']=a.PREFIX+"['c8-1', 'c8-0']"
            if mutation=='unexpected':row['arrays'].append({**extra[0],'check':'unrecognized extra'})
            stream,state=self.checker(base,event,row)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):state.check_arrays(row,[],set(),False,2)

    def test_original_checker_still_rejects_omitted_original_group_and_bad_finite(self):
        for corruption in ('omit','finite','capacity'):
            base,event,row=fixture()
            if corruption=='omit':row['arrays']=[x for x in row['arrays'] if x['check']!='before-retirement-state']
            if corruption=='finite':row['arrays'][0]['reference_finite']=False
            if corruption=='capacity':next(x for x in row['arrays'] if x['check']=='before-step-state' and x['name'].endswith('/K'))['shape'][2]=1024
            _,state=self.checker(base,event,row)
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):state.check_arrays(row,[],set(),False,2)

    def test_transition_and_counter_witness_cannot_be_self_reported(self):
        for corruption in ('old-id','old-lens','new-id','keep','capacity','step','fallback','stale','extra-field'):
            base,event,row=fixture();stream,state=self.checker(base,event,row)
            bad=copy.deepcopy(event)
            if corruption=='old-id':bad['transition']['old_ids']=['c8-9']
            if corruption=='old-lens':bad['transition']['old_lens']=[12]
            if corruption=='new-id':bad['transition']['new_ids']=['c8-9']
            if corruption=='keep':bad['transition']['keep']=[]
            if corruption=='capacity':bad['transition']['capacity']=1024
            if corruption=='step':bad['observer_step']=2
            if corruption=='fallback':bad['candidate_witness']['after']['fallback']=2
            if corruption=='stale':bad['candidate_witness']['after']['identity_refill']=0
            if corruption=='extra-field':bad['trusted']=True
            stream.stream=io.BytesIO(dumps(bad))
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):state.check_arrays(row,[],set(),False,2)

    def test_lifecycle_observer_step_includes_c8_steps_and_retirement_is_fallback(self):
        base,event,row=fixture('lifecycle',retire=True)
        stream,state=self.checker(base,event,row,'lifecycle')
        state.check_arrays(row,[],set(),False,2)
        self.assertEqual(event['observer_step'],23)
        self.assertEqual(stream.counters[1],dict(identity_refill=1,fallback=2))
        self.assertEqual(state.old_ids,['probe-1']);self.assertEqual(state.capacity,768)

    def test_no_group_on_growth_or_no_new_and_no_unmatched_events(self):
        for case in ('growth','no-new'):
            base,event,row=fixture()
            if case=='growth':row['schedule']['requests'][1]['sampling']['max_tokens']=1000
            else:row['schedule']['decode']=['c8-0']
            _,state=self.checker(base,event,row)
            with self.subTest(case=case),self.assertRaisesRegex(ValueError,'without eligible'):state.check_arrays(row,[],set(),False,2)
        refills=a.Refills(io.BytesIO(dumps(event)))
        with self.assertRaisesRegex(ValueError,'unmatched'):refills.finish({},base.same)

    def test_stream_rejects_missing_truncated_duplicate_events_and_wrong_coverage(self):
        base,event,row=fixture()
        for data in (b'',dumps(event)[:-1]):
            stream,state=self.checker(base,event,row);stream.stream=io.BytesIO(data)
            with self.assertRaisesRegex(ValueError,'missing or truncated'):state.check_arrays(row,[],set(),False,2)
        stream,state=self.checker(base,event,row);stream.stream=io.BytesIO(dumps(event)*2)
        state.check_arrays(row,[],set(),False,2)
        with self.assertRaisesRegex(ValueError,'unmatched'):stream.finish({},base.same)
        stream,state=self.checker(base,event,row);state.check_arrays(row,[],set(),False,2)
        coverage=dict(candidate_installs=2,paired_identity_refills=1,compared_arrays=128,per_case={'D-C8':1},
                      restored_observers=4,candidate_final_counters=copy.deepcopy(stream.counters))
        self.assertEqual(stream.finish(coverage,base.same)['events'],1)
        for key,value in (('restored_observers',3),('per_case',{'lifecycle':1}),('compared_arrays',127)):
            bad={**coverage,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):stream.finish(bad,base.same)

    def test_missing_terminal_and_incomplete_do_not_open_ledgers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.assertEqual(a.audit(root)['status'],'INCOMPLETE')
            (root/'base').mkdir()
            for name in ('complete.json','plan.json','base/complete.json','base/plan.json'):
                (root/name).write_text(json.dumps({'status':'INCOMPLETE'}))
            for name in ('refills.jsonl','base/steps.jsonl'):(root/name).write_text('MUST NOT BE PARSED')
            result=a.audit(root)
            self.assertEqual(result['status'],'INCOMPLETE',result);self.assertFalse(result['qualified'] or result['ledger_opened'])
            with self.assertRaisesRegex(ValueError,'streamed'):a.read(root/'refills.jsonl')

    def test_real_small_plan_source_and_result_binding_rejects_corruption(self):
        # Actual pinned plans/small sources/config only. The result is explicitly
        # synthetic and no model tensor, runtime binary or ledger is read.
        base=a.source_auditor()
        plan=HERE/'prereg-model-v1/qualification-plan.json'
        child_plan=a.read(plan);numeric=a.read(child_plan['qualification']['path'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'base').mkdir()
            (root/'plan.json').write_bytes(plan.read_bytes())
            (root/'base/plan.json').write_bytes(Path(child_plan['qualification']['path']).read_bytes())
            result=dict(status='PASS',plan_sha256=a.BASE_PLAN_SHA,qualifier_sha256=numeric['execution']['qualifier']['sha256'],
                        numeric_only_not_http=True,source_model_pins_rechecked=True,total_requests=37,candidate=numeric['candidate'])
            (root/'base/complete.json').write_bytes(dumps(result))
            top=dict(schema='E028-model-qualification-result-v1',status='PASS',plan_sha256=a.PLAN_SHA,sources=child_plan['sources'],
                     source_pins_rechecked=True,numerical_only_not_http=True,performance_qualified=False,total_requests=37,
                     base_result=dict(path='base/complete.json',sha256=a.sha(root/'base/complete.json'),status='PASS',returncode=0))
            (root/'complete.json').write_bytes(dumps(top))
            actual=a.binding(root,base)
            self.assertEqual(len(actual[2]),8);self.assertEqual(actual[3]['vocab_size'],248320)
            for key,value in (('plan_sha256','0'*64),('source_pins_rechecked',False),('total_requests',36),('performance_qualified',True)):
                (root/'complete.json').write_bytes(dumps({**top,key:value}))
                with self.subTest(key=key),self.assertRaises(ValueError):a.binding(root,base)
            (root/'complete.json').write_bytes(dumps(top))
            (root/'base/complete.json').write_bytes(dumps({**result,'source_model_pins_rechecked':False}))
            with self.assertRaisesRegex(ValueError,'base result SHA'):a.binding(root,base)

    def test_audit_route_preserves_fail_and_does_not_grant_pass_on_stream_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'base').mkdir()
            for name in ('complete.json','plan.json','base/complete.json','base/plan.json'):(root/name).write_text('{}')
            for name in ('refills.jsonl','base/steps.jsonl'):(root/name).write_text('CPU_FAKE')
            base=a.source_auditor()
            claimed={'cases':[dict(name=n,steps=1) for n in base.CASES]}
            with patch.object(a,'binding',return_value=({},claimed,FIXTURES,CONFIG,[99])), \
                 patch.object(a,'source_auditor',return_value=base), \
                 patch.object(base,'stream_ledger',side_effect=ValueError('old full-vocabulary/state gate failed')):
                result=a.audit(root)
            self.assertEqual(result['status'],'FAIL');self.assertFalse(result['qualified']);self.assertTrue(result['ledger_opened'])
            self.assertIn('full-vocabulary',result['first_failure']['message'])

    def test_private_frozen_auditor_selftest_and_no_measurement_imports(self):
        import builtins,sys
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ('mlx','numpy','torch','tutorial_code'):raise AssertionError(name)
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',guarded):
            base=a.source_auditor();original_case=base.CaseAudit;base.self_test()
            other=a.source_auditor();a.extended_case(base,a.Refills(io.BytesIO()),20)
            self.assertIs(base.CaseAudit,original_case);self.assertIsNot(other.CaseAudit,original_case)
        self.assertFalse(any(x=='mlx' or x.startswith('mlx.') for x in sys.modules))


if __name__=='__main__':unittest.main()
