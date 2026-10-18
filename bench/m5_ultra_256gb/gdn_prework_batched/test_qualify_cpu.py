"""Fake-only batched causality/first-stop checks; not BF16 or device evidence."""
import builtins
import copy
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_prework_batched import qualify as q


class Tests(unittest.TestCase):
    def fixtures(self):
        return [dict(fixture_id=f'd{i}',prompt_ids=[i]*(886+i*13),max_output_tokens=512,normal_eos=True) for i in range(8)]

    def test_import_has_no_device_or_model_imports(self):
        original=builtins.__import__
        def guarded(name,*args,**kw):
            if name.split('.')[0] in ('mlx','mlx_lm','tutorial_code'):raise AssertionError('backend import')
            return original(name,*args,**kw)
        with patch('builtins.__import__',side_effect=guarded):runpy.run_path(q.__file__)

    def test_actual_distinct_fixture_selection_preserves_row_order_and_inputs(self):
        fixtures=self.fixtures();before=copy.deepcopy(fixtures);cases=q.choose_cases(fixtures[::-1])
        self.assertEqual([c['case_id'] for c in cases],['b2-early','b2-late','b8-early','b8-late'])
        self.assertEqual([[f['fixture_id'] for f in c['fixtures']] for c in cases],
            [['d0','d1'],['d6','d7'],[f'd{i}' for i in range(8)],[f'd{i}' for i in range(8)]])
        self.assertEqual(fixtures,before)
        fixtures[-1]['fixture_id']='d0'
        with self.assertRaises(ValueError):q.choose_cases(fixtures)

    def test_late_batch_uses_each_rows_actual_previous_token(self):
        calls=[];events=[]
        def advance(tokens,step,captured):
            calls.append((list(tokens),step,captured));return [tokens[0]+1,tokens[1]+3],dict(step=step)
        raw,records=q.rollout_window([10,20],{9999},512,129,advance,events.append)
        self.assertEqual(raw,[[10+i for i in range(132)],[20+3*i for i in range(132)]])
        self.assertEqual(calls[-3:],[([138,404],129,True),([139,407],130,True),([140,410],131,True)])
        self.assertEqual([r['step'] for r in records],[129,130,131])
        self.assertEqual(events[-1]['input_tokens'],[140,410])

    def test_eight_rows_preserve_independent_causal_order(self):
        first=list(range(8));seen=[]
        def advance(tokens,step,captured):
            seen.append(tokens);return [t+i+1 for i,t in enumerate(tokens)],step
        raw,records=q.rollout_window(first,{9999},512,1,advance,lambda row:None)
        self.assertEqual(first,list(range(8)))
        self.assertEqual(raw,[[i+n*(i+1) for n in range(4)] for i in range(8)])
        self.assertEqual(seen[2],[i+2*(i+1) for i in range(8)])
        self.assertEqual(records,[1,2,3])

    def test_one_early_eos_stops_whole_fixed_batch_without_forced_decode(self):
        called=[];events=[]
        def advance(tokens,step,captured):called.append(tokens);return [tokens[0]+1,9],step
        with self.assertRaises(q.IncompleteCoverage):q.rollout_window([2,4],{9},512,129,advance,events.append)
        self.assertEqual(called,[[2,4]])
        self.assertEqual(events[-1]['ended_rows'],[1]);self.assertEqual(events[-1]['raw_ids'],[[2,3],[4,9]])

    def test_eos_on_final_window_output_does_not_hide_a_missing_call(self):
        called=[]
        def advance(tokens,step,captured):called.append(step);return [tokens[0]+1,tokens[1]+1],step
        raw,records=q.rollout_window([0,10],{3},512,1,advance,lambda row:None)
        self.assertEqual(called,[1,2,3]);self.assertEqual(raw[0],[0,1,2,3]);self.assertEqual(records,[1,2,3])

    def test_invalid_batched_greedy_or_execution_error_stops_without_retry(self):
        for bad in ([True,4],[4],(3,4)):
            calls=[]
            def advance(*args):calls.append(1);return bad,{}
            with self.assertRaises(ValueError):q.rollout_window([1,2],set(),512,1,advance,lambda row:None)
            self.assertEqual(calls,[1])
        calls=[]
        def advance(*args):calls.append(1);raise RuntimeError('actual native execution')
        with self.assertRaisesRegex(RuntimeError,'native execution'):
            q.rollout_window([1,2],set(),512,1,advance,lambda row:None)
        self.assertEqual(calls,[1])

    def test_batched_candidate_state_is_initialized_once_then_continued(self):
        records=[dict(x=[1,10],before=[0,0],after=[1,10]),dict(x=[2,20],before=[1,10],after=[3,30]),
                 dict(x=[3,30],before=[3,30],after=[6,60])]
        initial=[];calls=[]
        def initialize(row):s=SimpleNamespace(values=list(row['before']));initial.append(s);return s
        def invoke(layer,x,state):
            calls.append((id(state),list(state.values)));state.values=[a+b for a,b in zip(state.values,x)];return list(state.values)
        def compare(kind,i,value,expected):
            if kind=='output':self.assertEqual(value[0],expected['after'])
            else:self.assertEqual(value,expected['before'])
        q.replay_chain(None,records,initialize=initialize,input_at=lambda r:(r['x'],r),invoke=invoke,
            observe=lambda x,s:list(s.values),compare=compare)
        self.assertEqual(len(initial),1);self.assertEqual(len({v[0] for v in calls}),1)
        self.assertEqual([v[1] for v in calls],[[0,0],[1,10],[3,30]])

    def test_first_output_mismatch_prevents_second_layer_call(self):
        records=[dict(x=[1,2],before=[0,0])]*3;calls=[]
        def invoke(layer,x,state):calls.append(1);return ['different']
        def compare(kind,*args):
            if kind=='output':raise ValueError('first original-bit difference')
        with self.assertRaisesRegex(ValueError,'original-bit'):
            q.replay_chain(None,records,initialize=lambda r:SimpleNamespace(values=[0,0]),
                input_at=lambda r:(r['x'],r),invoke=invoke,observe=lambda x,s:list(s.values),compare=compare)
        self.assertEqual(calls,[1])

    def test_counter_stale_and_native_fallback_do_not_claim_evaluated_candidate(self):
        for mode in ('stale','fallback','candidate'):
            handle=SimpleNamespace(calls=dict(candidate=1,fallback=0));evaluated=[]
            def decode(*args,**kw):
                if mode!='stale':handle.calls[mode]+=1
                return 'actual'
            invoke=lambda:q.checked_invoke(handle,SimpleNamespace(_decode=decode),'x',SimpleNamespace(conv='conv',S='S'),
                lambda *args:evaluated.append(args))
            if mode=='candidate':self.assertEqual(invoke(),'actual')
            else:
                with self.assertRaisesRegex(ValueError,'stale'):invoke()
            self.assertEqual(evaluated,[('actual','conv','S')])

    def test_plan_rejects_other_candidate_or_extra_fields_before_model_read(self):
        plan=dict(schema='GDN-batched-prework-real-layer-plan-v1',qualifier_sha256=q.sha(q.__file__),
            candidate=dict(path=str(q.HERE/'candidate.py'),sha256=q.CANDIDATE_SHA),e024_plan={},deadline_s=1800)
        with patch.object(q,'load',side_effect=AssertionError('must not load')):
            bad=copy.deepcopy(plan);bad['candidate']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'default-OFF'):q.validate(bad)
            bad=copy.deepcopy(plan);bad['ignore_eos']=True
            with self.assertRaisesRegex(ValueError,'exact batched'):q.validate(bad)
            bad=copy.deepcopy(plan);bad['deadline_s']=True
            with self.assertRaisesRegex(ValueError,'deadline'):q.validate(bad)

    def test_failed_validation_retains_failure_without_device_or_signal_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();plan=root/'plan.json';plan.write_text('{}');out=root/'out'
            with (patch.object(q,'device',side_effect=AssertionError('no device')),
                  patch.object(q.signal,'getitimer',side_effect=AssertionError('no alarm'))):
                code=q.main(['--plan',str(plan),'--plan-sha256',q.sha(plan),'--out',str(out)])
            result=json.loads((out/'complete.json').read_bytes())
            self.assertEqual(code,1);self.assertEqual(result['status'],'FAIL')
            self.assertFalse(result['device_run']);self.assertFalse(result['canary_qualified'])
            self.assertEqual(result['candidate_calls'],0)


if __name__=='__main__':unittest.main()
