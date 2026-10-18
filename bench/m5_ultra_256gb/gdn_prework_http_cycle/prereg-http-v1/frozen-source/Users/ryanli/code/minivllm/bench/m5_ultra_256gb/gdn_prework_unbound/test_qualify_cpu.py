"""Canary mechanism tests only; source prepared, NOT EXECUTED during timing.

No fake numeric claim: these test token causality, independent state threading,
instance restoration and rejection mechanisms. Real BF16/FP32 bits await GPU.
"""
import builtins
import gc
from pathlib import Path
import runpy
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch
import weakref

from bench.m5_ultra_256gb.gdn_prework_unbound import qualify as q


class Layer:
    def _decode(self,x,state,**flags):
        state.conv += x; state.S += 2*x
        return state.S


class Tests(unittest.TestCase):
    def test_source_import_never_imports_device(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='mlx' or name.startswith('mlx.') or name.startswith('tutorial_code'):
                raise AssertionError('device/model import during CPU source import')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded): runpy.run_path(q.__file__)

    def test_original_fixture_selection_and_no_mutation(self):
        fixtures=[dict(fixture_id=f'd{i}',prompt_ids=[i]*(886+i*13),max_output_tokens=512,normal_eos=True) for i in range(8)]
        before=[list(f['prompt_ids']) for f in fixtures]
        chosen=q.choose_cases(fixtures)
        self.assertEqual([(f['fixture_id'],start) for f,start in chosen],[('d0',1),('d7',129)])
        self.assertIs(chosen[0][0],fixtures[0]); self.assertEqual(before,[f['prompt_ids'] for f in fixtures])
        for key,value in (('normal_eos',False),('max_output_tokens',True),('max_output_tokens',513)):
            wrong=[dict(f) for f in fixtures]; wrong[3][key]=value
            with self.assertRaises(ValueError):q.choose_cases(wrong)

    def test_late_window_uses_actual_predecessor_tokens_without_skipping(self):
        calls=[]; events=[]
        def advance(token,step,active):
            calls.append((token,step,active)); return token+1,dict(actual_token=token,step=step)
        raw,rows=q.rollout_window(0,{9999},512,129,advance,events.append)
        self.assertEqual(raw,list(range(132)))
        self.assertEqual(len(calls),131)
        self.assertEqual([x for x in calls if x[2]],[(128,129,True),(129,130,True),(130,131,True)])
        self.assertEqual([r['step'] for r in rows],[129,130,131])
        self.assertEqual(len(events),131)

    def test_early_eos_is_incomplete_without_advancing_eos(self):
        calls=[]; events=[]
        def advance(token,step,active): calls.append(token); return token+1,{}
        with self.assertRaises(q.IncompleteCoverage):q.rollout_window(4,{6},512,129,advance,events.append)
        self.assertEqual(calls,[4,5]);self.assertEqual(events[-1]['event'],'coverage-shortfall')
        self.assertEqual(events[-1]['raw_ids'],[4,5,6]);self.assertTrue(events[-1]['normal_eos'])
        calls.clear()
        with self.assertRaises(q.IncompleteCoverage):q.rollout_window(6,{6},512,1,advance,events.append)
        self.assertEqual(calls,[])

    def test_eos_on_final_captured_output_completes_local_window(self):
        raw,records=q.rollout_window(3,{6},512,1,lambda t,s,a:(t+1,s),lambda row:None)
        self.assertEqual(raw,[3,4,5,6]);self.assertEqual(records,[1,2,3])

    def test_capture_uses_original_returns_and_restores_on_exception(self):
        layer=Layer(); module=SimpleNamespace(GatedDeltaNet=Layer)
        with self.assertRaisesRegex(RuntimeError,'caller'):
            with q.DecodeCapture({32:layer},module,dict) as scope:
                state=SimpleNamespace(conv=3,S=9);scope.begin(129,True)
                result=layer._decode(5,state,_compile_gates=True,_norm_reuse_normal=True)
                row=scope.finish()[32]
                self.assertEqual(result,19);self.assertEqual(row['input'],dict(x=5,conv=3,S=9))
                self.assertEqual(row['output'],dict(output=19,conv=8,S=19))
                scope.begin(130,False);self.assertEqual(layer._decode(1,state),21);self.assertEqual(scope.finish(),{})
                raise RuntimeError('caller')
        self.assertNotIn('_decode',vars(layer));self.assertTrue(scope.closed);self.assertEqual(scope.bindings,[])
        wr=weakref.ref(layer);del layer;gc.collect();self.assertIsNone(wr())

    def test_capture_rejects_missing_duplicate_and_nonordinary_calls(self):
        layer=Layer();module=SimpleNamespace(GatedDeltaNet=Layer)
        with q.DecodeCapture({0:layer},module,dict) as scope:
            scope.begin(1,True)
            with self.assertRaisesRegex(ValueError,'complete'):scope.finish()
            with self.assertRaisesRegex(ValueError,'ordinary'):layer._decode(1,SimpleNamespace(conv=0,S=0))
            layer._decode(1,SimpleNamespace(conv=0,S=0),_compile_gates=True,_norm_reuse_normal=True)
            with self.assertRaisesRegex(ValueError,'ordinary'):
                layer._decode(1,SimpleNamespace(conv=0,S=0),_compile_gates=True,_norm_reuse_normal=True)
            scope.finish()

    def test_capture_partial_install_failure_restores_already_installed_instance(self):
        first=Layer();second=Layer();second._decode=MethodType(lambda self,*a,**kw:None,second)
        with self.assertRaisesRegex(ValueError,'untouched'):
            q.DecodeCapture({0:first,32:second},SimpleNamespace(GatedDeltaNet=Layer),dict)
        self.assertNotIn('_decode',vars(first));self.assertIn('_decode',vars(second))

    def test_three_steps_continue_candidate_state_not_reference_reinitialization(self):
        records=[dict(x=1,before=0,after=1),dict(x=2,before=1,after=3),dict(x=3,before=3,after=6)]
        initialized=[];calls=[];comparisons=[]
        def initialize(record):
            state=SimpleNamespace(value=record['before']); initialized.append(state); return state
        def invoke(layer,x,state):
            calls.append((id(state),state.value));state.value+=x;return state.value
        def compare(kind,n,value,expected):
            comparisons.append((kind,n))
            if kind=='output':self.assertEqual((value[0],value[1].value),(expected['after'],expected['after']))
            else:self.assertEqual(value,expected['before'])
        q.replay_chain(None,records,initialize=initialize,input_at=lambda r:(r['x'],r),invoke=invoke,
                       observe=lambda x,st:st.value,compare=compare)
        self.assertEqual(len(initialized),1);self.assertEqual([x[1] for x in calls],[0,1,3])
        self.assertEqual(len({x[0] for x in calls}),1);self.assertEqual(len(comparisons),9)

    def test_wrong_continuation_state_rejects_before_next_candidate_call(self):
        records=[dict(x=1,before=0),dict(x=2,before=88),dict(x=3,before=3)];calls=[]
        def compare(kind,n,value,expected):
            if kind!='output':q.need(value==expected['before'],'state causal mismatch')
        def invoke(layer,x,state):calls.append(x);state.value+=x;return state.value
        with self.assertRaisesRegex(ValueError,'causal'):
            q.replay_chain(None,records,initialize=lambda r:SimpleNamespace(value=r['before']),
                input_at=lambda r:(r['x'],r),invoke=invoke,observe=lambda x,st:st.value,compare=compare)
        self.assertEqual(calls,[1])

    def test_stale_counter_or_fallback_cannot_claim_actual_candidate(self):
        state=SimpleNamespace(conv='conv',S='S');evaluated=[]
        for mode in ('fallback','stale','candidate'):
            handle=SimpleNamespace(calls=dict(candidate=9,fallback=3))
            def invoke(x,st,**flags):
                self.assertEqual(flags,dict(_compile_gates=True,_norm_reuse_normal=True))
                if mode!='stale':handle.calls[mode]+=1
                return 'output'
            layer=SimpleNamespace(_decode=invoke)
            if mode=='candidate':
                self.assertEqual(q.checked_invoke(handle,layer,'x',state,lambda *v:evaluated.append(v)),'output')
            else:
                with self.assertRaisesRegex(ValueError,'stale'):
                    q.checked_invoke(handle,layer,'x',state,lambda *v:evaluated.append(v))
        self.assertEqual(evaluated,[('output','conv','S')]*3)

    def test_evaluation_failure_has_no_fallback_retry(self):
        calls=[];handle=SimpleNamespace(calls=dict(candidate=0,fallback=0))
        def decode(*args,**kwargs):calls.append('candidate');handle.calls['candidate']+=1;return 'out'
        def evaluate(*args):raise RuntimeError('device error')
        with self.assertRaisesRegex(RuntimeError,'device error'):
            q.checked_invoke(handle,SimpleNamespace(_decode=decode),'x',SimpleNamespace(conv=1,S=2),evaluate)
        self.assertEqual(calls,['candidate']);self.assertEqual(handle.calls['fallback'],0)


if __name__=='__main__':unittest.main()
