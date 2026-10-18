"""Pure stdlib mechanism tests; source prepared, not run during formal timing.

These tests never qualify an MLX operation, model or actual verifier result.
"""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().with_name('qualify.py')
spec=importlib.util.spec_from_file_location('_exact_bridge_cpu_subject',PATH)
q=importlib.util.module_from_spec(spec); spec.loader.exec_module(q)


class Array:
    def __init__(self,name): self.name=name
    def __getitem__(self,index):
        if index is not None: raise AssertionError('no payload slicing/casting in representation helper')
        return ('singleton-batch-view',self.name)


class BridgeTests(unittest.TestCase):
    def test_default_is_inert_and_limits_claims(self):
        output=io.StringIO()
        with patch.object(q,'validate',side_effect=AssertionError('must not validate/run')), \
             patch.object(q,'device',side_effect=AssertionError('must not run')), redirect_stdout(output):
            self.assertEqual(q.main([]),0)
        value=json.loads(output.getvalue())
        self.assertFalse(value['enabled_by_default']); self.assertEqual(value['widths'],[2,4,8])
        self.assertFalse(value['rollback_qualified']); self.assertFalse(value['performance_qualified'])

    def test_partial_cli_refuses_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'absent'
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                q.main(['--out',str(out)])
            self.assertEqual(caught.exception.code,2); self.assertFalse(out.exists())

    def test_teacher_block_is_only_reference_causal_output(self):
        calls=[]; anchors=[]; events=[]
        def advance(token,step):
            calls.append((token,step)); return token+1
        raw,block=q.reference_tokens(100,set(),129,advance,lambda *x:anchors.append(x),events.append)
        self.assertEqual(anchors,[(228,129)])
        self.assertEqual(block,list(range(228,236)))
        self.assertEqual(len(calls),136); self.assertEqual(raw,list(range(100,237)))
        self.assertEqual([x[0] for x in calls],raw[:-1])
        self.assertEqual([e['output_token'] for e in events],raw[1:])

    def test_eos_stops_coverage_but_final_output_eos_is_allowed(self):
        calls=[]
        with self.assertRaises(q.IncompleteCoverage):
            q.reference_tokens(1,{4},1,lambda token,step:(calls.append(token) or token+1),lambda *x:None,lambda x:None)
        self.assertEqual(calls,[1,2,3])
        raw,block=q.reference_tokens(1,{9},1,lambda token,step:token+1,lambda *x:None,lambda x:None)
        self.assertEqual(block,list(range(1,9))); self.assertEqual(raw[-1],9)

    def test_fixture_selection_is_cross_product_without_body_rewrite(self):
        fixtures=[dict(fixture_id=f'D{i}',normal_eos=True,max_output_tokens=512,prompt_ids=[i]*(886+i)) for i in range(8)]
        cases=q.selected(fixtures)
        self.assertEqual([(f['fixture_id'],s) for f,s in cases],[('D0',1),('D0',129),('D7',1),('D7',129)])
        self.assertIs(cases[0][0],fixtures[0]); self.assertEqual(len(cases)*len(q.WIDTHS),12)
        fixtures[0]['max_output_tokens']=True
        with self.assertRaises(ValueError): q.selected(fixtures)

    def test_verify_snapshots_before_single_existing_prefill_call_without_rollback(self):
        events=[]
        class Cache:
            def snapshot(self): events.append('snapshot')
            def rollback(self,*x): raise AssertionError('rollback is a separate gate')
        caches=[Cache(),Cache()]; tokens=object(); expected=object()
        def context(mode,actual):
            self.assertIs(actual,caches); events.append(mode); return ('ctx',mode)
        def model(actual,ctx,**kwargs):
            self.assertIs(actual,tokens); self.assertEqual(ctx,('ctx','prefill')); self.assertEqual(kwargs,{'capture':()})
            events.append('model'); return expected
        self.assertIs(q.verify_forward(model,tokens,caches,context),expected)
        self.assertEqual(events,['snapshot','snapshot','prefill','model'])

    def test_full_state_representation_preserves_entire_kv_handles(self):
        layout=tuple(i%4==3 for i in range(64)); batch=[]; staging=[]
        for i,attn in enumerate(layout):
            a,b=Array(str(i)+'a'),Array(str(i)+'b')
            batch.append(SimpleNamespace(K=a,V=b) if attn else SimpleNamespace(conv=a,S=b))
            staging.append(SimpleNamespace(keys=a,values=b) if attn else SimpleNamespace(conv_state=a,S=b))
        a=q.canonical_state(batch,layout); b=q.canonical_state(staging,layout,staging=True)
        self.assertEqual(sum(len(x) for x in a),128)
        for i,attn in enumerate(layout):
            if attn:
                self.assertIs(a[i]['K'],batch[i].K); self.assertIs(b[i]['K'],staging[i].keys)
            else: self.assertEqual(b[i]['S'],('singleton-batch-view',str(i)+'b'))
        with self.assertRaises(ValueError): q.canonical_state(batch[:-1],layout)

    def test_failed_child_retains_prefix_without_any_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan=root/'plan.json'; plan.write_text('{"qualifier_sha256":"fake","deadline_s":100}')
            out=root/'out'; support=SimpleNamespace(); old=SimpleNamespace(MODEL_SOURCE_SHA='core',SUPPORT_SHA='comparator')
            original={'e023_qualifier':{'path':'comparator.py'}}
            def failed(*args):
                result=args[-1]; directory=args[-4]
                result['cases'].append(dict(width=2,status='FAIL'))
                (directory/'failure.json').write_text('{"label":"full-output/root/logits"}')
                raise RuntimeError('original-bit difference')
            with ExitStack() as stack:
                stack.enter_context(patch.object(q,'validate',return_value=(None,old,support,original,{})))
                stack.enter_context(patch.object(q,'device',side_effect=failed))
                stack.enter_context(patch.object(q.signal,'getitimer',return_value=(0.0,0.0)))
                stack.enter_context(patch.object(q.signal,'getsignal',return_value=None))
                stack.enter_context(patch.object(q.signal,'signal'))
                stack.enter_context(patch.object(q.signal,'setitimer'))
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(q.main(['--plan',str(plan),'--plan-sha256',q.sha(plan),'--out',str(out)]),1)
            result=json.loads((out/'complete.json').read_text())
            self.assertEqual(result['status'],'FAIL'); self.assertEqual(result['cases'],[{'width':2,'status':'FAIL'}])
            self.assertTrue((out/'failure.json').is_file()); self.assertFalse(result['forward_qualified'])
            self.assertFalse(result['rollback_qualified']); self.assertFalse(result['source_model_pins_rechecked'])
            self.assertEqual(result['error']['message'],'original-bit difference')


if __name__=='__main__': unittest.main()
