"""CPU-only guards for deferred diagnostic semantics and actual-input provenance."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
from op_audit import Hooks, prefix_lengths, summarize
D = Path(__file__).resolve().parent


class Tensor:
    def __init__(self, shape):
        self.shape = shape;self.ndim = len(shape);self.dtype = 'fake_bf16'
    def __getitem__(self, item):
        return Tensor((1, *self.shape[1:]))


class Tests(unittest.TestCase):
    def test_causal_right_alignment(self):
        self.assertEqual(prefix_lengths(5, 24), [20,21,22,23,24])
        self.assertEqual(prefix_lengths(1, 20), [20])
        for args in ((0,20),(5,4),(-1,2)):
            with self.assertRaises(ValueError):prefix_lengths(*args)

    def test_hooks_preserve_output_and_defer_alternatives(self):
        calls=[]
        class Op:
            q=None;weight=Tensor((8,16))
            def __call__(self,x,*args):
                calls.append(x);return original_output
        class Linear(Op):pass
        class RMS(Op):pass
        class Gated(Op):pass
        class Embed(Op):
            def as_linear(self,x):return original_output
        class GDN:
            def _gates(self,b,a):return b,a
        op=Linear();original_output=Tensor((5,8));x=Tensor((5,16))
        original_method=Linear.__call__
        fake=lambda *a,**k: a[0]
        mx=SimpleNamespace(sigmoid=fake,fast=SimpleNamespace(rope=fake,scaled_dot_product_attention=fake),concatenate=lambda xs,axis:xs[0])
        nn=SimpleNamespace(silu=fake,softplus=fake)
        model=SimpleNamespace(named_modules=lambda:[('layers.0.linear_attn.in_proj_qkv',op)])
        layers=SimpleNamespace(Linear=Linear,Embedding=Embed,RMSNorm=RMS,GatedRMSNorm=Gated)
        hooks=Hooks(mx,nn,model,layers,SimpleNamespace(GatedDeltaNet=GDN));hooks.T=5;hooks.install();hooks.active=True
        try:
            actual=op(x)
            self.assertIs(actual,original_output);self.assertEqual(len(calls),1)
            self.assertEqual(len(hooks.records),1)
            self.assertTrue(callable(hooks.records[0]['build']))
            hooks.active=False;hooks.records[0]['build']()
            self.assertEqual(len(calls),6)
        finally:hooks.close()
        self.assertIs(Linear.__call__,original_method)

    def test_summary_keeps_first_strict_difference(self):
        rows=[dict(order=0,kind='RMSNorm',variant='row_M1',strict_equal=True,different_elements=0),
              dict(order=1,kind='RMSNorm',name='layer.0.norm',component=0,first_difference={'index':[0,0]},variant='row_M1',strict_equal=False,different_elements=2)]
        s=summarize(rows)
        self.assertEqual(s['first_differing_captured_operation']['order'],1)
        self.assertEqual(s['by_kind']['RMSNorm:row_M1']['strict_different'],1)
        self.assertEqual(s['first_difference_by_kind']['RMSNorm:row_M1']['order'],1)

    def test_actual_saved_drafts_and_no_gold_feedback(self):
        b=json.loads((D/'actual-block.json').read_text())
        self.assertEqual(b['drafts'],[6297,220,95761,95779]);self.assertEqual(b['verify_inputs'],5)
        self.assertEqual(len(b['initial_cache']),64)
        self.assertEqual(len(b['prompt_ids']),19)
        tree=ast.parse((D/'diagnose.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        text=ast.unparse(main)
        self.assertIn("mx.array([anchor] + block['drafts'])",text)
        self.assertIn("anchor = int(anchor_array.item())",text)
        self.assertLess(text.index('mx.eval(posterior, new_features, *transaction.eval_arrays())'),text.index("rec['build']()"))
        self.assertNotIn('load_drafter',text)

    def test_import_is_inert_and_frozen_candidate_not_written(self):
        spec=importlib.util.spec_from_file_location('v112diagnose',D/'diagnose.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules))
        source=(D/'diagnose.py').read_text()
        self.assertNotIn('ROOT.mkdir',source);self.assertNotIn('set_verify_transaction(',source)
        for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')


if __name__=='__main__':unittest.main()
