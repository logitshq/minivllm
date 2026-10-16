"""CPU-only V70 composition checks; no MLX, native dylib or model execution."""
import ast
import copy
import hashlib
from pathlib import Path
import types
import unittest

D=Path(__file__).resolve().parent
P=D.parent
TEXT=(D/'fullqual.py').read_text()
TREE=ast.parse(TEXT)
OLDTEXT=(P/'v64-native-direct-async-runner-build1/fullqual.py').read_text()
OLD=ast.parse(OLDTEXT)
MAIN=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='main')


class Checks(unittest.TestCase):
    def test_old_native_lifecycle_gates_unchanged(self):
        for name in ('assert_pipeline','snapshot_rows'):
            a=next(n for n in TREE.body if getattr(n,'name',None)==name)
            b=next(n for n in OLD.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(a),ast.dump(b))
        start='        requests = {}';end='        r.update(status='
        self.assertEqual(TEXT[TEXT.index(start):TEXT.index(end)],OLDTEXT[OLDTEXT.index(start):OLDTEXT.index(end)])
        self.assertEqual(hashlib.sha256((D/'native_runner.py').read_bytes()).hexdigest(),'acd59a4d817cb73fba07f6757b3a170a057e3c78ea707710624dda0a65e4bc07')

    def test_source_and_beforetoken_contract(self):
        self.assertIn('ROOT = P / "candidate-v65-residual-carry-build1-root"',TEXT)
        self.assertIn('QUAL = P / "v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json"',TEXT)
        self.assertLess(TEXT.index('hidden-norm-metadata-before-any-token.json'),TEXT.index('ordinary = e.generate'))
        self.assertLess(TEXT.index('e.model.set_residual_carry(True)'),TEXT.index('ordinary = e.generate'))
        self.assertIn('len(hidden_metadata)==129',TEXT)
        self.assertIn('len(live_parameters)==1349',TEXT)
        self.assertIn('len(actual_source) == 29',TEXT)
        self.assertIn('if carry_binding["status"] != "PASS_BOUND":',TEXT)
        self.assertIn('assert args.inert,"V65 qualification pending: model execution forbidden"',TEXT)
        self.assertLess(TEXT.index('if args.inert:'),TEXT.index('import mlx.core as mx'))
        compile(TEXT,str(D/'fullqual.py'),'exec')

    def test_capture_wrapper_restores_both_hooks(self):
        node=copy.deepcopy(next(n for n in ast.walk(MAIN) if isinstance(n,ast.FunctionDef) and n.name=='observe_build'))
        for fail in (False,True):
            class GDN: pass
            original_conv=lambda self,*a,**kw:None
            GDN._decode_conv_norm_reuse_cache=original_conv
            original_residual=lambda *a:None
            residual=types.SimpleNamespace(residual_norm=original_residual)
            weights=[object() for _ in range(129)]
            names=[name for i in range(64) for name in (f'layers.{i}.input_layernorm.weight',f'layers.{i}.post_attention_layernorm.weight')]+['norm.weight']
            labels={id(w):name for w,name in zip(weights,names)}
            by_name=dict(zip(names,weights))
            order=[name for i in range(64) for name in (f'layers.{i}.post_attention_layernorm.weight',f'layers.{i+1}.input_layernorm.weight' if i<63 else 'norm.weight')]
            gdns=[GDN() for _ in range(48)];holders=[types.SimpleNamespace(bytes_view=b'x') for _ in range(131)]
            adapter=types.SimpleNamespace(inputs=holders,outputs=[types.SimpleNamespace(bytes_view=b'y') for _ in range(132)],shared=lambda value,label:value)
            calls=[];conv_calls=[];r={};captured={}
            def original_build(values):
                for name in order:residual.residual_norm(None,None,by_name[name],1e-6)
                for g in gdns:GDN._decode_conv_norm_reuse_cache(g)
                if fail:raise ValueError('deliberate CPU-only exception')
            ns={'adapter':adapter,'original_build':original_build,'e':types.SimpleNamespace(model=types.SimpleNamespace(_residual_carry_enabled=True)),
                'fusion_calls':calls,'conv_calls':conv_calls,'residual_module':residual,'GatedDeltaNet':GDN,
                'original_residual':original_residual,'original_conv':original_conv,'norm_labels':labels,'r':r,'gdns':gdns,'captured_originals':captured}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<CPU-build-wrapper>','exec'),ns)
            if fail:
                with self.assertRaises(ValueError):ns['observe_build'](holders)
            else:
                ns['observe_build'](holders);self.assertTrue(r['build_input131_readonly'])
            self.assertIs(residual.residual_norm,original_residual)
            self.assertIs(GDN._decode_conv_norm_reuse_cache,original_conv)
            self.assertEqual(len(calls),128);self.assertEqual(len(conv_calls),48)
            self.assertTrue(r['B_capture_engagement']['counter_removed_after_build'])


if __name__=='__main__':unittest.main()
