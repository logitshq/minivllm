"""CPU-only AST and scope/restore checks; import neither model nor MLX."""
import ast
import copy
from contextlib import contextmanager
from pathlib import Path
import types
import unittest

D=Path(__file__).resolve().parent
P=D.parent
TREE=ast.parse((D/'fullqual.py').read_text())
OLD=ast.parse((P/'v44-native-fullqual-build1/fullqual.py').read_text())
MAIN=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='main')


def nested(name):
    return copy.deepcopy(next(n for n in ast.walk(MAIN) if isinstance(n,ast.FunctionDef) and n.name==name))


def execute_function(node, ns):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<CPU-scope>','exec'),ns)
    return ns[node.name]


class Checks(unittest.TestCase):
    def test_native_boundary_and_old_state_code_unchanged(self):
        for name in ('Native','metadata_finite','originals_readonly','assert_icb'):
            a=next(n for n in OLD.body if getattr(n,'name',None)==name)
            b=next(n for n in TREE.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)
        oldmain=next(n for n in OLD.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        for name in ('shared','ctx_layers','state_of','outputs','setup'):
            old=next(n for n in ast.walk(oldmain) if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(old),ast.dump(nested(name)),name)
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_model' for n in ast.walk(MAIN)),1)

    def test_baseline_carry_OFF_and_GDN_finally_restore(self):
        class Model:
            _residual_carry_enabled=True
            def set_residual_carry(self,x): self._residual_carry_enabled=x
        class GDN: pass
        a,b,legacy,residual=object(),object(),object(),object()
        GDN._decode_dispatch=a;GDN._decode_conv_norm_reuse_cache=b
        model=Model();mod=types.SimpleNamespace(residual_norm=residual);events=[]
        ns={'contextmanager':contextmanager,'GDN':GDN,'candidate_decode':a,'candidate_norm':b,
            'legacy_decode':legacy,'model':model,'residual_module':mod,'candidate_residual':residual,'baseline_calls':events}
        baseline=execute_function(nested('baseline_decode'),ns)
        for fail in (False,True):
            try:
                with baseline():
                    self.assertIs(model._residual_carry_enabled,False)
                    self.assertIs(GDN._decode_dispatch,legacy)
                    self.assertIs(GDN._decode_conv_norm_reuse_cache,legacy)
                    if fail: raise ValueError('deliberate CPU-only exception')
            except ValueError: self.assertTrue(fail)
            self.assertIs(model._residual_carry_enabled,True)
            self.assertIs(GDN._decode_dispatch,a);self.assertIs(GDN._decode_conv_norm_reuse_cache,b)
        self.assertTrue(all(e['carry_during'] is False and e['carry_restored'] for e in events))
        self.assertTrue(all(e['residual_counter_installed'] is False for e in events))

    def test_B_only_hook_is_removed_on_success_and_failure(self):
        class Weight: pass
        w=Weight();calls=[];base=lambda *args:'symbolic-result'
        model=types.SimpleNamespace(_residual_carry_enabled=True)
        mod=types.SimpleNamespace(residual_norm=base)
        ns={'contextmanager':contextmanager,'model':model,'residual_module':mod,'candidate_residual':base,
            'residual_capture_calls':calls,'norm_labels':{id(w):'weight'}}
        capture=execute_function(nested('capture_residual_engagement'),ns)
        value=types.SimpleNamespace(shape=(1,5120),dtype='fake-bfloat16')
        for fail in (False,True):
            try:
                with capture():
                    self.assertEqual(mod.residual_norm(value,value,w,1e-6),'symbolic-result')
                    if fail: raise ValueError('deliberate CPU-only exception')
            except ValueError: self.assertTrue(fail)
            self.assertIs(mod.residual_norm,base)
        self.assertEqual(len(calls),2)
        contexts=[n for n in ast.walk(MAIN) if isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name) and i.context_expr.func.id=='capture_residual_engagement' for i in n.items)]
        self.assertEqual(len(contexts),1)
        text=ast.unparse(contexts[0]);self.assertIn('capture_norm_engagement()',text);self.assertIn('mx.eval(*captured)',text)
        self.assertNotIn('v34_replay_step',text)

    def test_before_token_and_strict_archive_contracts(self):
        text=(D/'fullqual.py').read_text()
        self.assertLess(text.index('hidden-norm-metadata-before-any-token.json'),text.index('warm, warm_token = setup(base_prompt)'))
        self.assertLess(text.index('array_parameter_identity_before_any_token'),text.index('warm, warm_token = setup(base_prompt)'))
        self.assertIn('len(hidden_metadata) == 129 and all(row["fusion_eligible"]',text)
        self.assertIn('len(residual_capture_calls)==128',text)
        self.assertIn('"A_saved_V38_hash_checks":132,"A_saved_V44_hash_checks":132',text)
        self.assertIn('total_steps==93',text);self.assertIn('strict_output_checks=total_steps*132',text)
        self.assertIn('if args.inert:',text)
        self.assertLess(text.index('if args.inert:'),text.index('import mlx.core as mx'))


if __name__=='__main__':unittest.main()
