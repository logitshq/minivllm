"""Bounded stdlib-only source/behavior checks; never import inference packages."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
D=Path(__file__).resolve().parent;P=D.parent
sys.path.insert(0,str(D))
import adapter

class CPU(unittest.TestCase):
    def fake(self):
        calls=[]
        class G:
            def set_gates_compile(self,v):calls.append(('gates',v))
            def set_conv_norm_reuse_cache(self,v):calls.append(('convnorm',v))
        runner=types.SimpleNamespace(spec=None,_ahead='ordinary-original-ahead')
        e=types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(is_attn=False,linear_attn=G()) for _ in range(48)]),runner=runner,eos_ids={248044,248046})
        class Spec:
            _verify_m1_ops114=tuple(range(497))
            def set_verify_transaction(self,*a,**kw):calls.append(('transaction',a,kw))
            def set_verify_broadcast_m1(self,*a,**kw):calls.append(('broadcast',a,kw))
        def runner_class(*a,**kw):calls.append(('runner',a,kw));return types.SimpleNamespace(spec=Spec())
        def loader(*a,**kw):calls.append(('loader',a,kw));return 'actual-draft-placeholder'
        return e,calls,runner_class,loader
    def test_ordinary_retains_ahead_and_does_not_load_draft(self):
        e,c,r,l=self.fake();original=e.runner
        adapter.assemble(e,'ordinary',ModelRunner=r,load_drafter=l,draft_path='draft')
        self.assertIs(e.runner,original);self.assertEqual(e.runner._ahead,'ordinary-original-ahead')
        self.assertEqual(c.count(('gates',True)),48);self.assertEqual(c.count(('convnorm',True)),48)
        self.assertFalse(any(x[0] in ('loader','runner') for x in c))
    def test_spec_uses_bounded_loader_block5_and_no_observer(self):
        e,c,r,l=self.fake();adapter.assemble(e,'speculative',ModelRunner=r,load_drafter=l,draft_path='draft')
        self.assertIn(('loader',('draft',),{'bounded_load':True}),c)
        self.assertEqual(next(x for x in c if x[0]=='runner')[2],{'spec_block':5})
        self.assertIn(('transaction',(True,),{'eos_ids':e.eos_ids,'capacity':64,'observer':None}),c)
        self.assertIn(('broadcast',(True,),{'logits_observer':None}),c)
    def test_invalid_mode_never_loads(self):
        e,c,r,l=self.fake()
        with self.assertRaises(ValueError):adapter.assemble(e,'bad',ModelRunner=r,load_drafter=l,draft_path='draft')
        self.assertFalse(c)
    def test_pending_binding_stops_before_import(self):
        with self.assertRaisesRegex(RuntimeError,'HOLD'):adapter.build('ordinary',D/'qualification-binding-pending-example.json',execute_gpu=True)
        self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules))
    def test_root_opt_in_required(self):
        with self.assertRaisesRegex(RuntimeError,'execute-gpu'):adapter.build('speculative',D/'qualification-binding.json')
    def test_frozen_request_group_AST(self):
        old=ast.parse((P/'v105-cap64-direct-http-screen-build1/driver.py').read_text());new=ast.parse((D/'client.py').read_text())
        for name in ('request','group'):
            a=next(n for n in ast.walk(old) if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
    def test_imports_and_syntax_inert(self):
        for path in D.glob('*.py'):compile(path.read_text(),str(path),'exec')
        for name in ('server','client'):
            spec=importlib.util.spec_from_file_location('v120_inert_'+name,D/(name+'.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            self.assertEqual(mod.main(['--inert']),0)
        self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules))

if __name__=='__main__':unittest.main()
