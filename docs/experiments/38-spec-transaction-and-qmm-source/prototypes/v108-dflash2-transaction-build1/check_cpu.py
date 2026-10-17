"""Pure stdlib transaction/API/source checks; no MLX or NumPy imports."""
import ast
import copy
import importlib.util
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
sys.dont_write_bytecode = True
D=Path(__file__).resolve().parent; P=D.parent
C=P/'candidate-v108-dflash2-transaction-build1-root/tutorial_code/miniengine'
A=P/'candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine'
spec=importlib.util.spec_from_file_location('tx108_cpu',C/'spec/transaction.py')
tx=importlib.util.module_from_spec(spec);spec.loader.exec_module(tx)

class Array:
    def __init__(self,shape,dtype='bf16',data=None):
        self.shape,self.dtype=tuple(shape),dtype
        self.data=[0]*__import__('math').prod(shape) if data is None else list(data)
    def _indices(self,key):
        key=key if isinstance(key,tuple) else (key,)
        key=key+(slice(None),)*(len(self.shape)-len(key));axes=[];shape=[]
        for item,size in zip(key,self.shape):
            if isinstance(item,int):axes.append([item])
            else:
                axis=list(range(size))[item];axes.append(axis);shape.append(len(axis))
        strides=[__import__('math').prod(self.shape[i+1:]) for i in range(len(self.shape))]
        return [sum(i*s for i,s in zip(index,strides)) for index in itertools.product(*axes)],shape
    def __getitem__(self,key):
        ids,shape=self._indices(key);return Array(shape,self.dtype,[self.data[i] for i in ids])
    def __setitem__(self,key,value):
        ids,shape=self._indices(key);assert tuple(shape)==value.shape
        self.data=self.data[:]
        for i,v in zip(ids,value.data):self.data[i]=v

class Check(unittest.TestCase):
    def test_delivery_frontier_budget_EOS(self):
        cases=[([10,11],[90,91,92],4,(),[90],0,1),
               ([10,11],[10,90,92],4,(),[10,90],1,2),
               ([10,11],[10,11,12],3,(),[10,11,12],2,3),
               ([10,11],[10,11,12],1,(),[10],2,1),
               ([10,11],[10,11,12],3,(10,),[10],2,1),
               ([10,11],[10,11,12],3,(11,),[10,11],2,2),
               ([10,11],[10,11,12],3,(12,),[10,11,12],2,3)]
        for draft,post,budget,eos,out,n,keep in cases:
            self.assertEqual(tx.select_delivery(draft,post,budget,eos), (out,n,keep))
            self.assertEqual(keep,len(out))
        with self.assertRaises(ValueError):tx.select_delivery([], [10],0)
        with self.assertRaises(ValueError):tx.select_delivery([1],[1],1)

    def test_literal_GDN_history_all_prefixes_and_abort(self):
        for keep in range(6):
            c=NS(S=object(),conv_state=object(),offset=19,record=False,tape=None,_snap=None)
            start=(c.conv_state,c.S)
            with tx.VerifyTransaction([c],5) as t:
                for i in range(5):
                    conv,state,pc,ps=object(),object(),object(),object()
                    t.traces[0].record_gdn(conv,state,pc,ps)
                    self.assertIs(t.traces[0].history[-1][1],state)
                    self.assertIs(t.traces[0].producer_owners[-1][1],ps)
                expected=t.witness(keep)[0]
                t.commit(keep)
                self.assertEqual((c.conv_state,c.S),expected[0]);self.assertEqual(c.offset,19+keep)
                self.assertIsNone(c._verify108);self.assertFalse(c.record)
                with self.assertRaises(RuntimeError):t.commit(keep)
            if keep==0:self.assertEqual((c.conv_state,c.S),start)
        c=NS(S=object(),conv_state=object(),offset=20,record=False,tape=None,_snap=None);before=(c.conv_state,c.S,c.offset)
        with self.assertRaisesRegex(RuntimeError,'producer failure'):
            with tx.VerifyTransaction([c],3):raise RuntimeError('producer failure')
        self.assertEqual((c.conv_state,c.S,c.offset),before);self.assertIsNone(c._verify108)

    def test_same_batched_KV_prefix_bytes_tail_and_unchanged_owners(self):
        mx=NS(zeros=lambda shape,dtype:Array(shape,dtype))
        for keep in range(6):
            k=Array((1,1,10,2),data=list(range(20)));v=Array((1,1,10,2),data=list(range(20,40)))
            c=NS(keys=k,values=v,offset=2,step=10);before=(k.data[:],v.data[:])
            updates=Array((1,1,5,2),data=list(range(100,110)))
            with tx.VerifyTransaction([c],5) as t:
                t.traces[0].attention_update(updates,updates,mx)
                histories=[tuple(a.data[:]) for a,b in t.traces[0].history]
                t.commit(keep)
                self.assertEqual(c.offset,2+keep)
                self.assertEqual(tuple(c.keys.data),histories[keep])
                self.assertEqual((k.data,v.data),before)
                if keep:
                    self.assertEqual(c.keys.data[:4],k.data[:4])
                    self.assertEqual(c.keys.data[4:4+2*keep],updates.data[:2*keep])
                    self.assertEqual(c.keys.data[4+2*keep:],[0]*(16-2*keep))
        c=NS(keys=Array((1,1,3,2)),values=Array((1,1,3,2)),offset=2,step=3)
        with tx.VerifyTransaction([c],2) as t:
            with self.assertRaises(ValueError):t.traces[0].attention_update(Array((1,1,2,2)),Array((1,1,2,2)),mx)

    def test_atomic_guard_and_partial_producer_failure(self):
        a=NS(S=object(),conv_state=object(),offset=19,record=False,tape=None,_snap=None)
        b=NS(S=object(),conv_state=object(),offset=19,record=False,tape=None,_snap=None)
        before=(a.S,b.S)
        with tx.VerifyTransaction([a,b],2) as t:
            for i in range(2):t.traces[0].record_gdn(object(),object(),object(),object())
            with self.assertRaises(RuntimeError):t.commit(1)
            self.assertEqual((a.S,b.S),before)
        self.assertEqual((a.S,b.S),before)
        for n in [0,9,True]:
            with self.assertRaises(ValueError):tx.VerifyTransaction([a],n)
        with self.assertRaises(ValueError):tx.VerifyTransaction([a,a],2)

    def test_original_AST_restoration_defaultOFF(self):
        def normalize(path):
            tree=ast.parse(path.read_text())
            for cls in [n for n in tree.body if isinstance(n,ast.ClassDef)]:
                cls.body=[n for n in cls.body if not(isinstance(n,ast.FunctionDef) and n.name in ['_prefill_verify108','set_verify_transaction','_step_transaction108'])]
                for fn in [n for n in cls.body if isinstance(n,ast.FunctionDef)]:
                    if cls.name=='AttnCache' and fn.name=='update':fn.body=fn.body[:1]+fn.body[3:]
                    if cls.name=='GatedDeltaNet' and fn.name=='_prefill':fn.body=fn.body[:1]+fn.body[2:]
                    if cls.name=='SpecRunner' and fn.name=='__init__':
                        fn.body=[n for n in fn.body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='set_verify_transaction')]
                    if cls.name=='SpecRunner' and fn.name=='step':fn.body=fn.body[:1]+fn.body[2:]
            return ast.dump(tree,include_attributes=False)
        for rel in ['cache.py','model/gdn.py','engine/spec_runner.py']:
            self.assertEqual(normalize(C/rel),ast.dump(ast.parse((A/rel).read_text()),include_attributes=False),rel)
        for path in A.rglob('*.py'):
            rel=path.relative_to(A)
            if str(rel) not in ['cache.py','model/gdn.py','engine/spec_runner.py']:
                self.assertEqual(path.read_bytes(),(C/rel).read_bytes())
        s=(C/'model/gdn.py').read_text();self.assertIn('raw = self.in_proj_qkv(x)',s)
        s=(C/'spec/transaction.py').read_text();self.assertNotIn('gated_delta_rule_chunk(',s)
        self.assertIn('trace.record_gdn(next_conv[0], next_state[0], next_conv, next_state)',s)
        self.assertIn('self.set_verify_transaction(False)',(C/'engine/spec_runner.py').read_text())

    def test_syntax_and_no_imported_MLX(self):
        for p in list(C.rglob('*.py'))+list(D.glob('*.py')):compile(ast.parse(p.read_text()),str(p),'exec')
        source=(D/'fullqual.py').read_text()
        self.assertIn('gdns=[layer.linear_attn for layer in engine.model.layers if not layer.is_attn]',source)
        self.assertNotIn('layer.mixer',source)
        self.assertNotIn('expected_IDs',(C/'engine/spec_runner.py').read_text())
        self.assertIn('emitted==expected',source)
        self.assertIn('mx.eval(posterior, new_feats, *transaction.eval_arrays())',(C/'engine/spec_runner.py').read_text())
        for p in (D/'tiny.py',D/'fullqual.py',D/'runtime_support.py'):
            for node in ast.parse(p.read_text()).body:
                if isinstance(node,ast.Import):self.assertFalse(any(a.name.startswith(('mlx','numpy')) for a in node.names))
                if isinstance(node,ast.ImportFrom):self.assertFalse((node.module or '').startswith(('mlx','numpy')))
        self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules))

if __name__=='__main__':unittest.main()
