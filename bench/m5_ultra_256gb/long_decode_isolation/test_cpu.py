"""CPU mechanism mocks; no real weights, MLX, GPU, or numerical qualification."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('isolation_cpu',HERE/'qualify.py')
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
s = q.load(HERE.parent/'pareto_realistic/qualify_native.py','isolation_frozen_comparator',
    '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf')


class MX:
    bfloat16 = np.float16
    def __getattr__(self,key): return getattr(np,key)
    def eval(self,*args): pass
    def save_safetensors(self,path,arrays): Path(path).write_bytes(b'explicit CPU mock')
    def compile(self,fn,*,inputs):
        self.seen_model = inputs
        self.calls = 0
        def call(*args): self.calls += 1; return fn(*args)
        return call
    def contiguous(self,value): return np.ascontiguousarray(value)


class KV:
    def __init__(self,K,V): self.K,self.V = K,V


class State:
    def __init__(self,conv,S): self.conv,self.S = conv,S


def runtime(mx):
    def states(caches): return [(c.K,c.V) if type(c) is KV else (c.conv,c.S) for c in caches]
    def context(mode,caches,**kw): return NS(mode=mode,caches=caches,**kw)
    return NS(mx=mx,BatchKV=KV,BatchState=State,ForwardContext=context,states=states)


class Tests(unittest.TestCase):
    def test_fixed_case_order_bounds_and_scope(self):
        plan = q.fixed_plan()
        self.assertEqual([(a['name'],a['mode'],a['bound']) for a in plan['arms']],list(q.ARMS))
        self.assertEqual(plan['comparisons'],['A-B','A-C','B-D','A-D'])
        self.assertEqual((plan['batch'],plan['used'],plan['states'],plan['KV_backing']),(1,257,128,1536))
        self.assertFalse(plan['no_fuse_used']); self.assertFalse(plan['performance_qualified'])

    def test_bound257_forward_is_original_candidate_forward_ast(self):
        old = ast.parse((HERE.parent/'long_decode_graph/candidate.py').read_text())
        new = ast.parse((HERE/'qualify.py').read_text())
        left = next(n for n in ast.walk(old) if isinstance(n,ast.FunctionDef) and n.name == 'forward')
        right = next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name == 'forward')
        self.assertEqual(ast.dump(left,include_attributes=False),ast.dump(right,include_attributes=False))

    def test_compiled_call_captures_model_bound_and_returns_changed_state(self):
        mx = MX(); r = runtime(mx); events = []
        class Model:
            layers = [NS(is_attn=True),NS(is_attn=False)]
            def __call__(self,ids,ctx):
                events.append((ctx.max_len,ids.tolist(),ctx.positions.tolist(),ctx.lens.tolist()))
                ctx.caches[0].K = ctx.caches[0].K+1
                ctx.caches[1].S = ctx.caches[1].S+2
                return ids.astype(np.float32)
        model = Model(); fn = q.make_bound257(model,r)
        states = [(np.zeros((1,1,1536,1),np.float32),np.zeros((1,1,1536,1),np.float32)),
                  (np.zeros((1,3,2),np.float32),np.zeros((1,1,2,2),np.float32))]
        output,after = fn(np.array([7]),np.array([256]),np.array([257]),states)
        self.assertIs(mx.seen_model,model); self.assertEqual(mx.calls,1)
        self.assertEqual(events,[(257,[7],[256],[257])])
        self.assertEqual(float(after[0][0].max()),1); self.assertEqual(float(after[1][1].max()),2)
        self.assertEqual(float(states[0][0].max()),0); self.assertEqual(float(states[1][1].max()),0)

    def test_context_uses_distinct_array_handles_original_input_survives(self):
        r = runtime(MX())
        arrays = {'root/state/0/K':np.ones((1,2,3,4),np.float32),'root/state/0/V':np.zeros((1,2,3,4),np.float32),
            'root/state/1/conv':np.ones((1,3,4),np.float32),'root/state/1/S':np.ones((1,2,3,4),np.float32),
            'root/tokens':np.array([7],np.int32),'root/positions':np.array([256],np.int32),'root/lens':np.array([257],np.int32)}
        ctx = q.context_from_input(arrays,[True,False],r,1536)
        self.assertEqual(ctx.max_len,1536)
        self.assertIsNot(ctx.caches[0].K,arrays['root/state/0/K'])
        self.assertIsNot(ctx.caches[1].conv,arrays['root/state/1/conv'])
        # NumPy cannot model MLX descriptor replacement, so replace explicitly.
        ctx.caches[0].K = np.zeros_like(ctx.caches[0].K)
        self.assertTrue(np.all(arrays['root/state/0/K'] == 1))

    def test_bit_differences_keep_all_checks_including_signed_zero(self):
        a = {'x':np.array([0.,1.],np.float32),'y':np.array([2.],np.float32),'z':np.array([4.],np.float32)}
        b = {'x':np.array([-0.,1.],np.float32),'y':np.array([3.],np.float32),'z':np.array([4.],np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            row = q.compare_finite_outputs(s,MX(),Path(tmp),'A-B',s.snapshot(a),s.snapshot(b))
            self.assertEqual(row['status'],'DIFFERENT'); self.assertFalse(row['numerical_equal'])
            self.assertEqual([r['bit_equal'] for r in row['arrays']],[False,False,True])
            self.assertTrue((Path(tmp)/'comparisons/A-B/failure.json').exists())

    def test_nonfinite_shape_or_metadata_failure_remains_fatal(self):
        left = s.snapshot(np.array([1.],np.float32))
        wrong = [s.snapshot(np.array([np.inf],np.float32)),s.snapshot(np.array([1.,2.],np.float32)),
                 s.snapshot({'other':np.array([1.],np.float32)})]
        for value in wrong:
            with self.subTest(value=value),tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(s.QualificationFailure):
                    q.compare_finite_outputs(s,MX(),Path(tmp),'A-B',left,value)

    def test_exact_plan_rejects_extra_fields_and_unbounded_deadline(self):
        for budget in (0,-1,1801,True,float('nan')):
            plan = dict(schema='E024-dispatch-isolation-plan-v1',qualifier_sha256=q.sha(q.__file__),
                        e024_plan={},deadline_s=budget)
            with self.subTest(budget=budget),self.assertRaisesRegex(ValueError,'bounded deadline'): q.validate(plan)
        with self.assertRaisesRegex(ValueError,'exact isolation plan'): q.validate({'schema':'wrong'})

    def test_actual_mlx_safetensors_null_metadata_header_and_payload_bounds(self):
        # Same __metadata__:null layout observed in actual E024 first-failure file.
        values = np.array([1.,2.],np.float32)
        header = {'__metadata__':None,'root/logits':{'data_offsets':[0,8],'dtype':'F32','shape':[2]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'raw.safetensors'
            def save(payload):
                h = json.dumps(header).encode(); path.write_bytes(struct.pack('<Q',len(h))+h+payload)
            save(values.tobytes())
            row = q.compact_manifest(path,{'root/logits':values},{'root/logits':True},lambda:None)
            self.assertEqual(row['root/logits']['bytes'],8)
            self.assertEqual(row['root/logits']['encoding'],'F32')
            save(values.tobytes()[:-1])
            with self.assertRaisesRegex(ValueError,'short tensor'): q.compact_manifest(path,{'root/logits':values},{'root/logits':True},lambda:None)
            save(values.tobytes()+b'!')
            with self.assertRaisesRegex(ValueError,'trailing tensor'): q.compact_manifest(path,{'root/logits':values},{'root/logits':True},lambda:None)

    def test_parameter_hash_original_unsigned_bytes_and_handle_not_gate(self):
        params = {str(dtype):np.arange(8,dtype=dtype).reshape(2,4)[:,::2]
                  for dtype in (np.uint8,np.uint16,np.uint32,np.uint64,np.float32)}
        params['signed-zero'] = np.array([0.,-0.],np.float32)
        class Model:
            def parameters(self): return params
        rows = q.parameter_hashes(MX(),Model(),lambda values:values.items(),lambda:None)
        for name,array in params.items():
            import hashlib
            self.assertEqual(rows[name]['sha256'],hashlib.sha256(array.tobytes()).hexdigest())
            self.assertEqual(rows[name]['bytes'],array.nbytes)
        changed_handles = {k:{**v,'handle_id':v['handle_id']+1} for k,v in rows.items()}
        self.assertEqual(q.parameter_values(rows),q.parameter_values(changed_handles))
        changed_handles['signed-zero']['sha256'] = '0'*64
        self.assertNotEqual(q.parameter_values(rows),q.parameter_values(changed_handles))

    def test_help_blocks_all_model_imports(self):
        code = '''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*a):
  if name.split('.')[0] in ('mlx','numpy','tutorial_code'): raise RuntimeError('device import '+name)
sys.meta_path.insert(0,Block());sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
'''
        p = subprocess.run([sys.executable,'-I','-B','-c',code,str(HERE/'qualify.py')],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr); self.assertIn('--plan',p.stdout)


if __name__ == '__main__': unittest.main()
