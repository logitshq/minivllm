"""Fake-only probe checks; original BF16 bytes and timing counts, never GPU math."""
import ast
from collections import Counter
import contextlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

P=Path(__file__).with_name('real_projection_probe_v1.py')
s=importlib.util.spec_from_file_location('m8_register_probe_cpu',P);p=importlib.util.module_from_spec(s);s.loader.exec_module(p)

def array(words=None):
    return SimpleNamespace(shape=(8,4),dtype='bf16',data=struct.pack('<32H',*([0x3f80]*32 if words is None else words)))

class Fake:
    bfloat16='bf16'
    def __init__(self):self.evals=0;self.syncs=0
    def synchronize(self):self.syncs+=1
    def eval(self,*args):self.evals+=1
    def get_active_memory(self):return 0
    get_peak_memory=get_active_memory
    get_cache_memory=get_active_memory
    def save_safetensors(self,path,arrays):
        head={};chunks=[];offset=0
        for name,a in arrays.items():
            head[name]=dict(dtype='BF16',shape=list(a.shape),data_offsets=[offset,offset+len(a.data)])
            chunks.append(a.data);offset+=len(a.data)
        raw=json.dumps(head).encode();raw+=b' '*((-len(raw))%8)
        with Path(path).open('xb') as f:f.write(struct.pack('<Q',len(raw))+raw+b''.join(chunks))

class Tests(unittest.TestCase):
    def test_default_cli_inert_and_exact_native_m8_route(self):
        with patch.object(p,'device',side_effect=AssertionError('must remain inert')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(p.main([]),0)
            with self.assertRaisesRegex(ValueError,'requires --run'):p.main(['--out','/tmp/unused'])
        tree=ast.parse(P.read_text())
        routes=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='routes' for t in n.targets))
        self.assertEqual([k.arg for k in routes.keywords],['native_m8','candidate'])
        native=routes.keywords[0].value.body;candidate=routes.keywords[1].value.body
        self.assertEqual(ast.dump(native),ast.dump(ast.parse('mx.quantized_matmul(x,w,s,**Q)',mode='eval').body))
        self.assertEqual(ast.dump(candidate),ast.dump(ast.parse('project(x,w,s)',mode='eval').body))
        loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='T']
        self.assertEqual(len(loops),1);self.assertEqual(ast.literal_eval(loops[0].iter),(8,))
        self.assertEqual(len(p.PROJECTIONS),2);self.assertEqual([x[0] for x in p.PROJECTIONS],['head','up'])
        self.assertEqual(p.Q,dict(transpose=True,group_size=16,bits=4,mode='nvfp4'))
        self.assertEqual((p.LIMIT,p.DEADLINE,p.K),(4<<30,120,5120))
        self.assertEqual(p.sha(P.with_name('candidate.py')),p.PINS[str(P.with_name('candidate.py'))])
        self.assertNotIn('native_b1',P.read_text());self.assertNotIn('custom_b1',P.read_text())

    def test_full_bf16_bytes_one_comparison_and_failures_retained(self):
        cases=[('equal',[0x3f80]*32,True),('last_word',[0x3f80]*31+[0x8000],False),('nonfinite',[0x7f80]*32,False)]
        for label,words,ok in cases:
            with self.subTest(label=label),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);mx=Fake();left=array(words);right=left if label=='nonfinite' else array()
                routes=dict(native_m8=lambda:right,candidate=lambda:left)
                if ok:p.qualify(mx,folder,array(),routes,8,4)
                else:
                    with self.assertRaisesRegex(ValueError,'raw BF16 qualification failed'):p.qualify(mx,folder,array(),routes,8,4)
                record=json.loads((folder/'numerical.json').read_text())
                self.assertEqual(record['status'],'PASS' if ok else 'FAIL')
                self.assertEqual(list(record['comparisons']),['candidate__native_m8'])
                self.assertTrue((folder/'outputs.safetensors').is_file());self.assertTrue((folder/'inputs.safetensors').is_file())
                if label=='last_word':
                    mismatch=record['comparisons']['candidate__native_m8']['first_mismatch']
                    self.assertEqual((mismatch['row'],mismatch['column']),(7,3))
                if label=='nonfinite':self.assertFalse(record['finite'])

    def test_two_cases_produce_exact_400_timed_samples_with_abba_order(self):
        for label in ('head','up'):
            with self.subTest(label=label),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);mx=Fake();calls=[]
                def route(name):calls.append(name);return None
                routes=dict(native_m8=lambda:route('native_m8'),candidate=lambda:route('candidate'))
                clock=itertools.count(0,17)
                with patch.object(p.time,'perf_counter_ns',side_effect=lambda:next(clock)):
                    result=p.timing(mx,folder,routes)
                rows=[json.loads(v) for v in (folder/'timings.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),200);self.assertEqual(Counter(calls),dict(native_m8=105,candidate=105))
                self.assertEqual(Counter(r['route'] for r in rows),dict(native_m8=100,candidate=100))
                for group in range(5):
                    for pair in range(20):
                        part=rows[(group*20+pair)*2:(group*20+pair)*2+2]
                        expected=['native_m8','candidate'] if pair%2==0 else ['candidate','native_m8']
                        self.assertEqual([r['route'] for r in part],expected)
                        self.assertEqual([r['position'] for r in part],[0,1])
                        self.assertTrue(all(r['group']==group and r['pair']==pair and r['elapsed_ns']==17 for r in part))
                self.assertEqual(list(result),['native_m8']);self.assertEqual(len(result['native_m8']['groups']),5)
                self.assertEqual(result['native_m8']['median_group_ratio'],1)
                self.assertEqual(mx.evals,210);self.assertEqual(mx.syncs,425)

if __name__=='__main__':unittest.main()
