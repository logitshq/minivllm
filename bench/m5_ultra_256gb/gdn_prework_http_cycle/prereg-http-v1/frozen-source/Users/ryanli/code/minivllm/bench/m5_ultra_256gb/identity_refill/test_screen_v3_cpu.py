"""CPU-only E028 contract and saved-byte negative tests; no MLX/socket/services."""
import importlib.util
import ast
import json
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).with_name('screen_v3.py')
spec = importlib.util.spec_from_file_location('e028_screen_v3_test',PATH)
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def snapshot(out,name,values,finite=True):
    header={}; payload=b''; metadata={}
    for key,(dtype,shape,raw) in values.items():
        header[key]=dict(dtype=dtype,shape=shape,data_offsets=[len(payload),len(payload)+len(raw)])
        payload+=raw
        metadata[key]=dict(shape=shape,encoding=dtype,finite=finite)
    text=json.dumps(header).encode(); path=out/name
    path.write_bytes(struct.pack('<Q',len(text))+text+payload)
    return dict(path=name,sha256=s.sha(path),tensors=metadata,finite=finite)


class OperatorArray:
    """Array-left integer operators only, matching the observed MLX restriction.

    Deliberately has no __ror__: int | array must fail on this fake, so running
    the actual generator catches scalar-left expressions instead of testing an
    independent all-Python-int restatement only.
    """
    def __init__(self, words, dtype='uint32'):
        self.words = tuple(words)
        self.dtype = dtype
        self.shape = (len(self.words),)

    def binary(self, other, operation):
        rhs = other.words if isinstance(other, OperatorArray) else (other,)*len(self.words)
        if len(rhs) != len(self.words): raise ValueError('shape mismatch')
        return OperatorArray([operation(a,b) for a,b in zip(self.words,rhs)],self.dtype)

    def __add__(self, other): return self.binary(other,lambda a,b:a+b)
    def __floordiv__(self, other): return self.binary(other,lambda a,b:a//b)
    def __mod__(self, other): return self.binary(other,lambda a,b:a%b)
    def __and__(self, other): return self.binary(other,lambda a,b:a&b)
    def __or__(self, other): return self.binary(other,lambda a,b:a|b)
    def __lshift__(self, other): return self.binary(other,lambda a,b:a<<b)

    def astype(self, dtype):
        mask = {'uint16':0xffff,'uint32':0xffffffff}[dtype]
        return OperatorArray([x&mask for x in self.words],dtype)

    def view(self, dtype):
        needed = {'bfloat16':'uint16','float32':'uint32'}[dtype]
        if self.dtype != needed: raise ValueError('wrong unsigned view width')
        return OperatorArray(self.words,dtype)

    def reshape(self, shape):
        size = 1
        for n in shape: size *= n
        if size != len(self.words): raise ValueError('reshape size')
        self.shape = tuple(shape)
        return self


class OperatorMX:
    uint16, uint32 = 'uint16','uint32'
    bfloat16, float32 = 'bfloat16','float32'

    @staticmethod
    def arange(size, dtype): return OperatorArray(range(size),dtype)


class ScreenTests(unittest.TestCase):
    def test_value_evaluates_each_compact_parent_before_return(self):
        events=[]
        parent=object()
        class MX:
            @staticmethod
            def eval(*arrays): events.append(('eval',arrays))
        def make(*args):
            events.append(('pattern',args[1:]))
            return parent
        with mock.patch.object(s,'pattern',side_effect=make):
            actual=s.value(MX,(2,3),'BF16',11,False)
        self.assertIs(actual,parent)
        self.assertEqual(events,[('pattern',((2,3),'BF16',11)),('eval',(parent,))])

    def test_value_evaluates_backing_before_creating_offset_strided_view(self):
        for shape,expanded in (((2,4,1536,256),(3,4,1538,512)),
                               ((4,3,10240),(5,3,20480)),
                               ((48,128,128),(49,128,256))):
            with self.subTest(shape=shape):
                events=[]
                class Parent:
                    evaluated=False
                    def __getitem__(self,index):
                        if not self.evaluated: raise AssertionError('view made before backing eval')
                        events.append(('slice',index))
                        return ('view',self,index)
                parent=Parent()
                class MX:
                    @staticmethod
                    def eval(*arrays):
                        if arrays!=(parent,): raise AssertionError('must eval original backing exactly once')
                        events.append(('eval',parent)); parent.evaluated=True
                def make(_mx,got_shape,dtype,seed):
                    events.append(('pattern',got_shape,dtype,seed)); return parent
                with mock.patch.object(s,'pattern',side_effect=make):
                    actual=s.value(MX,shape,'BF16',23,True)
                self.assertEqual([e[0] for e in events],['pattern','eval','slice'])
                self.assertEqual(events[0],('pattern',expanded,'BF16',23))
                self.assertEqual(actual[0],'view'); self.assertIs(actual[1],parent)
                index=actual[2]
                self.assertEqual(index[0],slice(1,shape[0]+1))
                self.assertEqual(index[-1],slice(None,None,2))
                if len(shape)==4: self.assertEqual(index[2],slice(1,shape[2]+1))

    def test_pattern_and_rebuild_observation_boundaries_unchanged_from_v2(self):
        def functions(path):
            tree=ast.parse(path.read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
        old=functions(PATH.with_name('screen_v2.py')); new=functions(PATH)
        for name in ('pattern','run_case','make_arm','mutate','sources','outputs'):
            self.assertEqual(new[name],old[name],name)

    def test_fake_reproduces_scalar_left_or_failure(self):
        low = OperatorArray([1,2,3])
        self.assertFalse(hasattr(low,'__ror__'))
        with self.assertRaises(TypeError): _ = 0x3e00 | low
        with self.assertRaises(TypeError): _ = 0x3e800000 | (low << 16)
        self.assertEqual((low | 0x3e00).words,(0x3e01,0x3e02,0x3e03))

    def test_actual_bf16_generator_with_array_only_operators(self):
        shape,seed = (2,129),37
        actual = s.pattern(OperatorMX,shape,OperatorMX.bfloat16,seed)
        expected=[]
        for i in range(258):
            mixed=i+i//127+seed
            expected.append(0x3e00 | (mixed%128) | ((mixed&1)<<15))
        self.assertEqual(actual.words,tuple(expected))
        self.assertEqual(actual.shape,shape)
        self.assertEqual(actual.dtype,'bfloat16')
        self.assertEqual({bool(v&0x8000) for v in actual.words},{False,True})
        self.assertTrue(all((v&0x7f80)!=0x7f80 for v in actual.words))

    def test_actual_fp32_generator_with_array_only_operators(self):
        shape,seed = (3,131),97
        actual = s.pattern(OperatorMX,shape,OperatorMX.float32,seed)
        expected=[]
        for i in range(393):
            mixed=i+i//127+seed
            expected.append(0x3e800000 | ((mixed%128)<<16) | ((mixed&1)<<31))
        self.assertEqual(actual.words,tuple(expected))
        self.assertEqual(actual.shape,shape)
        self.assertEqual(actual.dtype,'float32')
        self.assertEqual({bool(v&0x80000000) for v in actual.words},{False,True})
        self.assertTrue(all((v&0x7f800000)!=0x7f800000 for v in actual.words))

    def test_fixed_matrix_and_counts(self):
        plan=s.fixed_plan(); cases=plan['cases']
        self.assertEqual(len(cases),10); self.assertEqual(sum(c['checks'] for c in cases),88)
        self.assertEqual(2+5*len(cases)+sum(c['reverse_output_write'] for c in cases),53)
        self.assertEqual(sum(c['on_path']=='identity_refill' for c in cases),5)
        self.assertEqual([c['old_B'] for c in cases[:3]],[2,4,6])
        self.assertTrue(cases[-1]['delayed']); self.assertTrue(cases[3]['views'])
        self.assertEqual(plan['cache_limit_bytes'],0)
        self.assertEqual(plan['mlx_observation_budget_bytes'],1<<30)
        self.assertEqual(plan['physical_observation_budget_bytes'],2<<30)
        self.assertFalse(plan['performance_qualified']); self.assertFalse(plan['full_model_qualified'])

    def test_reorder_removal_growth_ids_and_lengths(self):
        cases={c['id']:c for c in s.cases()}
        reordered=s.expected_metadata(cases['control-reorder'])
        self.assertEqual(reordered['req_ids'],['old-3','old-2','old-1','old-0','new-0','new-1'])
        self.assertEqual(reordered['lens'],[941,934,927,920,935,954])
        removed=s.expected_metadata(cases['control-removal'])
        self.assertEqual(removed['req_ids'],['old-0','old-2','new-0','new-1'])
        self.assertEqual(s.expected_metadata(cases['control-length-growth'])['shapes']['K'],[6,4,2048,256])
        self.assertEqual(s.expected_metadata(cases['control-empty-old'])['req_ids'],['new-0','new-1'])

    def test_cli_imports_no_runtime(self):
        code='''import importlib.abc,sys,runpy
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('mlx','numpy','torch','tutorial_code'): raise AssertionError('runtime import '+name)
sys.meta_path.insert(0,Deny()); sys.argv=[sys.argv[1],*sys.argv[2:]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
        for argv,rc in (([],0),(['--describe'],0),(['--help'],0),(['--run'],2)):
            p=subprocess.run([sys.executable,'-I','-B','-c',code,str(PATH),*argv],capture_output=True,text=True,timeout=5)
            self.assertEqual(p.returncode,rc,p.stderr)

    def test_bf16_and_fp32_full_snapshot_compare(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td); result={'checks':[]}
            values={'conv':('BF16',[2],b'\0\0\0\x80'),'S':('F32',[2],struct.pack('<ff',1.25,-0.0))}
            a=snapshot(out,'a.safetensors',values); b=snapshot(out,'b.safetensors',values)
            s.compare(out,result,'mock','allbytes',a,b,lambda _:None)
            self.assertEqual(result['checks'][0]['status'],'PASS')
            self.assertEqual(set(result['checks'][0]['rawpair']['reference']['tensors']),{'conv','S'})

    def test_only_expected_first_scalar_may_change(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td); result={'checks':[]}
            values={'conv':('BF16',[2],b'\0\x3f\0\x40'),'S':('F32',[2],struct.pack('<ff',1,2))}
            changed={k:(dtype,shape,s.scalar_bytes(-16,dtype)+raw[len(s.scalar_bytes(-16,dtype)):]) for k,(dtype,shape,raw) in values.items()}
            a=snapshot(out,'a.safetensors',values); b=snapshot(out,'b.safetensors',changed)
            s.compare(out,result,'mock','mutation',a,b,lambda _:None,mutation=-16)
            changed['S']=('F32',[2],struct.pack('<ff',-16,3))
            bad=snapshot(out,'bad.safetensors',changed)
            with self.assertRaisesRegex(ValueError,'strict state gate'):
                s.compare(out,result,'mock','second-element-corruption',a,bad,lambda _:None,mutation=-16)
            self.assertEqual([r['status'] for r in result['checks']],['PASS','FAIL'])
            self.assertTrue((out/'first-difference.json').exists())

    def test_signed_zero_and_nonfinite_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            a=snapshot(out,'a',{'x':('BF16',[1],b'\0\0')})
            b=snapshot(out,'b',{'x':('BF16',[1],b'\0\x80')})
            with self.assertRaises(ValueError): s.compare(out,{'checks':[]},'c','zero',a,b,lambda _:None)
            with self.assertRaises(ValueError): s.compare(out,{'checks':[]},'c','finite',dict(a,finite=False),a,lambda _:None)

    def test_shape_mismatch_and_file_corruption_fail(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            a=snapshot(out,'a',{'x':('F32',[2],struct.pack('<ff',1,2))})
            b=snapshot(out,'b',{'x':('F32',[1,2],struct.pack('<ff',1,2))})
            with self.assertRaises(ValueError): s.compare(out,{'checks':[]},'c','shape',a,b,lambda _:None)
            (out/'a').write_bytes((out/'a').read_bytes()+b'x')
            with self.assertRaises(ValueError): s.read_tensors(out/'a')

    def test_signals_preserve_cancellation_reason(self):
        with self.assertRaisesRegex(TimeoutError,'SIGALRM'): s.stop_signal(signal.SIGALRM)
        with self.assertRaisesRegex(InterruptedError,'SIGTERM'): s.stop_signal(signal.SIGTERM)

    def test_source_pattern_distinguishes_old_rows(self):
        # Check the registered bit-pattern at row boundaries for all four fields.
        for stride in (4*1536*256,3*10240,48*128*128):
            words=[(i+i//127+11)%128 for i in [r*stride for r in range(8)]]
            self.assertEqual(len(set(words)),8)


if __name__=='__main__': unittest.main()
