"""CPU-only fixed bundle/schema tests; synthetic tiny files, no real payload/GPU."""
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


w = load('e026_test_layer_weights', HERE / 'layer_weights.py')
s = load('e026_test_layer_screen', HERE / 'layer_screen.py')


def tiny_source(root):
    """A 12-byte synthetic payload: this never opens any real model weight file."""
    model = root / 'model'; model.mkdir()
    rows = (('A_log', 'BF16', (2,), (0, 4)), ('conv1d.weight', 'BF16', (2, 2, 1), (4, 12)))
    header = {w.PREFIX+n: dict(dtype=d, shape=list(s), data_offsets=list(o)) for n,d,s,o in rows}
    header_raw = json.dumps(header).encode()
    payload = struct.pack('<6H', 0x3f80, 0xbf80, 0, 1, 0x3f00, 0x8000)
    shard = struct.pack('<Q', len(header_raw)) + header_raw + payload
    config = b'{"synthetic_cpu_fixture":true}'
    index = json.dumps({'weight_map': {k: w.SHARD for k in header}}).encode()
    (model / w.SHARD).write_bytes(shard)
    (model / 'config.json').write_bytes(config)
    (model / 'model.safetensors.index.json').write_bytes(index)
    patches = dict(ROWS=rows, SHARD_SHA=w.digest(shard), SHARD_BYTES=len(shard),
                   HEADER_BYTES=len(header_raw), HEADER_SHA=w.digest(header_raw),
                   CONFIG_SHA=w.digest(config), INDEX_SHA=w.digest(index))
    return model, patches, payload


class WeightPreparation(unittest.TestCase):
    def test_registered_allowlist_is_exact_and_bounded(self):
        rows = w.descriptors()
        self.assertEqual(len(rows), 14)
        self.assertEqual(sum(r['bytes'] for r in rows), 65239488)
        self.assertEqual([r['key'] for r in rows], sorted(r['key'] for r in rows))
        self.assertEqual(rows[0]['bundle_offsets'][0], 0)
        self.assertEqual(rows[-1]['bundle_offsets'][1], 65239488)
        for a, b in zip(rows, rows[1:]):
            self.assertEqual(a['bundle_offsets'][1], b['bundle_offsets'][0])
        self.assertEqual({r['dtype'] for r in rows}, {'BF16', 'U32', 'U8'})
        self.assertEqual([r['parameter'] for r in rows if 'conv' in r['key']], ['conv1d_weight'])

    def test_tiny_prepare_verify_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); model, pins, payload = tiny_source(root)
            for key, value in pins.items():
                stack.enter_context(patch.object(w, key, value))
            out = root / 'bundle'
            with patch('builtins.print'):
                self.assertEqual(w.prepare(NS(model_dir=model, out=out, deadline_seconds=5)), 0)
            manifest_path = out / 'manifest.json'
            manifest = w.verify_bundle(manifest_path, w.sha(manifest_path))
            self.assertEqual((out / 'weights.bin').read_bytes(), payload)
            self.assertEqual(manifest['source']['fd_before'], manifest['source']['fd_after'])
            self.assertTrue(manifest['source']['fully_hashed_before_extract'])
            self.assertEqual(manifest['tensors'][0]['sha256'], w.digest(payload[:4]))
            with self.assertRaises(FileExistsError):
                w.prepare(NS(model_dir=model, out=out, deadline_seconds=5))
            wrong = copy.deepcopy(manifest); wrong['bundle']['filename'] = '../weights.bin'
            with self.assertRaisesRegex(ValueError, 'extent'):
                w.validate_manifest(wrong)
            wrong = copy.deepcopy(manifest); wrong['tensors'][0]['bundle_offsets'] = [0, 5]
            with self.assertRaisesRegex(ValueError, 'descriptor'):
                w.validate_manifest(wrong)
            corrupted = bytes([payload[0] ^ 1]) + payload[1:]
            (out / 'weights.bin').write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError, 'tensor SHA'):
                w.verify_bundle(manifest_path, w.sha(manifest_path))

    def test_full_source_sha_failure_and_fd_change_leave_partial(self):
        for kind in ('hash', 'fd'):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp); model, pins, _ = tiny_source(root)
                for key, value in pins.items():
                    stack.enter_context(patch.object(w, key, value))
                if kind == 'hash':
                    stack.enter_context(patch.object(w, 'SHARD_SHA', '0' * 64))
                else:
                    original = w.stat_identity
                    calls = []
                    def changing(fd):
                        result = original(fd); calls.append(1)
                        if len(calls) > 1: result['st_ino'] += 1
                        return result
                    stack.enter_context(patch.object(w, 'stat_identity', side_effect=changing))
                out = root / 'bad'
                with patch('builtins.print'):
                    self.assertEqual(w.prepare(NS(model_dir=model, out=out, deadline_seconds=5)), 1)
                self.assertFalse((out / 'manifest.json').exists())
                record = json.loads((out / 'preparation.json').read_text())
                self.assertEqual(record['status'], 'FAIL')
                self.assertIn('SHA' if kind == 'hash' else 'identity', record['error'])

    def test_header_key_shape_extent_and_index_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            model, pins, _ = tiny_source(Path(tmp))
            for key, value in pins.items(): stack.enter_context(patch.object(w, key, value))
            raw = (model / w.SHARD).read_bytes()
            header = json.loads(raw[8:8+w.HEADER_BYTES])
            index = json.loads((model / 'model.safetensors.index.json').read_text())
            w.validate_header(header, index, len(raw))
            for mutation in ('extra', 'shape', 'offset', 'index', 'size'):
                h, ix, size = copy.deepcopy(header), copy.deepcopy(index), len(raw)
                key = w.PREFIX + 'A_log'
                if mutation == 'extra': h[w.PREFIX + 'unexpected'] = h[key]
                if mutation == 'shape': h[key]['shape'] = [1, 2]
                if mutation == 'offset': h[key]['data_offsets'] = [1, 5]
                if mutation == 'index': ix['weight_map'][key] = 'other.safetensors'
                if mutation == 'size': size -= 1
                with self.assertRaises(ValueError, msg=mutation): w.validate_header(h, ix, size)

    def test_load_conversions_exactly_follow_frozen_loader(self):
        class Tensor:
            def __init__(self, shape, dtype): self.shape, self.dtype = tuple(shape), dtype
            def view(self, dtype): return Tensor(self.shape, dtype)
            def reshape(self, a, b): return Tensor((a, 4) if b == -1 else (a, b), self.dtype)
            def astype(self, dtype): return Tensor(self.shape, dtype)
        class Host:
            def __init__(self, dtype): self.dtype = dtype
            def reshape(self, shape): self.shape = shape; return self
        np = NS(frombuffer=lambda raw, dtype: Host(dtype))
        mx = NS(array=lambda host: Tensor(host.shape, host.dtype), bfloat16='bf16', float32='f32')
        rows = []
        for name, dtype, shape in [('A_log','BF16',[48]), ('dt_bias','BF16',[48]),
                                   ('conv1d_weight','BF16',[10240,4,1]), ('norm.weight','BF16',[128]),
                                   ('in_proj_a.weight','U32',[48,640]), ('in_proj_a.scales','U8',[48,320])]:
            rows.append(dict(parameter=name, dtype=dtype, shape=shape, bytes=1, sha256=w.digest(b'x')))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'weights.bin').write_bytes(b'x' * len(rows))
            values = dict(w.load_arrays(mx, np, tmp, {'tensors': rows}))
        self.assertEqual(values['A_log'].dtype, 'f32')
        self.assertEqual(values['dt_bias'].dtype, 'f32')
        self.assertEqual((values['conv1d_weight'].dtype, values['conv1d_weight'].shape), ('bf16', (10240,4)))
        self.assertEqual(values['norm.weight'].dtype, 'bf16')
        self.assertEqual(values['in_proj_a.weight'].dtype, '<u4')
        self.assertEqual(values['in_proj_a.scales'].dtype, 'u1')

    def test_default_help_and_import_do_not_touch_runtime_or_payload(self):
        code = '''
import importlib.abc,importlib.util,socket,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*a):
  if name.split('.')[0] in {'mlx','numpy','torch','tutorial_code'}: raise AssertionError(name)
sys.meta_path.insert(0,Block());socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('socket'))
s=importlib.util.spec_from_file_location('weights',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
raise SystemExit(m.main(sys.argv[2:]))
'''
        for args, expected in (([],0), (['--help'],0), (['--prepare'],2)):
            result = subprocess.run([sys.executable,'-I','-B','-c',code,str(HERE/'layer_weights.py'),*args],
                                    capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode, expected, result.stderr)


class LayerScreenReview(unittest.TestCase):
    def test_contract_and_default_cli_no_runtime_or_files(self):
        contract=s.contract('on-lazy')
        self.assertEqual(contract['case_ids'],[f'T{t}-{kind}' for t in (1,19,935,978) for kind in ('fresh','continuing')])
        self.assertEqual(contract['phase_count'],4+8*len(contract['case_phases']))
        self.assertEqual(contract['continuing_offset'],7)
        self.assertFalse(contract['continuing_prefix_forward'])
        self.assertEqual(contract['mlx_observation_budget_bytes'],1<<30)
        code='''
import importlib.abc,importlib.util,socket,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*a):
  if name.split('.')[0] in {'mlx','numpy','torch','tutorial_code'}: raise AssertionError(name)
sys.meta_path.insert(0,Block());socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('socket'))
s=importlib.util.spec_from_file_location('screen',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
raise SystemExit(m.main(sys.argv[2:]))
'''
        for args, expected in (([],0),(['--help'],0),(['--run'],2),(['--out','unused'],2)):
            result=subprocess.run([sys.executable,'-I','-B','-c',code,str(HERE/'layer_screen.py'),*args],
                                  capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,expected,result.stderr)

    def test_footprint_exact_pid_units_and_current_peak_fail_closed(self):
        raw=dict(unit='byte',errors=[],warnings=[],processes=[dict(pid=123,footprint=100,
            auxiliary=dict(phys_footprint=100,phys_footprint_peak=1<<31))])
        raw['bytes per unit']=1
        self.assertEqual(s.parse_footprint(raw,123),(100,1<<31))
        for bad in ('unit','bool-unit','pid','bool-pid','duplicate','warnings','errors','peak','negative','inconsistent'):
            item=copy.deepcopy(raw)
            if bad=='unit':item['unit']='MiB'
            if bad=='bool-unit':item['bytes per unit']=True
            if bad=='pid':item['processes'][0]['pid']=124
            if bad=='bool-pid':item['processes'][0]['pid']=True
            if bad=='duplicate':item['processes']*=2
            if bad=='warnings':item['warnings']=['incomplete']
            if bad=='errors':item['errors']=['failure']
            if bad=='peak':item['processes'][0]['auxiliary']['phys_footprint_peak']=(1<<31)+1
            if bad=='negative':item['processes'][0]['auxiliary']['phys_footprint_peak']=-1
            if bad=='inconsistent':item['processes'][0]['auxiliary']['phys_footprint']=101
            with self.assertRaises(ValueError,msg=bad):s.parse_footprint(item,123)
        item=copy.deepcopy(raw);item['processes'][0]['pid']=True
        with self.assertRaises(ValueError):s.parse_footprint(item,1)

    def test_memory_observation_budget_not_tiny_budget(self):
        row=dict(active_bytes=(1<<30)-1,cache_bytes=1,peak_bytes=1<<30,unique_buffer_bytes=100)
        s.memory_guard(row)
        for key,value in (('cache_bytes',2),('peak_bytes',(1<<30)+1),('active_bytes',True),('unique_buffer_bytes',1<<31)):
            with self.assertRaises(ValueError):s.memory_guard(dict(row,**{key:value}))

    def test_actual_return_profile_restores_previous_on_success_and_failure(self):
        first,second=object(),object()
        def conv():return first,second
        calls=[]
        def prior(frame,event,arg):
            if frame.f_code is conv.__code__:calls.append(event)
        previous=sys.getprofile();sys.setprofile(prior)
        try:
            observed={'count':0}
            with s.observe_convolution(conv.__code__,observed):
                result=conv()
            self.assertIs(result,observed['tuple'])
            self.assertEqual(observed['count'],1)
            self.assertTrue(observed['restored'])
            self.assertIs(sys.getprofile(),prior)
            self.assertIn('return',calls)
            def broken():raise RuntimeError('original model failure')
            failed={'count':0}
            with self.assertRaisesRegex(RuntimeError,'original model failure'):
                with s.observe_convolution(broken.__code__,failed):broken()
            self.assertTrue(failed['restored'])
            self.assertTrue(failed['invalid_return'])
            self.assertIs(sys.getprofile(),prior)
        finally:sys.setprofile(previous)

    def test_fixed_seeds_and_original_unsigned_artifacts(self):
        import numpy as np  # CPU NumPy only; never MLX.
        for kind in ('x','conv','S'):
            words=s.seed_words(np,67,kind)
            expected=[]
            for n in range(67):
                if kind=='S':value=0x37800000|((13*n+37)&0x7fffff)|(((n+37)&1)<<31)
                else:
                    base,salt=(0x3c00,11) if kind=='x' else (0x3b00,23)
                    value=base|((17*n+salt)&127)|(((n+salt)&1)<<15)
                expected.append(value)
            self.assertEqual(words.tolist(),expected)
        mx=NS(bfloat16='bf16',float32='f32',uint16='u16',uint32='u32',uint8='u8')
        class Value:
            dtype='bf16';shape=(2,2)
            def view(self,dtype):
                self.requested=dtype
                return np.array([0x0000,0x8000,0x3f80,0xbf80],dtype='<u2').reshape(self.shape)
        with tempfile.TemporaryDirectory() as tmp:
            value=Value(); row=s.array_record(mx,np,value,Path(tmp),'original.bin')
            expected=struct.pack('<4H',0,0x8000,0x3f80,0xbf80)
            self.assertEqual((Path(tmp)/'original.bin').read_bytes(),expected)
            self.assertEqual(value.requested,'u16')
            self.assertEqual(row,dict(file='original.bin',sha256=w.digest(expected),bytes=8,
                                     dtype='bfloat16',shape=[2,2],encoding='uint16-le',finite=True))
            value.view=lambda dtype:np.array([0x7f80,0x7fc1,0,0],dtype='<u2').reshape(2,2)
            failed=s.array_record(mx,np,value,Path(tmp),'nonfinite.bin')
            self.assertFalse(failed['finite'])
            self.assertTrue((Path(tmp)/'nonfinite.bin').is_file())
            with self.assertRaisesRegex(ValueError,'nonfinite'):s.require_finite({'array':failed})

    def test_parameter_hash_difference_saves_actual_changed_bits(self):
        import numpy as np
        mx=NS(bfloat16='bf16',float32='f32',uint16='u16',uint32='u32',uint8='u8')
        class Value:
            dtype='bf16';shape=(1,)
            def __init__(self,word):self.word=word
            def view(self,dtype):return np.array([self.word],dtype='<u2')
        layer=NS(parameters=lambda:{'norm':{'weight':Value(0x3f80)}})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            baseline=s.parameter_records(mx,np,s.flat_parameters(layer),out,'before')
            s.same_parameters(mx,np,layer,baseline,out,'changed')
            layer.parameters=lambda:{'norm':{'weight':Value(0x7fc1)}}
            with self.assertRaisesRegex(ValueError,'parameter bits'):
                s.same_parameters(mx,np,layer,baseline,out,'changed')
            self.assertEqual((out/'changed/norm.weight.bin').read_bytes(),struct.pack('<H',0x7fc1))


if __name__ == '__main__':
    unittest.main()
