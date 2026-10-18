"""Stdlib raw-parser/audit rejection fixtures; source only, not yet executed.

Tiny real safetensors bytes exercise bitwise/finite/shape/oracle checks. They
are not model activations and do not constitute numerical/device qualification.
"""
import builtins
import json
from pathlib import Path
import runpy
import struct
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_prework_unbound import audit as a


def tensor_file(specs):
    header={};payload=b''
    for name,encoding,shape,words in specs:
        raw=b''.join(struct.pack('<H' if encoding=='BF16' else '<I',word) for word in words)
        header[name]=dict(dtype=encoding,shape=shape,data_offsets=[len(payload),len(payload)+len(raw)])
        payload+=raw
    header['__metadata__']=None
    encoded=json.dumps(header,separators=(',',':')).encode();return struct.pack('<Q',len(encoded))+encoded+payload


class Tests(unittest.TestCase):
    def test_import_is_independent_of_model_numpy_and_producer(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='mlx' or name.startswith(('mlx.','numpy','tutorial_code')) or name.endswith('.qualify'):
                raise AssertionError('runtime/producer import')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):runpy.run_path(a.__file__)

    def test_bf16_and_fp32_preserve_exact_signed_zero_and_small_bits(self):
        raw=tensor_file([('bf','BF16',[4],[0,0x8000,1,0x3f80]),('fp','F32',[3],[0x80000000,1,0xbf800000])])
        got=a.tensors(raw)
        self.assertEqual(list(a.words(got['bf']['raw'],'BF16')),[0,0x8000,1,0x3f80])
        self.assertEqual(list(a.words(got['fp']['raw'],'F32')),[0x80000000,1,0xbf800000])
        negative=dict(got['bf'],raw=struct.pack('<4H',0x8000,0x8000,1,0x3f80))
        with self.assertRaisesRegex(a.Reject,'first differing byte 1'):a.pair(got['bf'],negative,'signed-zero')

    def test_nan_and_infinity_original_bits_are_rejected(self):
        for encoding,value in (('BF16',0x7f80),('BF16',0xff80),('BF16',0x7fc1),('F32',0x7f800000),('F32',0xff800000),('F32',0x7f800001)):
            with self.subTest(encoding=encoding,value=value),self.assertRaisesRegex(a.Reject,'nonfinite'):
                a.tensors(tensor_file([('x',encoding,[1],[value])]))

    def test_truncation_trailing_data_wrong_range_and_shape_reject(self):
        raw=tensor_file([('x','BF16',[2],[0,0x3f80])])
        for invalid in (raw[:-1],raw+b'x',tensor_file([('x','BF16',[3],[0,0x3f80])])):
            with self.assertRaises(a.Reject):a.tensors(invalid)
        n,=struct.unpack_from('<Q',raw);header=json.loads(raw[8:8+n]);header['x']['data_offsets']=[1,5]
        encoded=json.dumps(header).encode()
        with self.assertRaisesRegex(a.Reject,'range'):a.tensors(struct.pack('<Q',len(encoded))+encoded+raw[8+n:])

    def test_duplicate_header_keys_cannot_hide_a_second_tensor(self):
        header=b'{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
        with self.assertRaisesRegex(a.Reject,'duplicate'):a.tensors(struct.pack('<Q',len(header))+header+b'\0\0')

    def test_full_vocab_argmax_is_independent_and_ties_use_first_index(self):
        # Five columns, including maxima away from the first/top1-like position.
        for encoding,words in (('BF16',[0xbf80,0,0x4000,0x4000,0x3f80]),
                               ('F32',[0xbf800000,0,0x40000000,0x40000000,0x3f800000])):
            value=a.tensors(tensor_file([('logits',encoding,[1,5],words)]))['logits']
            self.assertEqual(a.argmax(value),2)
        wrong=dict(value,shape=[5])
        with self.assertRaisesRegex(a.Reject,'logits'):a.argmax(wrong)

    def test_same_shape_mismatching_recurrent_state_byte_rejects(self):
        before=a.tensors(tensor_file([('S','F32',[2],[0x3f800000,0x40000000])]))['S']
        later=a.tensors(tensor_file([('S','F32',[2],[0x3f800000,0x40000001])]))['S']
        with self.assertRaisesRegex(a.Reject,'first differing byte 4'):a.pair(before,later,'continued-state')
        with self.assertRaisesRegex(a.Reject,'shape/dtype'):a.pair(before,dict(before,shape=[1,2]),'reshaped')

    def test_raw_artifact_sha_dtype_metadata_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            child=Path(directory).resolve();raw=tensor_file([('root/x','BF16',[1,2],[0,0x3f80])]);path=child/'one.safetensors';path.write_bytes(raw)
            payload=a.tensors(raw)['root/x']['raw']
            desc=dict(path=path.name,sha256=a.sha_bytes(raw),finite=True,metadata={'x':{'array':'root/x'}},
                tensors={'root/x':dict(shape=[1,2],dtype='mlx.core.bfloat16',encoding='BF16',bytes=4,sha256=a.sha_bytes(payload),finite=True)})
            path.with_suffix('.json').write_text(json.dumps(desc))
            seen=set();got=a.raw_file(child,desc,{'root/x':([1,2],{'BF16'})},seen)
            self.assertEqual(got['root/x']['raw'],payload)
            with self.assertRaisesRegex(a.Reject,'duplicate raw'):a.raw_file(child,desc,{'root/x':([1,2],{'BF16'})},seen)
            with self.assertRaisesRegex(a.Reject,'shape/dtype'):a.raw_file(child,desc,{'root/x':([1,2],{'F32'})},set())
            with self.assertRaisesRegex(a.Reject,'metadata'):a.raw_file(child,{**desc,'metadata':{'other':{'array':'root/x'}}},{'root/x':([1,2],{'BF16'})},set())
            path.write_bytes(raw[:-1]+b'\x7f')
            with self.assertRaisesRegex(a.Reject,'SHA'):a.raw_file(child,desc,{'root/x':([1,2],{'BF16'})},set())

    def test_counter_delta_and_point_binding_reject_stale_fallback_and_reorder(self):
        point=dict(fixture_id='long',layer=32,decode_step=129,used=1107)
        row=dict(**point,before=dict(candidate=3,fallback=0),after=dict(candidate=4,fallback=0))
        a.dispatch(row,point,3)
        for corrupt in ({**row,'after':dict(candidate=3,fallback=0)},
                        {**row,'after':dict(candidate=4,fallback=1)},
                        {**row,'before':dict(candidate=2,fallback=0)},
                        {**row,'decode_step':130},{**row,'layer':0},
                        {**row,'after':dict(candidate=4.0,fallback=0)}):
            with self.assertRaises(a.Reject):a.dispatch(corrupt,point,3)

    def test_parameter_contract_covers_all14_actual_arrays(self):
        spec=a.parameter_layout();self.assertEqual(len(spec),14)
        self.assertEqual(spec['in_proj_qkv.weight'],([10240,640],'mlx.core.uint32',4))
        self.assertEqual(spec['conv1d_weight'],([10240,4],'mlx.core.bfloat16',2))
        self.assertEqual(spec['A_log'],([48],'mlx.core.float32',4))

    def test_configuration_and_scope_receipt_cannot_be_relabelled(self):
        expected=a.expected_contract()
        for key,value in (('norm_reuse',True),('compile_decode',True),('gates_compile',1)):
            changed=a.expected_contract();changed['configuration'][key]=value
            with self.assertRaises(a.Reject):a.same(changed,expected,'fixed experiment')
        changed=a.expected_contract();changed['full_model_qualified']=True
        with self.assertRaises(a.Reject):a.same(changed,expected,'fixed scope')

    def test_missing_terminal_is_incomplete_without_tensor_reads(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(a,'raw_file',side_effect=AssertionError('raw opened')):
            result=a.audit(directory,'0'*64)
        self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['canary_qualified']);self.assertEqual(result['raw_files'],0)

    def test_failed_or_incomplete_producer_never_gets_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            child=Path(directory)
            for status,expected in (('FAIL','FAIL'),('INCOMPLETE','INCOMPLETE')):
                (child/'complete.json').write_text(json.dumps(dict(schema='GDN-prework-real-layer-result-v1',status=status)))
                with patch.object(a,'raw_file',side_effect=AssertionError('raw opened')):result=a.audit(child,'0'*64)
                self.assertEqual(result['status'],expected);self.assertFalse(result['canary_qualified'])
                self.assertFalse(result['full_model_qualified']);self.assertFalse(result['performance_qualified'])


if __name__=='__main__':unittest.main()
