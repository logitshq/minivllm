"""Tiny raw byte corruptions only; no GPU, model, socket or actual run evidence."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from bench.m5_ultra_256gb.long_decode_isolation import audit_raw as a


def save(root,name,items):
    header={};payload=b'';tensors={}
    for key,(dtype,shape,raw) in items.items():
        offset=len(payload);payload+=raw
        header[key]=dict(dtype=dtype,shape=shape,data_offsets=[offset,len(payload)])
        tensors[key]=dict(shape=shape,dtype=a.DTYPE[dtype],encoding=dtype,bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),finite=True)
    data=json.dumps(header,separators=(',',':')).encode();path=root/name
    path.write_bytes(struct.pack('<Q',len(data))+data+payload)
    return dict(path=name,sha256=a.sha(path),finite=True,tensors=tensors)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_original_reader_agreement_for_bf16_f32(self):
        record=save(self.root,'float.safetensors',{'bf':('BF16',[3],struct.pack('<3H',0,0x8000,0x3f80)),
            'fp':('F32',[2],struct.pack('<2I',0x3f800000,0x3f800001))})
        old=a.reader().tensors(self.root,record);new=a.snapshot_index(self.root,record)
        for key,(dtype,shape,raw) in old.items():
            self.assertEqual(new['tensors'][key]['encoding'],dtype)
            self.assertEqual(new['tensors'][key]['shape'],shape)
            self.assertEqual(a.tensor_bytes(new,key),raw)

    def test_i32_context_extension_preserves_original_bytes(self):
        record=save(self.root,'input.safetensors',{'root/tokens':('I32',[1],struct.pack('<i',524)),
            'root/positions':('I32',[1],struct.pack('<i',256)),'root/lens':('I32',[1],struct.pack('<i',257))})
        index=a.snapshot_index(self.root,record)
        self.assertEqual(struct.unpack('<i',a.tensor_bytes(index,'root/tokens')),(524,))
        self.assertTrue(a.compare_snapshots(index,index)['bit_equal'])
        with self.assertRaises(ValueError):a.layout(index,'input')

    def test_unsigned_words_catch_signed_zero_and_small_f32_change(self):
        bf=a.compare_words(struct.pack('<3H',0x3f80,0,0x4000),struct.pack('<3H',0x3f80,0x8000,0x4001),'BF16',[1,3])
        self.assertFalse(bf['bit_equal']);self.assertEqual(bf['differing_words'],2)
        self.assertEqual(bf['first_difference']['coordinates'],[0,1]);self.assertEqual(bf['first_difference']['byte_offset'],3)
        fp=a.compare_words(struct.pack('<2I',0x3f800000,0x3f800000),struct.pack('<2I',0x3f800000,0x3f800001),'F32',[2])
        self.assertEqual(fp['differing_words'],1);self.assertEqual(fp['first_difference']['byte_offset'],4)

    def test_nan_infinity_or_false_sha_cannot_pass_finite(self):
        for dtype,word,code in (('BF16',0x7f80,'H'),('BF16',0xffc0,'H'),('F32',0x7f800001,'I'),('F32',0xff800000,'I')):
            record=save(self.root,'bad.safetensors',{'x':(dtype,[1],struct.pack('<'+code,word))})
            with self.subTest(dtype=dtype,word=word),self.assertRaises(ValueError):a.snapshot_index(self.root,record)
        record=save(self.root,'valid.safetensors',{'x':('BF16',[1],struct.pack('<H',0x3f80))})
        record['sha256']='0'*64
        with self.assertRaises(ValueError):a.snapshot_index(self.root,record)

    def test_missing_cache_wrong_dtype_shape_and_context_arrays_rejected(self):
        for kind in ('input','output'):
            index={'tensors':{k:dict(shape=s,encoding=d) for k,(s,d) in a.expected_layout(kind).items()}}
            a.layout(index,kind)
            self.assertEqual(len(index['tensors']),131 if kind=='input' else 129)
            for change in ('missing','dtype','shape'):
                bad=copy.deepcopy(index);key='root/state/0/S'
                if change=='missing':bad['tensors'].pop(key)
                elif change=='dtype':bad['tensors'][key]['encoding']='BF16'
                else:bad['tensors'][key]['shape'][-1]=127
                with self.subTest(kind=kind,change=change),self.assertRaises(ValueError):a.layout(bad,kind)

    def test_header_corruption_and_tensor_changed_after_index_rejected(self):
        record=save(self.root,'x.safetensors',{'x':('BF16',[2],struct.pack('<2H',0x3f80,0x4000))})
        index=a.snapshot_index(self.root,record);path=self.root/record['path'];raw=path.read_bytes()
        path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with self.assertRaises(ValueError):a.tensor_bytes(index,'x')
        for header in (b'{"x":{"dtype":"BF16","shape":[true],"data_offsets":[0,4]}}',
            b'{"x":{"dtype":"BF16","shape":[2],"data_offsets":[1,5]}}',
            b'{"x":{},"x":{}}'):
            path.write_bytes(struct.pack('<Q',len(header))+header+raw[-4:]);record['sha256']=a.sha(path)
            with self.subTest(header=header),self.assertRaises(ValueError):a.snapshot_index(self.root,record)

    def test_parameter_handle_change_is_record_only_but_values_are_bound(self):
        before={'layer':dict(shape=[1],dtype='mlx.core.float32',bytes=4,sha256='1'*64,finite=True,handle_id=10)}
        after=copy.deepcopy(before);after['layer']['handle_id']=20
        self.assertEqual(a.parameter_values(before),a.parameter_values(after))
        after['layer']['sha256']='2'*64
        self.assertNotEqual(a.parameter_values(before),a.parameter_values(after))


if __name__=='__main__':unittest.main()
