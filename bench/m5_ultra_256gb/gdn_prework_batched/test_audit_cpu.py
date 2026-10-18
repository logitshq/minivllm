"""Small independent bytes/row-causality regressions; no model/device proof."""
import ast
import builtins
import copy
import inspect
import json
from pathlib import Path
import runpy
import struct
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_prework_batched import audit as a


def blob(values,encoding='BF16',shape=None):
    raw=b''.join(struct.pack('<H' if encoding=='BF16' else '<I',v) for v in values)
    header=json.dumps({'root':dict(dtype=encoding,shape=shape or [len(values)],data_offsets=[0,len(raw)])}).encode()
    return struct.pack('<Q',len(header))+header+raw


class Tests(unittest.TestCase):
    def chain(self,B=8,start=1):
        fs=[dict(fixture_id=f'd{i}',prompt_ids=[1]*(886+i),eos_token_ids=[9999]) for i in range(B)]
        raw=[[i*200+n for n in range(start+3)] for i in range(B)];ids=[f['fixture_id'] for f in fs]
        case=dict(case_id='case',batch_size=B,row_ids=ids,start=start,prompt_tokens=[len(f['prompt_ids']) for f in fs],raw_ids=raw,steps=[{}, {}, {}])
        tokens=[dict(batch_size=B,row_ids=list(ids),decode_step=d,input_tokens=[r[d-1] for r in raw],
                     output_tokens=[r[d] for r in raw],captured=d>=start) for d in range(1,start+3)]
        return case,fs,tokens

    def test_actual_eight_row_order_and_late_history_are_checked(self):
        case,fs,tokens=self.chain(8,129)
        self.assertIs(a.token_chain(case,'case',8,129,fs,tokens,20000),case['raw_ids'])
        tokens[-1]['input_tokens'][0],tokens[-1]['input_tokens'][1]=tokens[-1]['input_tokens'][1],tokens[-1]['input_tokens'][0]
        with self.assertRaisesRegex(a.Reject,'causality'):a.token_chain(case,'case',8,129,fs,tokens,20000)

    def test_eos_in_any_row_before_final_output_rejects(self):
        case,fs,tokens=self.chain(2,1)
        fs[1]['eos_token_ids']=[case['raw_ids'][1][-1]]
        a.token_chain(case,'case',2,1,fs,tokens,20000)
        fs[1]['eos_token_ids']=[case['raw_ids'][1][-2]]
        with self.assertRaisesRegex(a.Reject,'normal EOS'):a.token_chain(case,'case',2,1,fs,tokens,20000)

    def test_missing_step_boolean_token_and_reordered_rows_reject(self):
        for kind in ('missing','boolean','rows'):
            with self.subTest(kind=kind):
                case,fs,tokens=self.chain(2,129)
                if kind=='missing':tokens.pop()
                elif kind=='boolean':case['raw_ids'][0][0]=False
                else:case['row_ids'].reverse()
                with self.assertRaises(a.Reject):a.token_chain(case,'case',2,129,fs,tokens,20000)

    def test_full_batched_argmax_uses_all_columns_and_stable_tie_order(self):
        # Row0's winner is final column; row1 ties +0/-0 and keeps first index.
        tensor=a.tensors(blob([0xbf80,0x0000,0x3f00,0x4000,0x0000,0x8000,0xbf80,0xbf00],shape=[2,4]))['root']
        self.assertEqual(a.argmax_rows(tensor),[3,0])
        with self.assertRaises(a.Reject):a.argmax_rows(dict(tensor,shape=[4,2]))

    def test_original_bits_finiteness_and_contiguous_ranges_not_float_equality(self):
        positive=a.tensors(blob([0]))['root'];negative=a.tensors(blob([0x8000]))['root']
        with self.assertRaisesRegex(a.Reject,'byte'):a.pair(positive,negative,'signed zero')
        for encoding,value in [('BF16',0x7f80),('BF16',0x7fc0),('F32',0x7f800000),('F32',0xff800000)]:
            with self.subTest(encoding=encoding,value=value),self.assertRaisesRegex(a.Reject,'nonfinite'):
                a.tensors(blob([value],encoding))
        raw=blob([0,1]);n=struct.unpack_from('<Q',raw)[0];head=json.loads(raw[8:8+n]);head['root']['data_offsets']=[1,5]
        header=json.dumps(head).encode()
        with self.assertRaisesRegex(a.Reject,'range'):a.tensors(struct.pack('<Q',len(header))+header+raw[8+n:])

    def test_raw_file_descriptor_and_adjacent_receipt_are_both_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            child=Path(temp).resolve();raw=blob([0,0x3f80,0x4000,0x4040],shape=[2,2]);path=child/'one.safetensors';path.write_bytes(raw)
            payload=a.tensors(raw)['root']['raw']
            desc=dict(path=path.name,sha256=a.sha_bytes(raw),finite=True,metadata={'array':'root'},
                tensors={'root':dict(shape=[2,2],dtype='mlx.core.bfloat16',encoding='BF16',bytes=len(payload),sha256=a.sha_bytes(payload),finite=True)})
            sidecar=path.with_suffix('.json');sidecar.write_text(json.dumps(desc))
            got=a.raw_file(child,desc,{'root':([2,2],{'BF16'})},set());self.assertEqual(got['root']['raw'],payload)
            bad=copy.deepcopy(desc);bad['tensors']['root']['sha256']='0'*64
            with self.assertRaises(a.Reject):a.raw_file(child,bad,{'root':([2,2],{'BF16'})},set())
            sidecar.write_text('{}')
            with self.assertRaises(a.Reject):a.raw_file(child,desc,{'root':([2,2],{'BF16'})},set())

    def test_parser_only_widens_file_bound_and_imports_no_producer_or_device(self):
        source=a._tensor_source.replace('32*1024**2','16*1024**2',1)
        self.assertEqual(ast.dump(ast.parse(source)),ast.dump(ast.parse(inspect.getsource(a._base.tensors))))
        self.assertIs(a.raw_file.__globals__['tensors'],a.tensors)
        original=builtins.__import__
        def guarded(name,*args,**kw):
            if name.split('.')[0] in ('mlx','numpy','tutorial_code') or name.endswith('.qualify'):raise AssertionError('producer/device import')
            return original(name,*args,**kw)
        with patch('builtins.__import__',side_effect=guarded):runpy.run_path(a.__file__)

    def test_missing_or_unsuccessful_terminal_never_opens_raw_or_qualifies(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            with patch.object(a,'raw_file',side_effect=AssertionError('must not open raw')):
                self.assertEqual(a.audit(root,'0'*64)['status'],'INCOMPLETE')
                for status in ('FAIL','INCOMPLETE'):
                    (root/'complete.json').write_text(json.dumps(dict(schema='GDN-batched-prework-real-layer-result-v1',status=status,canary_qualified=False)))
                    result=a.audit(root,'0'*64)
                    self.assertEqual(result['status'],status);self.assertFalse(result['canary_qualified']);self.assertEqual(result['raw_files'],0)


if __name__=='__main__':unittest.main()
