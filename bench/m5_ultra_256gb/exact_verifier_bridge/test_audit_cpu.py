"""Focused stdlib fake/raw-byte tests; source prepared only, no device claims.

No model-size payloads. Real tiny safetensors exercise the frozen reader; full
layout tests use metadata only. These are not a synthetic twelve-case MLX PASS.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.exact_verifier_bridge import audit as a


def save(root, name, tensors):
    header, descriptors, body = {}, {}, b''
    for key,(encoding,shape,raw) in tensors.items():
        start = len(body); body += raw
        header[key] = dict(dtype=encoding,shape=shape,data_offsets=[start,len(body)])
        descriptors[key] = dict(shape=shape,dtype=a.DTYPES[encoding],encoding=encoding,bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),finite=True)
    encoded = json.dumps(header,separators=(',',':')).encode()
    path = root/name; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+body)
    desc = dict(path=name,sha256=a.sha(path),finite=True,tensors=descriptors,metadata={})
    path.with_suffix('.json').write_text(json.dumps(desc))
    return desc


def fixture(name='short', size=886):
    return dict(fixture_id=name,prompt_ids=[100]*size,normal_eos=True,max_output_tokens=512,eos_token_ids=[999])


def seed(f, start):
    raw = list(range(100,100+start+8))
    return dict(fixture_id=f['fixture_id'],start=start,prompt_ids=f['prompt_ids'],raw_ids=raw,
                tokens=raw[start-1:start+7],position=len(f['prompt_ids'])+start-1)


def check_rows(row):
    """Independent known producer order, using metadata only for large shapes."""
    tag = f"{row['fixture_id']}/s{row['start']}/T{row['width']}"
    inputs = a.expected_layout('input'); outputs = a.expected_layout('output',row['width'])
    staging = {k.replace('root/state/','root/'):(s,d) for k,(s,d) in inputs.items() if k.startswith('root/state/')}
    result = []
    for label,values in [('same-seed',inputs),('staging-representation',staging),('full-output',outputs),
                         ('reference-seed-unchanged',inputs),('candidate-seed-unchanged',inputs)]:
        for key,(shape,dtype) in values.items():
            result.append(dict(check=tag+'/'+label,name=key,shape=shape,dtype=a.DTYPES[dtype],
                reference_finite=True,candidate_finite=True,bit_equal=True))
    return result


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_contract_matches_actual_source_but_never_imports_runtime(self):
        from bench.m5_ultra_256gb.exact_verifier_bridge import qualify
        a.same(a.expected_contract(),qualify.contract(),'actual producer contract')
        self.assertEqual(a.expected_contract()['expected_cases'],12)
        self.assertFalse(a.expected_contract()['rollback_qualified'])
        self.assertFalse(a.expected_contract()['performance_qualified'])

    def test_real_unsigned_bits_reject_signed_zero_and_f32_lsb_with_location(self):
        b = a.raw_reader()
        for encoding,format,words,changed,name in (
            ('BF16','H',[0x3f80,0],[0x3f80,0x8000],'root/logits'),
            ('F32','I',[0x3f800000,0x3f800000],[0x3f800000,0x3f800001],'root/state/62/S')):
            left = save(self.root,'left.safetensors',{name:(encoding,[2],struct.pack('<2'+format,*words))})
            right = save(self.root,'right.safetensors',{name:(encoding,[2],struct.pack('<2'+format,*changed))})
            x,y = b.snapshot_index(self.root,left),b.snapshot_index(self.root,right)
            self.assertEqual(a.compare_pair(b,x,x,'same')['tensor_count'],1)
            with self.subTest(dtype=encoding),self.assertRaises(a.BitMismatch) as caught:
                a.compare_pair(b,x,y,'fixed/T2')
            info = caught.exception.details
            self.assertEqual(info['array'],name); self.assertEqual(info['coordinates'],[1])
            self.assertEqual(info['layer'],62 if encoding == 'F32' else None)
            self.assertEqual(info['byte_offset'],4 if encoding == 'F32' else 3)
            # Same numerical zero/greedy would not pass this gate.
            self.assertEqual(info['differing_words'],1)

    def test_raw_reader_rejects_nonfinite_even_with_all_refreshed_hashes(self):
        b = a.raw_reader()
        for encoding,word,fmt in [('BF16',0x7f80,'H'),('BF16',0xffc0,'H'),('F32',0x7f800001,'I')]:
            desc = save(self.root,'bad.safetensors',{'root/state/0/S':(encoding,[1],struct.pack('<'+fmt,word))})
            with self.subTest(dtype=encoding,word=word),self.assertRaisesRegex(ValueError,'finite'):
                b.snapshot_index(self.root,desc)

    def test_original_reader_rejects_stale_sha_overlap_and_changes_after_index(self):
        b = a.raw_reader()
        desc = save(self.root,'good.safetensors',{'x':('I32',[1],struct.pack('<i',257))})
        index = b.snapshot_index(self.root,desc)
        path = self.root/desc['path']; raw = path.read_bytes(); path.write_bytes(raw[:-1]+b'\x01')
        with self.assertRaisesRegex(ValueError,'changed'): b.tensor_bytes(index,'x')
        with self.assertRaisesRegex(ValueError,'SHA'): b.snapshot_index(self.root,desc)
        header = {'x':dict(dtype='I32',shape=[1],data_offsets=[1,5])}
        encoded = json.dumps(header).encode(); path.write_bytes(struct.pack('<Q',len(encoded))+encoded+struct.pack('<i',257))
        desc['sha256'] = a.sha(path)
        with self.assertRaises(ValueError): b.snapshot_index(self.root,desc)

    def test_full_original_layout_rejects_last_layer_missing_trimmed_kv_and_cast(self):
        for kind,width in [('input',1),('output',2),('output',8)]:
            expected = a.expected_layout(kind,width)
            index = dict(tensors={k:dict(shape=s,encoding=d) for k,(s,d) in expected.items()})
            a.layout(index,kind,width)
            self.assertEqual(len(expected),131 if kind == 'input' else 129)
            for change in ('missing','trim','cast','bool'):
                bad = copy.deepcopy(index)
                if change == 'missing': bad['tensors'].pop('root/state/63/V')
                elif change == 'trim': bad['tensors']['root/state/63/V']['shape'][2] = 978
                elif change == 'cast': bad['tensors']['root/state/62/S']['encoding'] = 'BF16'
                else: bad['tensors']['root/state/62/S']['shape'][0] = True
                with self.subTest(kind=kind,change=change),self.assertRaises(ValueError): a.layout(bad,kind,width)

    def test_teacher_causality_exact_anchor_and_normal_eos_boundary(self):
        f = fixture(); row = seed(f,129); a.teacher(row,f,129)
        for field,value in [('tokens',[0]*8),('position',row['position']+1),('start',True)]:
            bad = copy.deepcopy(row); bad[field] = value
            with self.subTest(field=field),self.assertRaises(ValueError): a.teacher(bad,f,129)
        final = copy.deepcopy(row); final['raw_ids'][-1] = 999
        a.teacher(final,f,129)  # Final produced EOS is allowed; it is not a subsequent teacher input.
        before = copy.deepcopy(row); before['raw_ids'][-2] = 999; before['tokens'][-1] = 999
        with self.assertRaisesRegex(ValueError,'EOS'): a.teacher(before,f,129)
        short = copy.deepcopy(row); short['raw_ids'].pop()
        with self.assertRaisesRegex(ValueError,'complete'): a.teacher(short,f,129)

    def test_independent_selection_uses_ties_and_fixed_twelve_case_cross_product(self):
        values = [fixture('D'+str(i),886+i) for i in range(8)]
        values[6]['prompt_ids'] = values[7]['prompt_ids'][:]
        chosen = a.selected(values)
        self.assertEqual([(f['fixture_id'],s) for f,s in chosen],[('D0',1),('D0',129),('D7',1),('D7',129)])
        self.assertEqual(len(chosen)*len(a.WIDTHS),12)
        values[-1]['max_output_tokens'] = True
        with self.assertRaises(ValueError): a.selected(values)

    def test_missing_or_false_late_device_check_cannot_be_hidden_by_raw_match(self):
        row = dict(fixture_id='D0',start=129,width=8)
        row['checks'] = check_rows(row); self.assertEqual(len(row['checks']),650); a.device_checks(row)
        for change in ('missing','false','scalar-type','order'):
            bad = copy.deepcopy(row)
            if change == 'missing': bad['checks'].pop()
            elif change == 'false': bad['checks'][-1]['bit_equal'] = False
            elif change == 'scalar-type': bad['checks'][-1]['bit_equal'] = 1
            else: bad['checks'][0],bad['checks'][1] = bad['checks'][1],bad['checks'][0]
            with self.subTest(change=change),self.assertRaises(ValueError): a.device_checks(bad)

    def test_raw_argmax_uses_all_vocabulary_and_remains_separate_from_bit_gate(self):
        b = a.raw_reader(); words = [0]*248320; words[-1] = 0x4000
        first = save(self.root,'full.safetensors',{'root/logits':('BF16',[1,248320],struct.pack('<248320H',*words))})
        words[1] = 0x3f80
        second = save(self.root,'changed.safetensors',{'root/logits':('BF16',[1,248320],struct.pack('<248320H',*words))})
        x,y = b.snapshot_index(self.root,first),b.snapshot_index(self.root,second)
        self.assertEqual(a.greedy(b,x,1),[248319]); self.assertEqual(a.greedy(b,y,1),[248319])
        with self.assertRaises(a.BitMismatch): a.compare_pair(b,x,y,'same-greedy')

    def test_event_order_uses_real_teacher_edges_not_counts(self):
        f = fixture(); s = seed(f,1)
        rows = [dict(fixture_id='short',start=1,width=w) for w in a.WIDTHS]
        complete = dict(reference_blocks=[s],cases=rows,elapsed_wall_s=100)
        events = [dict(event='memory',phase='before-load',active_bytes=0,peak_bytes=0,cache_bytes=0,not_performance=True)]
        events += [dict(event='reference-token',fixture_id='short',start=1,decode_step=i,
            input_token=s['raw_ids'][i-1],output_token=s['raw_ids'][i]) for i in range(1,9)]
        events += [dict(event='case',**row) for row in rows]
        events += [dict(event='memory',phase='after-release',active_bytes=0,peak_bytes=0,cache_bytes=0,not_performance=True)]
        def write(values):
            (self.root/'events.jsonl').write_text(''.join(json.dumps(dict(**row,elapsed_wall_s=i))+'\n' for i,row in enumerate(values)))
        write(events); a.events(self.root,complete)
        reordered = copy.deepcopy(events); reordered[1],reordered[2] = reordered[2],reordered[1]
        write(reordered)
        with self.assertRaisesRegex(ValueError,'binding'): a.events(self.root,complete)
        write(events+[events[-1]])
        with self.assertRaisesRegex(ValueError,'after release'): a.events(self.root,complete)

    def test_fake_producer_pass_and_missing_terminal_never_grant_forward_qualification(self):
        missing = a.audit(self.root,'0'*64)
        self.assertEqual(missing['status'],'INCOMPLETE'); self.assertFalse(missing['forward_qualified'])
        path = self.root/'complete.json'
        data = dict(schema='Exact-verifier-bridge-result-v1',status='PASS',forward_qualified=True)
        path.write_text(json.dumps(data)); before = path.read_bytes()
        with patch.object(a,'raw_reader',side_effect=AssertionError('no numerical evidence')):
            result = a.audit(self.root,'0'*64)
        self.assertNotEqual(result['status'],'PASS'); self.assertFalse(result['forward_qualified'])
        self.assertEqual(path.read_bytes(),before)
        for flag in ('rollback_qualified','full_model_qualified','http_qualified','performance_qualified'):
            self.assertFalse(result[flag])

    def test_strict_json_and_fresh_output_leave_first_attempt_untouched(self):
        for value in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
            with self.assertRaises(ValueError): a.parse(value)
        out = self.root/'original-audit.json'; out.write_text('first attempt\n')
        with patch.object(a,'audit',side_effect=AssertionError('must refuse existing output')):
            with self.assertRaisesRegex(ValueError,'fresh'):
                a.main(['--child',str(self.root),'--plan-sha256','0'*64,'--out',str(out)])
        self.assertEqual(out.read_text(),'first attempt\n')


if __name__ == '__main__': unittest.main()
