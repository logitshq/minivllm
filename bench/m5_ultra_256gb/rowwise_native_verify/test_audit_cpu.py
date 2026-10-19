"""Stdlib/tiny-byte auditor tests; never a synthetic full-model qualification."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.rowwise_native_verify import audit as a


def save(root,name,encoding,shape,words,key='root/logits'):
    raw=struct.pack('<'+str(len(words))+('H' if encoding=='BF16' else 'I'),*words)
    header={key:dict(dtype=encoding,shape=shape,data_offsets=[0,len(raw)])}
    encoded=json.dumps(header).encode(); path=root/name
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+raw)
    return dict(path=name,sha256=a.sha(path),finite=True,metadata={},tensors={key:dict(
        shape=shape,dtype=a.DTYPES[encoding],encoding=encoding,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),finite=True)})


def fixture(name='D0',size=886):
    return dict(fixture_id=name,prompt_ids=[100]*size,normal_eos=True,max_output_tokens=512,eos_token_ids=[999])


def seed(f,used):
    prompt=f['prompt_ids'][:256] if used==257 else f['prompt_ids']; start=used-len(prompt)
    raw=list(range(100,100+start+8))
    return dict(fixture_id=f['fixture_id'],original_fixture=f,prompt_ids=prompt,
        domain='diagnostic-prefix256' if used==257 else 'original-full-prompt',used=used,position=used-1,
        start=start,backing=1536,raw_ids=raw,tokens=raw[start-1:start+7],prefill_token=raw[0])


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name).resolve()

    def test_default_describes_only_and_no_qualifier_or_backend_import(self):
        tree=ast.parse(Path(a.__file__).read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
        self.assertFalse(any(x.startswith(('mlx','numpy')) or 'qualify' in x for x in imports))
        with patch.object(a,'audit',side_effect=AssertionError('describe must not audit')),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(a.main([]),0)
        value=json.loads(out.getvalue()); self.assertEqual(value['output_tensor_pairs'],42*129)
        self.assertEqual(value['raw_files'],3*(2+8)+9*2+42+6)
        self.assertTrue(all(value[k] is False for k in a.EXCLUDED))

    def test_original_bits_reject_signed_zero_and_late_recurrent_word(self):
        b=a.raw_reader()
        for encoding,word,key in [('BF16',0x8000,'root/logits'),('F32',1,'root/state/62/S')]:
            left=save(self.root,'left.safetensors',encoding,[2],[0,0],key)
            right=save(self.root,'right.safetensors',encoding,[2],[0,word],key)
            x,y=b.snapshot_index(self.root,left),b.snapshot_index(self.root,right)
            self.assertEqual(a.compare_pair(b,x,x,'same')['tensor_count'],1)
            with self.subTest(dtype=encoding),self.assertRaises(a.BitMismatch) as caught:
                a.compare_pair(b,x,y,'used1024/T8/p8')
            self.assertEqual(caught.exception.details['coordinates'],[1]); self.assertEqual(caught.exception.details['array'],key)
            self.assertEqual(caught.exception.details['layer'],62 if encoding=='F32' else None)

    def test_refreshed_hashes_do_not_hide_nonfinite_or_truncated_storage(self):
        b=a.raw_reader()
        for encoding,word in [('BF16',0x7f80),('BF16',0xffc0),('F32',0x7f800001)]:
            desc=save(self.root,'bad.safetensors',encoding,[1],[word])
            with self.subTest(dtype=encoding,word=word),self.assertRaisesRegex(ValueError,'finite'):
                b.snapshot_index(self.root,desc)
        desc=save(self.root,'truncated.safetensors','I32',[2],[257]); desc['tensors']['root/logits']['shape']=[2]
        with self.assertRaisesRegex(ValueError,'bounds/shape'): b.snapshot_index(self.root,desc)

    def test_raw_stale_sha_and_postindex_mutation_rejected(self):
        b=a.raw_reader(); desc=save(self.root,'one.safetensors','I32',[1],[257]); index=b.snapshot_index(self.root,desc)
        path=self.root/desc['path']; raw=path.read_bytes(); path.write_bytes(raw[:-1]+b'\x01')
        with self.assertRaisesRegex(ValueError,'changed'): b.tensor_bytes(index,'root/logits')
        with self.assertRaisesRegex(ValueError,'SHA'): b.snapshot_index(self.root,desc)

    def test_full_original_layout_includes_all_padding_and_rejection_shapes(self):
        for kind,width,batch in [('canonical',1,1),('input',8,1),('input',2,2),('output',1,1)]:
            expected=a.expected_layout(kind,width,batch)
            index=dict(tensors={k:dict(shape=s,encoding=d) for k,(s,d) in expected.items()}); a.layout(index,kind,width,batch)
            self.assertEqual(len(expected),129 if kind=='output' else 131)
            for change in ('missing','trim','cast','bool'):
                bad=copy.deepcopy(index)
                if change=='missing': bad['tensors'].pop('root/state/63/V')
                elif change=='trim': bad['tensors']['root/state/63/K']['shape'][2]=1024
                elif change=='cast': bad['tensors']['root/state/62/S']['encoding']='BF16'
                else: bad['tensors']['root/state/0/conv']['shape'][0]=True
                with self.subTest(kind=kind,change=change),self.assertRaises(ValueError): a.layout(bad,kind,width,batch)

    def test_absolute_prefix_metadata_and_input_domain(self):
        output=a.metadata('output',1030,token=77,prefix=8)
        self.assertEqual((output['position'],output['lens'],output['max_len'],output['input_token'],output['prefix']),(1030,1031,1031,77,8))
        canonical=a.metadata('canonical',256); entered=a.metadata('input',256,8)
        self.assertEqual(set(entered)-set(canonical),{'mode','max_len','prefill_shape'})
        self.assertEqual(entered['mode'],'decode'); self.assertEqual(entered['max_len'],257)
        self.assertEqual(output['state'][63],{'K':{'array':'root/state/63/K'},'V':{'array':'root/state/63/V'}})

    def test_teacher_uses_exact_diagnostic_prompt_and_no_forced_eos(self):
        f=fixture()
        for used in a.USED:
            row=seed(f,used); a.teacher(row,f,used)
            for field,value in [('used',True),('position',row['position']+1),('tokens',[0]*8),('prefill_token',False)]:
                bad=copy.deepcopy(row); bad[field]=value
                with self.subTest(used=used,field=field),self.assertRaises(ValueError): a.teacher(bad,f,used)
            final=copy.deepcopy(row); final['raw_ids'][-1]=999; a.teacher(final,f,used)
            bad=copy.deepcopy(row); bad['raw_ids'][-2]=999; bad['tokens'][-1]=999
            with self.assertRaisesRegex(ValueError,'EOS'): a.teacher(bad,f,used)
        bad=seed(f,257); bad['prompt_ids']=f['prompt_ids']
        with self.assertRaisesRegex(ValueError,'seed domain'): a.teacher(bad,f,257)

    def test_shortest_fixture_selection_is_stable_and_original_contract_is_strict(self):
        fixtures=[fixture('D'+str(i),886+i) for i in range(8)]; fixtures[1]['prompt_ids']=fixtures[0]['prompt_ids'][:]
        self.assertEqual(a.selected(list(reversed(fixtures)))['fixture_id'],'D0')
        fixtures[0]['max_output_tokens']=True
        with self.assertRaises(ValueError): a.selected(fixtures)

    def test_argmax_checks_whole_vocabulary_but_never_replaces_bits(self):
        b=a.raw_reader(); words=[0]*248320; words[-1]=0x4000
        first=save(self.root,'first.safetensors','BF16',[1,248320],words)
        words[1]=0x3f80; second=save(self.root,'second.safetensors','BF16',[1,248320],words)
        x,y=b.snapshot_index(self.root,first),b.snapshot_index(self.root,second)
        self.assertEqual(a.greedy(b,x),248319); self.assertEqual(a.greedy(b,y),248319)
        with self.assertRaises(a.BitMismatch): a.compare_pair(b,x,y,'same-argmax')

    def test_events_bind_teacher_edges_and_fixed_case_order(self):
        s=seed(fixture(),257); complete=dict(reference_blocks=[s],elapsed_wall_s=100)
        memory=dict(active_bytes=0,peak_bytes=0,cache_bytes=0,not_performance=True)
        events=[dict(event='memory',phase='before-load',**memory)]
        events += [dict(event='reference-token',used=257,fixture_id='D0',decode_step=i,
            input_token=s['raw_ids'][i-1],output_token=s['raw_ids'][i]) for i in range(1,len(s['raw_ids']))]
        events += [dict(event='case',used=257,width=w,status='PASS') for w in a.WIDTHS]
        events += [dict(event='memory',phase='after-release',**memory)]
        def write(rows): (self.root/'events.jsonl').write_text(''.join(json.dumps(dict(**row,elapsed_wall_s=i))+'\n' for i,row in enumerate(rows)))
        write(events); a.events(self.root,complete)
        bad=copy.deepcopy(events); bad[1]['input_token']=0; write(bad)
        with self.assertRaisesRegex(ValueError,'binding'): a.events(self.root,complete)
        write(events+[events[-1]])
        with self.assertRaisesRegex(ValueError,'after release'): a.events(self.root,complete)

    def test_missing_and_fabricated_pass_never_qualify_and_do_not_mutate_child(self):
        result=a.audit(self.root,'0'*64); self.assertEqual(result['status'],'INCOMPLETE')
        path=self.root/'complete.json'; path.write_text(json.dumps(dict(schema='Rowwise-native-verifier-result-v1',status='PASS',forward_prefix_qualified=True)))
        before=path.read_bytes()
        with patch.object(a,'raw_reader',side_effect=AssertionError('no evidence')): result=a.audit(self.root,'0'*64)
        self.assertNotEqual(result['status'],'PASS'); self.assertFalse(result['forward_prefix_qualified']); self.assertEqual(path.read_bytes(),before)
        self.assertTrue(all(result[k] is False for k in a.EXCLUDED))

    def test_strict_json_pin_literals_and_fresh_output(self):
        for text in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
            with self.assertRaises(ValueError): a.parse(text)
        source=self.root/'pins.py'; source.write_text("CANDIDATE_SHA='UNSEALED'\n")
        self.assertEqual(a.literal(source,'CANDIDATE_SHA'),'UNSEALED')
        source.write_text("CANDIDATE_SHA='one'\nCANDIDATE_SHA='two'\n")
        with self.assertRaises(ValueError): a.literal(source,'CANDIDATE_SHA')
        out=self.root/'audit.json'; out.write_text('first attempt\n')
        with patch.object(a,'audit',side_effect=AssertionError('must refuse existing output')):
            with self.assertRaisesRegex(ValueError,'fresh'):
                a.main(['--child',str(self.root),'--plan-sha256','0'*64,'--out',str(out)])
        self.assertEqual(out.read_text(),'first attempt\n')


if __name__=='__main__': unittest.main()
