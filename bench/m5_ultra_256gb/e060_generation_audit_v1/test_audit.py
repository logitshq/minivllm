"""Small CPU contracts; large 130-array fixtures are mocked explicitly."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('e060_independent_audit',Path(__file__).with_name('audit.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def rejected_rounds(arm,output=None):
    output=list(range(1,33)) if output is None else list(output)
    rows=[]
    for i,token in enumerate(output[1:]):
        remaining=32-i-1
        width=int(arm[-1])
        row=dict(width=width,accepted=0,delivered=1,draft_ns=10,verify_commit_ns=20,
                 draft_ids=[200]*7,posterior=[token]+[201]*(width-1),produced=[token],position_after=1025+i)
        rows.append(row)
    return rows

def timings():
    rows=[]
    for g in range(3):
        for c in range(2):
            shift=(g+c)%4
            for arm in a.ARMS[shift:]+a.ARMS[:shift]:
                output=list(range(1,33)) if c==0 else [1,2,99]
                ledger=[] if arm=='native' else rejected_rounds(arm,output)
                rows.append(dict(group=g,case=c,arm=arm,elapsed_ns=2000 if arm=='native' else 1000,
                                 output_ids=output,decoded_tokens=len(output)-1,rounds=ledger))
    return rows,{'math':list(range(1,33)),'chinese':[1,2,99]}

def blob(payload=b'\x00\x00',dtype='BF16'):
    header=json.dumps({'x':dict(dtype=dtype,shape=[1],data_offsets=[0,len(payload)])},separators=(',',':')).encode()
    return struct.pack('<Q',len(header))+header+payload

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.raw=a.parser()

    def test_default_off_no_actual_reads(self):
        with patch.object(a,'audit',side_effect=AssertionError('audit called')),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(a.main([]),0)
        self.assertFalse(json.loads(out.getvalue())['enabled_by_default'])
        with self.assertRaisesRegex(ValueError,'explicit --run'):
            a.main(['--root','/no-such-evidence'])

    def test_frozen_parser_actual_tiny_signedzero_nonfinite_hash_and_path(self):
        raw=self.raw;left=raw.parse_safetensors(blob());right=raw.parse_safetensors(blob(b'\x00\x80'))
        with self.assertRaisesRegex(ValueError,'bytes differ'):
            raw.compare_arrays({'x':('BF16',(1,))},'x',left['x'],right['x'])
        for bits in (0x7f80,0x7fc1,0xff80):
            with self.assertRaisesRegex(ValueError,'nonfinite'):raw.parse_safetensors(blob(struct.pack('<H',bits)))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tiny.safetensors';p.write_bytes(blob())
            ref=dict(path=p.name,sha256=a.sha(p))
            self.assertEqual(raw._load_member(d,'reference',ref)[1],left)
            with self.assertRaisesRegex(ValueError,'sha256 mismatch'):raw._load_member(d,'reference',dict(ref,sha256='0'*64))
            with self.assertRaises(ValueError):raw._resolve_member(d,'../tiny.safetensors')
            (Path(d)/'link').symlink_to(p)
            with self.assertRaisesRegex(ValueError,'symlink'):raw._resolve_member(d,'link')

    def test_dynamic_full_snapshot_contract(self):
        s=a.array_spec(self.raw,2)
        self.assertEqual(len(s),130);self.assertEqual(s['features'],('BF16',(2,25600)))
        self.assertEqual(s['logits'],('BF16',(1,248320)))
        self.assertEqual(s['state/63/K'],('BF16',(1,4,1536,256)))
        self.assertEqual(s['state/62/S'],('F32',(1,48,128,128)))
        for bad in (0,32,True):
            with self.assertRaises(ValueError):a.array_spec(self.raw,bad)

    def test_prefix_mismatch_eos_and_cap_accounting(self):
        self.assertEqual(a.select_independent([3,4,5],[9,4,5,6],31,{99}),(0,[9]))
        self.assertEqual(a.select_independent([3,4,5],[3,7,5,6],31,{99}),(1,[3,7]))
        self.assertEqual(a.select_independent([3,4,5],[3,4,5,6],2,{99}),(3,[3,4]))
        self.assertEqual(a.select_independent([3,99,5],[3,99,5,6],31,{99}),(3,[3,99]))

    def test_width2_accept_reject_eos_and_cap(self):
        self.assertEqual(a.select_independent([3],[3,4],31,{99}),(1,[3,4]))
        self.assertEqual(a.select_independent([3],[7,4],31,{99}),(0,[7]))
        self.assertEqual(a.select_independent([99],[99,4],31,{99}),(1,[99]))
        self.assertEqual(a.select_independent([3],[3,4],1,{99}),(1,[3]))
        rows=rejected_rounds('fixed2',output=[1,2,99]);a.trajectory(rows,[1,2,99],'fixed2',{99})
        rows[0]['posterior'].append(1)
        with self.assertRaises(ValueError):a.trajectory(rows,[1,2,99],'fixed2',{99})

    def test_width2_full_accept_then_cap_preserves_raw_accepted_count(self):
        output=list(range(1,33));rows=[];count=1
        while count<32:
            next_token=count+1;produced=list(range(next_token,min(next_token+2,33)))
            count+=len(produced)
            rows.append(dict(width=2,accepted=1,delivered=len(produced),draft_ns=10,verify_commit_ns=20,
                draft_ids=[next_token]+[201]*6,posterior=[next_token,next_token+1],produced=produced,position_after=1024+count-1))
        a.trajectory(rows,output,'fixed2',{99});self.assertEqual(len(rows),16)
        self.assertEqual(rows[-1]['accepted'],1);self.assertEqual(rows[-1]['delivered'],1)
        bad=copy.deepcopy(rows);bad[-1]['accepted']=0
        with self.assertRaisesRegex(ValueError,'raw accepted prefix'):a.trajectory(bad,output,'fixed2',{99})

    def test_round_ledger_rejects_wrong_position_tokens_width_acceptance(self):
        good=rejected_rounds('fixed4');out=list(range(1,33));a.trajectory(good,out,'fixed4',{99})
        for key,value in (('position_after',1026),('accepted',1),('produced',[3]),('delivered',2),('width',8)):
            bad=copy.deepcopy(good);bad[0][key]=value
            with self.assertRaises(ValueError):a.trajectory(bad,out,'fixed4',{99})
        with self.assertRaisesRegex(ValueError,'no speculative rounds'):a.trajectory(good,out,'native',{99})

    def test_exact_timing_schedule_ids_denominator_and_phase_bounds(self):
        rows,outputs=timings();summary,total=a.timing_check(rows,outputs,{99})
        self.assertEqual(total,30000);self.assertEqual(summary[0]['samples'],6)
        self.assertEqual(summary[1]['native_over_arm'],2.0)
        for kind in ('order','ids','denominator','phases','width'):
            bad=copy.deepcopy(rows)
            if kind=='order':bad[0],bad[1]=bad[1],bad[0]
            elif kind=='ids':bad[1]['output_ids'][1]=77
            elif kind=='denominator':bad[0]['decoded_tokens']=32
            elif kind=='phases':bad[1]['elapsed_ns']=1
            else:next(r for r in bad if r['arm']=='fixed2')['rounds'][0]['width']=8
            with self.assertRaises(ValueError):a.timing_check(bad,outputs,{99})

    def test_seed_four_checks_hash_only_six_unique_files_and_reject_tamper(self):
        names=[('math-before','math-after-qualification'),
               ('chinese-before','chinese-after-qualification'),('math-before','test0-after-timing'),('chinese-before','test1-after-timing')]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[]
            for before,after in names:
                row={}
                for side,label in (('before',before),('after',after)):
                    p=root/('seed-'+label+'.safetensors');p.write_bytes(blob())
                    row[side]=dict(path=p.name,sha256=a.sha(p))
                rows.append(row)
            with patch.object(a,'sha',wraps=a.sha) as hash_call:
                seen,size=a.seed_check(self.raw,root,rows,{})
                self.assertEqual(hash_call.call_count,6);self.assertEqual(len(seen),6);self.assertEqual(size,6*len(blob()))
            (root/rows[-1]['after']['path']).write_bytes(blob(b'\x00\x80'))
            with self.assertRaisesRegex(ValueError,'actual original seed file SHA'):a.seed_check(self.raw,root,rows,{})

    def test_mock_full_raw_contract_counts_and_last_greedy_logit(self):
        # Only orchestration is mocked. Real raw parser tests above exercise bytes.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pairs=[];all_arrays={}
            logits=bytearray(a.VOCAB*2);logits[2*99:2*99+2]=struct.pack('<H',0x3f80)
            for label in a.labels():
                pair=dict(label=label,output_ids=[1,2,99],computed_rows=2)
                arrays={n:(dt,shape,bytes(logits) if n=='logits' else b'\0\0') for n,(dt,shape) in a.array_spec(self.raw,2).items()}
                for side in ('reference','candidate'):
                    rel='raw/'+label+'-'+side+'.safetensors';p=root/rel;p.parent.mkdir(exist_ok=True);p.write_bytes(b'fixture')
                    pair[side]=dict(path=rel,sha256=a.sha(p));all_arrays[rel]=arrays.copy()
                pairs.append(pair)
            def load(root,side,ref):return ref['path'],all_arrays[ref['path']],7
            mock=SimpleNamespace(expected_arrays=self.raw.expected_arrays,_resolve_member=self.raw._resolve_member,
                                 _load_member=load,compare_arrays=self.raw.compare_arrays)
            def result():return dict(raw_files=0,tensor_pairs=0,pairs=[],raw_bytes=0)
            manifest=dict(schema='E060-generation-pairs-v1',pairs=pairs);r=result()
            outputs,seen=a.raw_pairs(mock,root,manifest,r)
            self.assertEqual((r['raw_files'],r['tensor_pairs'],len(seen)),(16,1040,16));self.assertEqual(outputs['math'],[1,2,99])
            bad=copy.deepcopy(manifest);bad['pairs'][0]['computed_rows']=3
            with self.assertRaisesRegex(ValueError,'computed count'):a.raw_pairs(mock,root,bad,result())
            ref=all_arrays[pairs[-1]['candidate']['path']];name='state/63/V';dt,shape,data=ref[name];ref[name]=(dt,shape,b'\x00\x80')
            with self.assertRaisesRegex(ValueError,'bytes differ'):a.raw_pairs(mock,root,manifest,result())

    def test_parameter_invariance_ignores_only_handle_identity(self):
        row={'w':dict(shape=[1],dtype='mlx.core.bfloat16',bytes=2,sha256='a'*64,finite=True,handle_id=1)}
        other=copy.deepcopy(row);other['w']['handle_id']=2
        self.assertEqual(a.parameter_values(row),a.parameter_values(other))
        other['w']['sha256']='b'*64;self.assertNotEqual(a.parameter_values(row),a.parameter_values(other))
        other['w']['finite']=False
        with self.assertRaises(ValueError):a.parameter_values(other)

    def test_closed_restore_gate_runs_before_tensor_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'e060-short-verify-v1-numerical';root.mkdir();plan=root/'plan.json';plan.write_text('{}')
            closed=root.parent/'e060-short-verify-v1-execution';closed.mkdir();(closed/'final.json').write_text('{"status":"FAIL"}')
            with patch.object(a,'parser',side_effect=AssertionError('raw parser invoked')):
                report=a.audit(root,plan,a.sha(plan))
            self.assertEqual(report['status'],'FAIL');self.assertEqual(report['raw_files'],0)
            self.assertIn('closed and restored',report['error']['message'])

    def test_failure_receipt_and_fresh_output_cli(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);out=root/'audit-fail.json';plan=root/'plan.json';plan.write_text('{}')
            args=['--run','--root',str(root),'--plan',str(plan),'--plan-sha256',a.sha(plan),'--out',str(out)]
            self.assertEqual(a.main(args),1);report=json.loads(out.read_text())
            self.assertEqual(report['status'],'FAIL');self.assertFalse(report['generation_bytes_audited'])
            with self.assertRaisesRegex(ValueError,'fresh output'):a.main(args)

if __name__=='__main__':unittest.main()
