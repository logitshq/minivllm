"""CPU fixtures only. Never execute the screen's device path or open a socket."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('shared_w_screen_cpu',HERE/'screen.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class ScreenCPU(unittest.TestCase):
    def test_matrix_is_bounded_and_real_1024_is_not_invented(self):
        for real,count in ((False,27),(True,33)):
            contract=s.contract(real)
            self.assertEqual(contract['expected_cases'],count)
            self.assertEqual(contract['expected_comparisons'],count*2)
            self.assertEqual(contract['expected_phases'],count*2+2)
            self.assertEqual(len({r['case'] for r in contract['cases']}),count)
            self.assertEqual(contract['cache_limit_bytes'],0)
            self.assertEqual(contract['mlx_observation_budget_bytes'],1<<30)
            self.assertEqual(contract['physical_observation_budget_bytes'],2<<30)
            self.assertFalse(contract['full_model_qualified'])
            self.assertFalse(contract['performance_qualified'])
            actual=[r for r in contract['cases'] if r['real_weights']]
            self.assertEqual({r['N'] for r in actual},{48,10240} if real else set())
            self.assertEqual({r['T'] for r in contract['cases']},{2,4,5})

    def test_cli_no_device_import_socket_or_output_without_run(self):
        code='''
import importlib.abc,importlib.util,socket,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch','tutorial_code'}:raise AssertionError(name)
sys.meta_path.insert(0,Block());socket.socket=lambda *a,**k:(_ for _ in ()).throw(AssertionError('socket'))
sp=importlib.util.spec_from_file_location('screen',sys.argv[1]);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
raise SystemExit(m.main(sys.argv[2:]))
'''
        for args,expected in (([],0),(['--help'],0),(['--run'],2),(['--out','unused'],2),
                              (['--run','--out','unused','--bundle-manifest','missing'],2)):
            value=subprocess.run([sys.executable,'-I','-B','-c',code,str(HERE/'screen.py'),*args],
                                 capture_output=True,text=True,timeout=5)
            self.assertEqual(value.returncode,expected,value.stderr)

    def test_synthetic_banks_preserve_m4_signed_codes_and_repeated_rows(self):
        import numpy as np
        for bank in s.BANKS:
            x,w,sc=s.synthetic(np,8,1024,bank,124000)
            self.assertEqual((x.shape,w.shape,sc.shape),((5,1024),(8,128),(8,64)))
            self.assertEqual(w.dtype,np.uint32);self.assertEqual(sc.dtype,np.uint8)
            self.assertEqual(set(np.unique(sc).tolist()),set(s.SCALE_CODES))
            again=s.synthetic(np,8,1024,bank,124000)
            self.assertTrue(all(np.array_equal(a,b) for a,b in zip((x,w,sc),again)))
            if bank=='random':
                self.assertEqual(x.dtype,np.float32);self.assertTrue(np.isfinite(x).all())
            else:
                self.assertEqual(x.dtype,np.uint16);self.assertTrue(np.array_equal(x[0],x[2]))
                self.assertIn(0x8000,x);self.assertIn(0,x)
            if bank=='codes_signed_boundaries':
                codes=set((int(word)>>shift)&15 for word in w.flat for shift in range(0,32,4))
                self.assertEqual(codes,set(range(16)))
            if bank=='extreme_cancellation':
                self.assertTrue({0,0x8000,1,0x8001,0x5380,0xd380,0x3f80,0xbf80}.issubset(set(x.flat)))

    def test_bundle_reads_only_selected_original_bytes_and_rejects_corruption(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);raw=b'pad!'+struct.pack('<2I',0x76543210,0xfedcba98)+bytes([0x38])+b'unused'
            (out/'weights.bin').write_bytes(raw)
            rows=[dict(parameter='in_proj_a.weight',bundle_offsets=[4,12],bytes=8,shape=[1,2],
                       sha256=hashlib.sha256(raw[4:12]).hexdigest()),
                  dict(parameter='in_proj_a.scales',bundle_offsets=[12,13],bytes=1,shape=[1,1],
                       sha256=hashlib.sha256(raw[12:13]).hexdigest())]
            weight,scales=s.bundle_pair(np,out/'manifest.json',{'tensors':rows},'in_proj_a')
            self.assertEqual(weight.tolist(),[[0x76543210,0xfedcba98]])
            self.assertEqual(scales.tolist(),[[0x38]])
            bad=copy.deepcopy(rows);bad[0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'tensor changed'):
                s.bundle_pair(np,out/'manifest.json',{'tensors':bad},'in_proj_a')
            with self.assertRaisesRegex(ValueError,'exact bundle projection'):
                s.bundle_pair(np,out/'manifest.json',{'tensors':rows},'missing')

    def test_compare_rejects_signed_zero_nonfinite_and_preserves_first_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            def artifact(name,words,finite=True):
                raw=struct.pack('<2H',*words);(out/name).write_bytes(raw)
                return dict(file=name,shape=[1,2],dtype='bfloat16',strides=[2,1],bytes=len(raw),
                            encoding='uint16-le',sha256=hashlib.sha256(raw).hexdigest(),finite=finite,owner_id=1)
            a=artifact('a.bin',[0,0x3f80]);b=artifact('b.bin',[0x8000,0x3f80])
            self.assertTrue(s.compare(out,'case','equal',a,a)['exact'])
            failed=s.compare(out,'case','signed-zero',a,b)
            self.assertFalse(failed['exact']);self.assertEqual(failed['first_differing_byte'],1)
            first=(out/'first-difference.json').read_bytes()
            nonfinite=artifact('nonfinite.bin',[0x7f80,0x7fc1],False)
            self.assertFalse(s.compare(out,'case','nonfinite',nonfinite,nonfinite)['exact'])
            self.assertEqual((out/'first-difference.json').read_bytes(),first)
            (out/'a.bin').write_bytes(b'bad!')
            with self.assertRaisesRegex(ValueError,'saved output bytes/SHA'):
                s.compare(out,'case','tampered',a,b)

    def test_actual_cpu_buffer_metadata_and_identity_no_hidden_normalization(self):
        import numpy as np
        array=np.arange(16,dtype=np.uint32).reshape(2,8)
        mx=NS(bfloat16='bf16',uint16='u16',uint32=array.dtype,uint8=np.dtype('u1'),eval=lambda *args:None)
        class Value:
            dtype='bf16';shape=(1,2);nbytes=4
            def view(self,dtype):return np.array([0,0x8000],dtype='<u2').reshape(1,2)
        strides,raw=s.observe(mx,Value())
        self.assertEqual(strides,(2,1));self.assertEqual(raw,b'\x00\x00\x00\x80')
        with tempfile.TemporaryDirectory() as tmp:
            record=s.array_record(mx,array,Path(tmp),'weight.bin')
            self.assertEqual(record['owner_id'],id(array));self.assertEqual(record['strides'],[8,1])
            self.assertEqual((Path(tmp)/'weight.bin').read_bytes(),array.tobytes())
            with self.assertRaisesRegex(ValueError,'canonical host buffer'):
                s.observe(mx,array[:,::2])

    def test_gpu_descriptor_witness_is_from_probe_not_host_strides(self):
        import numpy as np
        value=np.arange(16,dtype=np.uint32).reshape(2,8)
        mx=NS(uint64='u64',eval=lambda *args:None)
        launches=[]
        def kernel(**kwargs):
            launches.append(kwargs)
            return [np.array([2,8,8,1],dtype='<u8')]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            record=s.gpu_layout(mx,kernel,value,out,'layout.bin','weight')
            self.assertIs(launches[0]['inputs'][0],value)
            self.assertEqual(launches[0]['output_dtypes'],['u64'])
            self.assertEqual(record['actual_strides_elements'],[8,1])
            self.assertFalse(record['ensure_row_contiguous'])
            self.assertEqual((out/'layout.bin').read_bytes(),struct.pack('<4Q',2,8,8,1))
            self.assertEqual(json.loads((out/'layout.json').read_text()),record)
            def bad_kernel(**kwargs):return [np.array([2,8,16,2],dtype='<u8')]
            with self.assertRaisesRegex(ValueError,'actual GPU descriptor'):
                s.gpu_layout(mx,bad_kernel,value,out,'invalid.bin','weight')
            self.assertTrue((out/'invalid.bin').exists())

    def test_phase_retains_three_raw_readings_and_checks_observed_budget(self):
        support=s.load('shared_w_test_footprint',s.SUPPORT/'layer_screen.py')
        raw={'unit':'byte','bytes per unit':1,'errors':[],'warnings':[],
             'processes':[{'pid':os.getpid(),'footprint':110,
                           'auxiliary':{'phys_footprint':100,'phys_footprint_peak':105}}]}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);result={'physical_receipts':[]}
            mx=NS(get_active_memory=lambda:0,get_peak_memory=lambda:1<<30,get_cache_memory=lambda:0)
            def probe(command,**kwargs):
                Path(command[-1]).write_text(json.dumps(raw));return NS(returncode=0,stdout='',stderr='')
            with patch.object(s.subprocess,'run',side_effect=probe):
                ledger=s.Ledger(out,result,mx,support,0);ledger.phase('initial')
                mx.get_peak_memory=lambda:(1<<30)+1
                with self.assertRaisesRegex(ValueError,'1GiB'):ledger.phase('after_case','c')
            rows=[json.loads(line) for line in (out/'phases.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['physical_top_bytes'],110)
            self.assertEqual(rows[0]['physical_aux_peak_bytes'],105)
            self.assertEqual(rows[0]['physical_current_bytes'],110)
            self.assertEqual(rows[0]['physical_peak_bytes'],110)
            self.assertEqual([r['status'] for r in rows],['PASS','FAIL'])

    def test_timeout_and_cancellation_are_distinct(self):
        with self.assertRaises(TimeoutError):s.stop_signal(signal.SIGALRM)
        for sig in (signal.SIGTERM,signal.SIGINT):
            with self.assertRaises(InterruptedError) as caught:s.stop_signal(sig)
            self.assertNotIn('120s',str(caught.exception))


if __name__=='__main__':unittest.main()
