"""CPU-only CAP64 canonical bytes, native inverse-diff and inert source guards."""
import ast
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
D=Path(__file__).resolve().parent;P=D.parent;N=P/"v89-native-capacity64-build1"
ns=runpy.run_path(str(D/"fullqual.py"),run_name="cpu_import_only")
canonical=ns["canonical_output"]
class Checks(unittest.TestCase):
 def test_native_exact_inverse(self):
  s=(N/"replay.mm").read_text();self.assertEqual(s.count('(capacity!=64&&capacity!=256)'),1)
  inverse=s.replace('(capacity!=64&&capacity!=256)','capacity!=256').replace('fixed capacity64-or256 and complete input/output/link counts','fixed capacity256 and complete input/output/link counts')
  self.assertEqual(inverse,(P/"v40-native-icb-build2/replay.mm").read_text())
 def test_capacity_and_uint32_guard(self):
  native=object.__new__(ns["Native"]);seen=[];native.lib=SimpleNamespace(v34_seal_feedback=lambda *v:seen.append(v) or 0)
  for cap in (64,256):native.call('v34_seal_feedback',128,cap)
  self.assertEqual(seen,[(128,64),(128,256)])
  for cap in (0,1,32,63,65,128,257,0xffffffff,-1,0x100000000,0x100000040,1<<80):
   with self.assertRaises(AssertionError):native.call('v34_seal_feedback',128,cap)
  self.assertEqual(seen,[(128,64),(128,256)])
 def test_canonical_per_head_complete256(self):
  valid=51;row=512;heads=[bytes([h+1])*(valid*row)+bytes((64-valid)*row) for h in range(4)]
  raw=b''.join(heads);out,shape=canonical(raw,[1,4,64,256],'mlx.core.bfloat16',True,valid)
  self.assertEqual(shape,[1,4,256,256]);self.assertEqual(len(out),524288)
  for h in range(4):self.assertEqual(out[h*256*row:(h+1)*256*row],heads[h]+bytes(192*row))
 def test_physical_tail_cannot_hide_write(self):
  raw=bytearray(4*64*512);raw[3*64*512+63*512+17]=1
  with self.assertRaises(AssertionError):canonical(raw,[1,4,64,256],'mlx.core.bfloat16',True,50)
  with self.assertRaises(AssertionError):canonical(bytes(4*64*512),[1,4,64,256],'mlx.core.bfloat16',True,65)
  data=b'\x80\x3f';self.assertEqual(canonical(data,[1],'mlx.core.float32',False,50),(data,[1]))
 def test_real_capacity_no_gpu_padding(self):
  s=(D/'fullqual.py').read_text();tree=ast.parse(s)
  self.assertIn('AttnCache.step=CAP;DecodeBatch.step=CAP',s)
  self.assertIn('cache.keys.shape[2]==CAP and cache.values.shape[2]==CAP',s)
  self.assertIn('same_weights_load',s);self.assertIn('total_steps==93',s)
  self.assertIn('archived44==archived',s);self.assertIn('v34_compare_output',s)
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='canonical_output')
  self.assertNotIn('mx.',ast.unparse(fn));self.assertNotIn('np.',ast.unparse(fn));self.assertNotIn('mx.pad',s)
  self.assertFalse(any(n=='mlx' or n.startswith('mlx.') for n in __import__('sys').modules))
if __name__=='__main__':unittest.main()
