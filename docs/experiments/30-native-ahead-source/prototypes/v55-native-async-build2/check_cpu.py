"""CPU-only V55 native scheduling/source tests. Never loads dylib/MLX/Metal."""
from pathlib import Path
import ast,hashlib,subprocess,sys,unittest
D=Path(__file__).resolve().parent
P=D.parent
class Contract(unittest.TestCase):
 def test_actual_portable_core_fifo_bank_reset_consume_discard(self):
  r=subprocess.run(['clang++','-std=c++17','-Wall','-Wextra','-Werror',str(D/'test_pipeline_core.cpp'),'-o',str(D/'test_pipeline_core')],capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr)
  r=subprocess.run([str(D/'test_pipeline_core')],capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr);self.assertIn('PASS_FIFO_DEPTH2_BANKS_SLOTS_CONSUME_RESET_WRITE_FAILED_DISCARD',r.stdout)
 def test_frozen_parent_exact(self):
  expected={'replay.mm':'25d9ff0d1478d9737c18913757845297f2b588c13bbd58a3d96869184282852b','libv40replay.dylib':'8ef20b52e80cd0798e80abf2037c98211f13437dbf5ea4ddb47dce1b00ff471b'}
  for n,d in expected.items():self.assertEqual(hashlib.sha256((P/'v40-native-icb-build2'/n).read_bytes()).hexdigest(),d)
 def test_captured_graph_math_icb_and_sync_off_paths_retained(self):
  a=(P/'v40-native-icb-build2/replay.mm').read_text();b=(D/'replay.mm').read_text()
  for start,end in [('static void encodeICB(', 'static void restoreArena('),('extern "C" int v34_seal_feedback(', 'extern "C" int v34_reset('),('extern "C" int v36_register_pingpong(', 'extern "C" const char*v36_timing_json(')]:
   self.assertIn(a[a.index(start):a.index(end)],b)
  self.assertIn(a[a.index('extern "C" int v38_enable_icb('):],b)
  start='extern "C" int v34_replay_step()';end='extern "C" int v34_copy_output('
  off=b[b.index(start):b.index(end)].replace('\n if(pipelineEnabled()){fail("legacy synchronous replay forbidden while async pipeline enabled");return 1;}','')
  self.assertEqual(off,a[a.index(start):a.index(end)])
 def test_per_slot_fences_retained_cb_and_output_contract(self):
  b=(D/'replay.mm').read_text()
  self.assertIn('id<MTLCommandBuffer>b=[[S().queue commandBuffer] retain]',b)
  self.assertIn('[e waitForFence:a.slots[a.previousSlot].feedbackDone]',b)
  self.assertIn('[capture waitForFence:slot.graphDone]',b);self.assertIn('[feedback waitForFence:slot.stageDone]',b)
  self.assertIn('[feedback updateFence:slot.feedbackDone]',b)
  self.assertLess(b.index('[job.cb waitUntilCompleted]'),b.index('[job.cb release]'))
  self.assertIn('c.last.outputBank?p->b:p->a',b);self.assertIn('A55().slots[c.last.slot].stage.contents',b)
  self.assertIn('!A55().core.canWriteSeed()||F().written.count(addr)',b)
  start=b.index('extern "C" int v34_write_input(');end=b.index('extern "C" int v34_replay_step(',start);write=b[start:end]
  self.assertLess(write.index('std::lock_guard<std::recursive_mutex>lock(S().lock)'),write.index('!A55().core.canWriteSeed()||F().written.count(addr)'))
  self.assertLess(write.index('!A55().core.canWriteSeed()||F().written.count(addr)'),write.index('if(!F().ready'))
  self.assertIn('empty drain retains last delivered',b)
  self.assertIn('if(!a.core.requestSubmitted){int32_t token=',b)
 def test_tiny_pipeline_schedule_and_inert_boundary(self):
  s=(P/'v55-native-async-smoke-build2/smoke.py').read_text();ast.parse(s)
  self.assertLess(s.index("if sys.argv[1:]==['--inert']:"),s.index('import mlx.core as mx'))
  self.assertIn('submit_one();submit_one()',s)
  self.assertLess(s.index("call('v55_wait_step',step)"),s.index("call('v34_compare_output',original.address"))
  self.assertLess(s.index("call('v34_compare_output',original.address"),s.index('if submitted<5:submit_one()'))
  self.assertNotIn("call('v34_replay_step')",s)
  self.assertFalse(any(n=='mlx'or n.startswith('mlx.')for n in sys.modules))
if __name__=='__main__':unittest.main()
