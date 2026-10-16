"""Fault rejection for the target-machine migration (CPU only)."""
import ast, copy, importlib.util, json, os, tempfile, types, unittest
from pathlib import Path
import driver, resource_base, resources
D=Path(__file__).resolve().parent
class Migration(unittest.TestCase):
 def test_desktop_ac_without_battery(self):
  old=resources._run
  try:
   resources._run=lambda a:{'returncode':0,'stdout':"Now drawing from 'AC Power'",'stderr':''}
   got=resources._battery();self.assertEqual(got['status'],'known');self.assertEqual(got['source'],'AC Power');self.assertIsNone(got['percent'])
   resources._run=lambda a:{'returncode':0,'stdout':'nonsense','stderr':''}
   self.assertEqual(resources._battery()['status'],'unknown')
  finally:resources._run=old
 def test_pressure_and_sustained_swap_stop(self):
  good={'pressure_level_value':1,'free_percent':90,'swap_used_mb':0,'process_inventory_ok':True,'vm_counters':{'Swapouts':0,'Pageouts':0},'only_owned_known_inference_processes':True}
  resource_base.guard([good])
  for key,value in [('pressure_level_value',2),('process_inventory_ok',False),('only_owned_known_inference_processes',False)]:
   with self.assertRaises(RuntimeError):resource_base.guard([dict(good,**{key:value})])
  with self.assertRaises(RuntimeError):resource_base.guard([good,dict(good,swap_used_mb=65),dict(good,swap_used_mb=130)])
 def test_single_attempt_cannot_change_output(self):
  original=(driver.P,driver.verify_frozen,driver.execute,driver.Resources)
  f=json.loads(driver.FROZEN.read_text());go=dict(f['GO_schema'],frozen_manifest_sha256='a'*64,expected_power='AC',authorizing_owner_pid=os.getppid())
  try:
   with tempfile.TemporaryDirectory() as t:
    driver.P=Path(t);driver.verify_frozen=lambda *a:f
    driver.execute=lambda *a:(_ for _ in ()).throw(ValueError('CPU sentinel'))
    driver.Resources=lambda *a:types.SimpleNamespace(sample=lambda *a:None,raws=[])
    path=Path(t)/'go.json';out=Path(t)/'output';go['output_dir']=str(out);path.write_text(json.dumps(go));argv=['--go',str(path),'--frozen-sha256','a'*64,'--output-dir',str(out)]
    self.assertEqual(driver.main(argv),1)
    out2=Path(t)/'other';go['output_dir']=str(out2);path.write_text(json.dumps(go));argv[-1]=str(out2)
    with self.assertRaises(FileExistsError):driver.main(argv)
    self.assertFalse(out2.exists())
  finally:driver.P,driver.verify_frozen,driver.execute,driver.Resources=original
 def test_core_qualification_and_fallback_are_unchanged(self):
  ref=D.parents[1]/'.artifacts/m5-ultra-256gb/source/reference/v19-fullmodel-qual-build2'
  self.assertEqual((D/'fallback.py').read_bytes(),(ref/'fallback.py').read_bytes())
  new=(D/'worker.py').read_text();old=(ref/'worker.py').read_text()
  for start,end in [("  w=json.loads", "  for name,module in list(sys.modules.items()):"),('def install_hooks','def run')]:
   self.assertEqual(new[new.index(start):new.index(end)],old[old.index(start):old.index(end)])
if __name__=='__main__':unittest.main()
