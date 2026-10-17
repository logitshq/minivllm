"""V126 CPU source/entry checks only; no model, fake inference or network."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
D=Path(__file__).resolve().parent;P=D.parent;OLD=P/'m4-final-native15-official-attempt2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return ast.parse(p.read_text())
def fn(t,name):return next(n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def main():
 for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
 assert (D/'c1_server.py').read_bytes()==(P/'v125-fresh-c1-native-official-build1/server.py').read_bytes()
 spec=importlib.util.spec_from_file_location('v126_policy',D/'entry_policy.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
 assert policy.choose()=='direct_c1' and policy.choose(1,enable_full_curve=True)=='direct_c1'
 for c in (2,4):
  assert policy.choose(c,enable_full_curve=True)=='legacy_batched'
  try:policy.choose(c)
  except ValueError:pass
  else:raise AssertionError('defaultOFF missing')
 for c,on in ((True,True),(3,True),(0,True),(8,True),(1,1)):
  try:policy.choose(c,enable_full_curve=on)
  except (ValueError,TypeError):pass
  else:raise AssertionError('invalid startup profile accepted')
 old,new=tree(OLD/'child.py'),tree(D/'batched_server.py')
 assert dump(fn(old,'Handler'))==dump(fn(new,'Handler'))
 assert dump(fn(new,'do_POST'))==dump(fn(tree(D/'c1_server.py'),'do_POST'))
 qual=tree(D/'qualify_batch.py');client=tree(P/'v125-fresh-c1-native-official-build1/driver.py')
 for name in ('request','group'):assert dump(fn(qual,name))==dump(fn(client,name))
 class Inverse(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='tutorial_code.miniengine.cache' and [a.name for a in n.names]==['AttnCache','DecodeBatch']:return None
   return n
  def visit_Expr(self,n):
   if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='runtime_identity.update' and any(k.arg=='static_profile' for k in n.value.keywords):return None
   return self.generic_visit(n)
  def visit_Assert(self,n):
   s=ast.unparse(n.test)
   if s.startswith(('(AttnCache.step, DecodeBatch.step) ==','e.scheduler.max_seqs == 4','route[\'decode_batch_counts\'].get(str(a.client_batch), 0) > 0','not live and (not e.requests) and (not e.runner.staging)','e.runner.batch.size == 0 and e.runner._ahead is None')):return None
   return self.generic_visit(n)
 assert dump(Inverse().visit(copy.deepcopy(fn(new,'work'))))==dump(fn(old,'work')),'old actual work core changed'
 profiles=json.loads((D/'profiles.json').read_text())
 assert profiles['direct_c1']['max_seqs']==1 and profiles['legacy_batched']['max_seqs']==4
 assert profiles['legacy_batched']['native_adapter'] is profiles['legacy_batched']['native_library'] is None
 assert profiles['legacy_batched']['physical_capacity']==256
 assert profiles['direct_c1']['physical_capacity']==64
 assert sha(D/'batched_server.py')==profiles['legacy_batched']['entry_sha256']
 plan=json.loads((D/'campaign-plan.json').read_text());assert len(plan['configs'])==21
 assert sum(6*x['batch'] for x in plan['configs'])==270
 assert all(x.get('static_native_profile')==('direct_c1' if x['batch']==1 else 'legacy_batched') for x in plan['configs'] if x['engine']=='native')
 assert all(x['batch']==1 for x in plan['configs'] if x['engine']=='lithos')
 evidence=json.loads((D/'historical-batch-evidence.json').read_text());assert len(evidence['arms'])==6
 assert all(x['requested_batch_GDN_calls']>0 and x['warm_hooks_restored'] and x['own_raw32_visible32_usage_exact'] for x in evidence['arms'])
 assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
 return dict(status='PASS_CPU_V126_STATIC_ENTRY_SOURCE_ONLY',policy_cases=11,c1_server_byte_exact125=True,
  accepted25_batched_work_AST_inverse_exact_native15=True,Handler_AST_exact_native15=True,POST_AST_exact125=True,
  actual_historical_C2_C4_arms=6,new_entry_GPU_qualified=False,fullcurve_driver_prepared=False,
  defaultOFF=True,dynamic_live_migration=False,GPU_used=False,MLX_imported=False)
if __name__=='__main__':print(json.dumps(main(),indent=2))
