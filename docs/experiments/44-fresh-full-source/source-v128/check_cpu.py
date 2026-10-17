"""CPU source/metrics/actual-binding checks only. No inference imports/server."""
import ast,hashlib,json,statistics,sys
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tree(p):return ast.parse(Path(p).read_text())
def fn(t,name):return next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name==name)
def dump(x):return ast.dump(x,include_attributes=False)
def main():
 for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
 a,b=tree(D/'driver.py'),tree(P/'v125-fresh-c1-native-official-build1/driver.py')
 for name in ('request','group','full_request_latency_p50_ms','assert_pipeline'):
  assert dump(fn(a,name))==dump(fn(b,name)),name
 ns={'statistics':statistics,'json':json,'Path':Path}
 for name in ('full_request_latency_p50_ms','auxiliary_median','signed_pairs'):
  exec(compile(ast.Module(body=[fn(a,name)],type_ignores=[]),'actual_pure_statistics','exec'),ns)
 group=lambda ds:{'requests':[dict(start_perf_counter_s=10.,done_perf_counter_s=10.+v) for v in ds]}
 assert ns['full_request_latency_p50_ms']([group([1.,100.]),group([2.]),group([3.])])==3000.
 assert ns['auxiliary_median']([1.,None,3.]) is None and ns['auxiliary_median']([1.,2.,3.])==2.
 plan=json.loads((D/'campaign-plan.json').read_text())['configs'];assert len(plan)==21 and len({p['label'] for p in plan})==21
 assert sum(6*p['batch'] for p in plan)==270 and sum(3*p['batch'] for p in plan)==135
 keys={(p['engine'],p['batch']) for p in plan};assert len(keys)==7
 assert all(sum((p['engine'],p['batch'])==k for p in plan)==3 for k in keys)
 assert not any(p['batch']>1 and p['engine']=='lithos' for p in plan)
 rows=[dict(label=p['label'],engine=p['engine'],concurrency=p['batch'],latency_p50_ms=2. if p['engine']=='native' else 1.,y=1. if p['engine']=='native' else 2.) for p in plan]
 pairs=ns['signed_pairs'](rows,plan);assert len(pairs)==12
 assert all(x['latency_reduction_percent']==-100. and x['throughput_gain_percent']==-50. and x['negative_values_retained'] for x in pairs)
 s=(D/'driver.py').read_text();assert s.count('time.sleep(60)')==1
 assert 'len(completed) < len(plan)' in s and 'completed.append(run_arm(' in s
 assert 'if batch == 1: env[\'DYLD_INSERT_LIBRARIES\']' in s
 assert "elif phase == 'warm' and engine == 'native':" in s
 assert 'formal_steps' in s and "max(q['active_decode_batch'] for q in formal_steps) == batch" in s
 assert "if engine == 'native' and batch == 1:" in s
 c=json.loads((D/'runtime-config.json').read_text());q=c['curve_qualification'];r=json.loads(Path(q['release']).read_text())
 assert sha(q['receipt'])==q['receipt_sha256'] and sha(q['release'])==q['release_sha256']
 assert r['status']=='PASS_EXTERNAL_RELEASE_V126_C2_C4_ENTRY' and r['GPU_released'] and r['tool_exit_code']==0
 assert sha(q['independent_review'])==q['independent_review_sha256']
 audit=json.loads(Path(q['independent_review']).read_text())
 assert audit['status']=='PASS_INDEPENDENT_V126_ACTUAL_ENTRY_RESULTS'
 assert audit['actual_receipt_sha256']==q['receipt_sha256'] and audit['actual_external_release_sha256']==q['release_sha256']
 assert audit['source_pins_sha256']==q['source_pins_sha256']
 assert c['dynamic_live_migration'] is False and c['old_results_pooled'] is False
 assert len(c['legacy_package_sources'])==25
 owner=(D/'owner.py').read_text();assert 'proc.wait(timeout=3600)' in owner and 'proc.wait(timeout=90)' in owner and 'proc.wait(timeout=20)' in owner
 protocol=json.loads((D/'protocol.json').read_text());assert protocol['cooling']=={'seconds':60,'gaps':20,'total_seconds':1200,'placement':'after successful owned unload/reap/refusedport/postresourceguard,before nextarm','adaptive':False}
 assert not(D/'attempt.consumed.json').exists()
 assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
 return dict(status='PASS_CPU_V128_FULLCURVE_SOURCE_AND_STATS',HTTP_request_group_clock_AST_exact125=True,
  latency_and_pipeline_AST_exact125=True,configs=7,fresh_loads=21,HTTP_requests=270,formal_requests=135,
  signed_pairs=12,negative_pair_cases_retained=True,auxiliary_missing_preserved=True,
  C1_and_C2_C4_route_assertions_separate=True,actual126_and_release_and_independentreview_bound=True,
  fixed_unloaded_gaps=20,fixed_cool_seconds=1200,owner_deadline_s=3600,GPU_used=False,MLX_imported=False)
if __name__=='__main__':print(json.dumps(main(),indent=2))
