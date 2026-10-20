"""Offline E050: E049 prefix checks plus actual batched-GDN composition binding.

The unchanged inherited plan describes the original replay contract, not the
actual new candidate. The outer source pins bind the private FunctionType globals,
engine-aware bridge and batched GDN dependency. Python branch counters include
tracing and do not establish GPU calls or a per-B invocation breakdown.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

B=Path(__file__).resolve().parents[1]
ORIGINAL=B/'c8_prefix_analysis_v1/audit.py'
ORIGINAL_SHA='936a21667c91b0f7e821628798b5398be973d075b581613ec1c729683999fe83'
BASE_PLAN_SHA='71f2891f24045f539a195711e8cdc97ea190bedd214e42173ab006b7eb700f1c'
NEW_SOURCES={
    'c8_batched_prefix_v1/prefix.py':'d16318dbad1181caac964ec5b331b4b208ca92c0ccd66fe87bbb01a54d4cd12e',
    'c8_batched_prefix_v1/bridge.py':'ee212bee9b0bf9e82d62c762b21dc630294e588538342b70df0f19fbf940bd07',
    'gdn_batched_graph/integration.py':'6e2e4d4350c36476292f2d0ff2bc7e095561f76a2c233b1a99575027378d50b7',
    'gdn_prework_batched/candidate.py':'acd2a23802296b9e420108203cc215c3170cd1eccd9a859e1a787432f98a5353',
}
COUNTER_SCOPE=('Per-layer Python candidate/fallback branches, including graph tracing; '
               'not GPU dispatches, completed kernels, or per-token device calls.')


def support():
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()!=ORIGINAL_SHA:raise ValueError('frozen E049 auditor changed')
    s=importlib.util.spec_from_file_location('_e050_prefix_audit',ORIGINAL)
    module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
    return module,module.support()


def composition(a,events,claimed):
    a.need(len(events)==2 and [r.get('event') for r in events]==['combined-installed','combined-restored'],
           'one exact combined install/restore, no hidden failure or extra lifetime')
    start,end=[{k:v for k,v in row.items() if k!='event'} for row in events]
    a.need(start['status']=='INSTALLED' and end['status']=='RESTORED','combined lifecycle states')
    a.need(start['counter_scope']==end['counter_scope']==COUNTER_SCOPE,'Python-only counter scope')
    a.need(start['graph_functions_before']==start['short_functions_before']==0
           and start['short_packed_before'] is False,'before-trace installation')
    for key in ('engine_id','runner_id','model_id','decoder_id','graph_handle_id'):
        a.need(type(start[key]) is int and start[key]>1 and end[key]==start[key], 'persistent actual object identity: '+key)
    indices=[i for i in range(64) if i%4!=3]
    a.same([r['layer'] for r in start['layers']],indices,'all48 original GDN layers in order')
    a.same([r['layer'] for r in end['layers']],indices,'all48 final GDN layers in order')
    totals={'candidate':0,'fallback':0}
    for key in ('layer_id','handle_id','wrapper_id'):
        ids=[r[key] for r in start['layers']]
        a.need(all(type(v) is int and v>1 for v in ids) and len(set(ids))==48,'distinct48 actual '+key)
    for initial,final in zip(start['layers'],end['layers']):
        a.same(initial['initial_counts'],dict(candidate=0,fallback=0),'fresh per-layer counters')
        a.same({k:v for k,v in final.items() if k!='final_counts'},initial,'same installed/restored layer identities')
        counts=final['final_counts'];a.need(set(counts)=={'candidate','fallback'},'exact branch counter fields')
        a.need(type(counts['candidate']) is int and counts['candidate']>0
               and type(counts['fallback']) is int and counts['fallback']==0,'candidate traced at every layer without fallback')
        for key in totals:totals[key]+=counts[key]
    extra={'handles_closed','native_decode_restored','caches_cleared','graph_functions_before_restore','short_functions_before_restore'}
    a.same({k:v for k,v in end.items() if k not in extra|{'status','layers'}},
           {k:v for k,v in start.items() if k not in {'status','layers'}},'unchanged lifetime metadata')
    a.need(end['handles_closed']==48 and end['native_decode_restored'] is end['caches_cleared'] is True,'all actual handles restored')
    # Frozen RouteObserver restores saved_functions (initially empty) before
    # Facade.restore; actual tracing lives in witness.functions and route events.
    a.need(type(end['graph_functions_before_restore']) is int and end['graph_functions_before_restore']==0
           and type(end['short_functions_before_restore']) is int and end['short_functions_before_restore']>=0,
           'original empty graph cache restored before combined release')
    a.same(claimed,dict(installs=1,restores=1,handle_count=48,records=[end],trace_counter_scope=COUNTER_SCOPE),
           'terminal composition exactly matches emitted lifecycle records')
    return dict(installs=1,restores=1,layers=48,python_branch_totals=totals,
        counts_by_layer=[dict(layer=r['layer'],**r['final_counts']) for r in end['layers']],
        trace_counter_scope=COUNTER_SCOPE,per_batch_counts_available=False,gpu_invocations_counted=False)


def audit(child,expected_sha):
    old,a=support();child=Path(child).resolve()
    result=dict(schema='E050-independent-prefix-audit-v1',status='INCOMPLETE',prefix_receipts_verified=False,
        completion_state='INCOMPLETE_BY_DESIGN',full_history_qualified=False,performance_qualified=False,
        owner_resources_restore_qualified=False,offline_tensor_bytes_rechecked=False,weight_shards_rehashed=False,
        child=str(child),plan_sha256=expected_sha,audit_source_sha256=a.sha(__file__),
        inherited_auditor=dict(path=str(ORIGINAL),sha256=ORIGINAL_SHA),
        scope='Actual batched GDN plus graph candidate against eager and recorded warm49 IDs. No full request drain, performance or Pareto grant.',
        provenance='inherited/plan.json remains the E049 replay contract. E050 private cloned run globals replace integration_module/validate/SCOPE; the bridge installs batched GDN before tracing. Outer pins and composition bind that actual candidate.',
        evidence_limit='Full successful tensor payloads were not retained. Finite/bit equality uses original source-bound per-array device receipts; no offline arithmetic or tensor-byte reread.')
    try:
        top=a.read(child/'complete.json');plan=a.read(child/'plan.json');inner=child/'inherited'
        a.need(a.sha(child/'plan.json')==expected_sha==top['plan_sha256'],'explicit actual E050 child plan')
        a.need(plan['schema']=='c8-batched-gdn-graph-prefix-plan-v1'
               and top['schema']=='c8-batched-gdn-graph-prefix-result-v1'
               and top['status']=='DIAGNOSTIC_PREFIX_COMPLETE' and top['source_input_pins_rechecked'] is True,'new actual source/terminal schema')
        for value in (top,plan):
            a.need(value['completion_state']=='INCOMPLETE_BY_DESIGN' and value['full_history_qualified'] is value['performance_qualified'] is False,'prefix-only scope')
        a.same(plan['switches'],dict(gdn_prework=True,route_graph=True),'both candidates explicitly ON')
        a.need(plan['stop_after_step']==49 and plan['base_prefix_plan']['sha256']==BASE_PLAN_SHA,'unchanged warm49 replay contract')
        for path,digest in {**plan['sources'],**plan['predecessor_inputs']}.items():a.pinned(dict(path=path,sha256=digest))
        for name in ('base_prefix_plan','predecessor','qualification','recording'):a.pinned(plan[name])
        for name,digest in NEW_SOURCES.items():a.need(plan['sources'][str(B/name)]==digest,'actual new source chain: '+name)
        inherited_plan=a.read(inner/'plan.json');a.same(inherited_plan,a.read(plan['base_prefix_plan']['path']),'unmodified actual inherited plan')
        for name in ('qualification','recording','recipe','environment','stop_after_step'):
            a.same(plan[name],inherited_plan[name],'new/old replay contract: '+name)
        inherited_top=a.read(inner/'complete.json')
        a.same(top['inherited_result'],dict(path=str(inner/'complete.json'),sha256=a.sha(inner/'complete.json'),
            returncode=0,status='DIAGNOSTIC_PREFIX_COMPLETE'),'actual inherited result binding')
        for key in ('pid','ppid','base_result','prefix','coverage','route_prefix','peak_mlx_bytes','elapsed_base_seconds'):
            a.same(top[key],inherited_top[key],'outer/inherited actual evidence: '+key)
        a.same(top['prefix'],a.read(plan['predecessor']['path'])['prefix'],'same49 prefix IDs and unfinished boundary as actual E049')
        event_path=a.pinned(top['composition_events']);a.need(event_path==child/'composition.jsonl','actual local composition sidecar')
        stable={str(p):a.sha(p) for p in (child/'plan.json',child/'complete.json',event_path)}
        with event_path.open('rb') as stream:
            events=[]
            for raw in stream:
                a.need(raw.endswith(b'\n'),'truncated composition event');events.append(a.parse(raw))
        result['composition']=composition(a,events,top['composition'])
        result['inherited_prefix_audit']=old.audit(inner,BASE_PLAN_SHA)
        a.need(result['inherited_prefix_audit']['status']=='PASS'
               and result['inherited_prefix_audit']['prefix_receipts_verified'] is True,'unchanged full49 independent checks failed')
        for path,digest in {**stable,**plan['sources'],**plan['predecessor_inputs']}.items():a.need(a.sha(path)==digest,'source/record changed during audit')
        result.update(status='PASS',prefix_receipts_verified=True,terminal_sha256=stable,
            actual_candidate_sources={str(B/k):v for k,v in NEW_SOURCES.items()},
            peak_mlx_bytes_receipt=top['peak_mlx_bytes'],mlx_peak_is_physical_footprint=False)
    except FileNotFoundError as exc:result.update(status='INCOMPLETE',reason=str(exc))
    except Exception as exc:result.update(status='FAIL',first_failure=dict(type=type(exc).__name__,message=str(exc)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child',required=True,type=Path)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',required=True,type=Path);args=p.parse_args(argv)
    if args.out.exists() or not args.out.parent.is_dir():raise ValueError('new output in existing directory required')
    result=audit(args.child,args.plan_sha256)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','prefix_receipts_verified','completion_state')}))
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
