"""Offline three-mode raw/count/closure/resource binding; never a performance point."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bench.m5_ultra_256gb.c1_profile_cycle_v1 import _reuse as reuse
_original=reuse.install(globals(),'gdn_graph_combined_formal/audit_formal.py')
PHASES=('off','runner','full')
OWNER_SHA=reuse.sha(HERE/'owner.py');CONTROLLER_SHA=reuse.sha(HERE/'controller.py')
OWNER_BINDING_SOURCE=reuse.variant(globals(),_original.owner_binding,[("plan['schema']=='GDN-graph-combined-http-formal-owner-plan-v1'","plan['schema']=='C1-profile-owner-plan-v1'")])
_r=reuse.fixed('lithos_d512_formal/audit_resources.py');_rn=dict(vars(_r));_rn['__file__']=__file__


def drained_absence(exact,pid,sub):
    raw=exact.get('raw') or {}
    need(sub=='draining' and exact['status']=='unknown' and exact['pid']==pid
         and raw.get('command')==['/bin/ps','-p',str(pid),'-o','pid=,ppid=,rss=,comm=']
         and raw.get('returncode')==1 and raw.get('stdout')==raw.get('stderr')=='','only strict natural-drain PID absence')


# Natural exit can race the post-200 drain checkpoint. Preserve its raw absent PID.
_rn['drained_absence']=drained_absence
RESOURCE_SOURCE=reuse.variant(_rn,_r.resources,[
    ("'formal','after-formal'","'diagnostic','after-diagnostic','draining'"),
    ("('warm','formal')","('warm','diagnostic','draining')"),
    ("else:need(False,'live formal server identity disappeared before reap')","else:drained_absence(exact,expected_pid,sub)")])
resources=_rn['resources']


def audit(journal,plan_sha256):
    result=dict(schema='C1-profile-closed-audit-v1',status='INCOMPLETE',diagnostic_complete=False,numerical_qualified=False,performance_qualified=False,modes=[])
    try:
        journal=Path(journal).resolve();restoration=reuse.local('audit_restore.py').audit(journal)
        need(restoration['status']=='PASS' and restoration['qualified'] is True,'independent restoration required')
        data=read(journal/'journal.json');need(data['owner_plan']['sha256']==plan_sha256,'actual owner plan digest')
        root=Path(data['owner_directory']);module=reuse.local('owner.py');args=SimpleNamespace(plan=Path(data['owner_plan']['path']),plan_sha256=plan_sha256,out=root)
        _,plan,_,numeric,specs=module.validate_plan(args);complete=read(root/'complete.json')
        need(complete['schema']==module.RESULT_SCHEMA and complete['status']=='PASS' and complete['errors']==[] and complete['diagnostic_modes_complete'] is True,'all three completed diagnostic modes')
        attempt,stopped,lives,proof=owner_binding(root,plan,complete)
        result['preparation']=preparation_pins(root,plan,numeric)
        same(read(root/'diagnostic-specs.json'),specs,'executed specs')
        same(read(root/'diagnostics.json'),dict(schema='C1-profile-modes-v1',trials=complete['phases'],performance_qualified=False),'published diagnostic refs')
        capture=reuse.fixed('c1_pipeline_profile/capture.py')
        for mode,spec,life,item in zip(PHASES,specs,lives,complete['phases']):
            trial=root/'phases'/mode/'trial';closed=read(trial/'complete.json');inner=read(trial/'release.json');outer=read(trial.parent/'release.json')
            module.accept_phase(closed,mode,inner,outer);same(read(trial/'spec.json'),spec,'actual capture spec')
            same(item,dict(phase=mode,status='PASS',result=dict(path=str((trial/'complete.json').relative_to(root)),sha256=sha(trial/'complete.json'))),'actual phase result')
            same(read(trial/'launch.json'),dict(command=spec['command'],environment=spec['environment']),'actual diagnostic launch')
            for name,ref in closed['artifacts'].items():
                path=(trial/name).resolve();need(path.is_relative_to(trial.resolve()) and ref==dict(sha256=sha(path),bytes=path.stat().st_size),'capture artifact binding')
            cap=capture.native().load('http_capture_v2')
            need(cap.ready_identity(read(trial/'ready.json'),spec['identity']) and cap.owned_listener(read(trial/'listener.json'),life['pid'],spec['transport']['port']),'actual ready identity/listener')
            need(set(closed['artifacts'])=={str(p.relative_to(trial)) for p in trial.rglob('*') if p.is_file() and p!=trial/'complete.json'},'complete capture artifact inventory')
            decoder=capture.System().decoder(spec);groups={stage:read(trial/(stage+'-group.json')) for stage in capture.STAGES};metrics={}
            for stage,group in groups.items():
                checked=capture.check_group(group,spec,stage,decoder);same(checked,read(trial/(stage+'-raw-check.json')),'independent raw count check');metrics[stage]=checked['diagnostic_metrics']
            same(capture.closed_receipts(trial,spec,life['pid'],groups),closed['closed_server'],'all16 actual DONE/profile/closure join')
            result['modes'].append(dict(mode=mode,pid=life['pid'],requests=16,metrics=metrics,natural_exit=True))
        result['resources']=resources(root,complete,lives,data['original']['pid'],attempt['created_at'],list(PHASES))
        result.update(status='PASS',diagnostic_complete=True,restoration=restoration,journal_sha256=sha(journal/'journal.json'),scope='48 actual diagnostic requests; shared control and observer overhead; overlapping spans are not exclusive GPU time.')
    except FileNotFoundError as exc:result['error']=str(exc)
    except Exception as exc:result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--journal',type=Path);p.add_argument('--plan-sha256');p.add_argument('--out',type=Path);a=p.parse_args(argv)
    if not any((a.journal,a.plan_sha256,a.out)):print('DESCRIPTION_ONLY: offline 48-request diagnostic audit.');return 0
    need(all((a.journal,a.plan_sha256,a.out)) and not a.out.exists(),'explicit fresh audit output');result=audit(a.journal,a.plan_sha256)
    with a.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]
if __name__=='__main__':raise SystemExit(main())
