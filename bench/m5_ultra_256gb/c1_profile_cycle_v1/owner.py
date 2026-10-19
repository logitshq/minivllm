"""Three diagnostic servers, unchanged direct-Popen run/reap algorithm. Default describe."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bench.m5_ultra_256gb.c1_profile_cycle_v1 import _reuse as reuse
_original=reuse.install(globals(),'gdn_graph_combined_formal/owner.py')
PHASES=('off','runner','full')
PLAN_SCHEMA='C1-profile-owner-plan-v1'
JOURNAL_SCHEMA='C1-profile-cold-journal-v1'
RESULT_SCHEMA='C1-profile-owner-result-v1'
SPEC=HERE.parents[2]/'.artifacts/m5-ultra-256gb/gdn-graph-combined-formal-v2-owner/phases/r1-c1-on/trial/spec.json'
SPEC_SHA='d0b993baf8374ae1e6e1219dfdc624009bbc426f356fe7c613a70c03b1a220b0'
E033=HERE.parents[2]/'docs/experiments/m5-ultra-256gb/E033-gdn-graph-combined-formal-v2/summary.json'
E033_SHA='84d061af6267b82376c4c3683116b82f456d63f2ffd1804bf11b0ff65be30275'


def load(name):
    return reuse.fixed('c1_pipeline_profile/capture.py') if name=='formal' else _original.load(name)
def source_pins():
    pins={**_original.source_pins(),**reuse.pins(),str(SPEC):SPEC_SHA,str(E033):E033_SHA}
    pins.update({str(p):sha(p) for p in (HERE.parent/'c1_pipeline_profile').glob('*.py')})
    need(all(sha(p)==h for p,h in pins.items()),'fixed inputs/source closure changed')
    return pins
def trial_names(value):
    need(type(value) is list and value==list(PHASES),'exact off/runner/full order');return value
def work_seconds(trials,deadlines):
    trial_names(trials);return PREPARE_SECONDS+len(trials)*(sum(deadlines[k] for k in ('overall','close','term','kill'))+12)
def validate_plan(args):
    base=load('qualify_absent_owner');raw=base.absolute(args.plan).read_bytes();need(base.sha_bytes(raw)==args.plan_sha256,'actual plan SHA')
    plan=json.loads(raw);need(set(plan)=={'schema','execution','qualification','formal_spec','prepared_specs','cold_restart','http','trials','deadline_seconds','outputs'} and plan['schema']==PLAN_SCHEMA,'exact diagnostic plan')
    trial_names(plan['trials']);need(plan['execution']==dict(owner=dict(path=str(HERE/'owner.py'),sha256=sha(__file__)),pins=source_pins()),'actual source pins')
    need(plan['formal_spec']==dict(path=str(SPEC),sha256=SPEC_SHA),'actual E033 combined-ON C1 input')
    formal=json.loads(SPEC.read_bytes());load('formal').factory().validate(formal)
    need(json.loads(E033.read_bytes())['performance_qualified'] is True and formal['arm']=='on' and formal['concurrency']==1,'closed E033 C1ON evidence')
    q=plan['qualification'];need(q==formal['base_plan'] and q['sha256']==NUMERICAL_SHA and sha(q['path'])==NUMERICAL_SHA,'original native preparation plan')
    cold=plan['cold_restart'];need(set(cold)=={'journal_path','controller','predecessor','approved_contract','proposal_plan_sha256'},'exact cold binding')
    base.absolute(cold['journal_path']);need(cold['controller']==dict(path=str(HERE/'controller.py'),sha256=sha(HERE/'controller.py')) and cold['approved_contract']==base.CONTRACT and cold['proposal_plan_sha256']==base.PROPOSAL_SHA,'actual controller/authorization')
    prior_replacement(plan);need(set(plan['outputs'])=={'owner'} and Path(plan['outputs']['owner']).is_absolute(),'actual owner output')
    if hasattr(args,'out'):need(str(args.out)==plan['outputs']['owner'],'registered owner output')
    need(plan['http']=={k:formal[k] for k in ('transport','deadlines','environment')},'original HTTP policy')
    need(type(plan['deadline_seconds']) is int and plan['deadline_seconds']==work_seconds(plan['trials'],plan['http']['deadlines']),'bounded actual three-mode work')
    need(set(plan['prepared_specs'])==set(PHASES),'three real prepared specs');specs=[]
    for mode in PHASES:
        ref=plan['prepared_specs'][mode];need(set(ref)=={'path','sha256'} and sha(base.absolute(ref['path']))==ref['sha256'],'prepared spec SHA')
        spec=json.loads(Path(ref['path']).read_bytes());load('formal').validate(spec)
        need(spec['mode']==mode and spec['formal_spec']==plan['formal_spec'] and spec['output_directory']==str(Path(plan['outputs']['owner'])/'phases'/mode/'trial'),'actual mode/output/input')
        need(all(plan['execution']['pins'].get(p)==h for p,h in spec['sources'].items()),'capture dependency closure')
        specs.append(spec)
    need(len({s['drain_token'] for s in specs})==3,'distinct mode drain nonces')
    return base,plan,raw,json.loads(Path(q['path']).read_bytes()),specs


def accept_phase(result,phase,inner,outer):
    reuse.fixed('c1_pipeline_profile/owner.py').accepted(result,phase)
    need(inner['natural_exit'] is True and inner['forced_stop'] is False and inner['returncode']==outer['returncode']==0,'natural inner/outer exit0')


# Keep trial_one, spawn, probes, deadlines, cancellation, release and finally byte-for-byte.
_start="            need(result['schema']=='GDN-graph-combined-http-formal-result-v1'"
_end="        need([row['phase'] for row in children] == list(phases)"
_body=reuse.inspect.getsource(_original.run);need(_body.count(_start)==_body.count(_end)==1,'fixed acceptance region')
_old=_body[_body.index(_start):_body.index(_end)]
_new="""            accept_phase(result,phase,inner_release,outer_release)
            check(); probe(phase + ':released')
            stages.append(dict(phase=phase,status='PASS',result=ref(folder/'trial/complete.json')))
            base.durable(args.out/'diagnostics.json',dict(schema='C1-profile-modes-v1',trials=stages,performance_qualified=False))
"""
ADAPTED_RUN=reuse.variant(globals(),_original.run,[(_old,_new),
    ("'formal-specs.json'","'diagnostic-specs.json'"),
    ("schema='GDN-graph-combined-http-formal-owner-result-v1'","schema=RESULT_SCHEMA"),
    ('formal_trials_complete=','diagnostic_modes_complete='),
    ("scope='Observer-OFF formal trials with captured-history numerical/count bindings; independent resource/restoration audit required.'",
     "scope='Three diagnostic captures only; independent raw/resource/restoration audit required; no Pareto point.'")])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true');mode.add_argument('--prepare-only',action='store_true')
    p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256');p.add_argument('--out',type=Path);a=p.parse_args(argv)
    if not (a.run or a.prepare_only):
        need(not any((a.plan,a.plan_sha256,a.out)),'explicit mode required');print('DESCRIPTION_ONLY: three direct diagnostic servers; actual sealed predecessor required.');return 0
    need(all((a.plan,a.plan_sha256,a.out)),'all explicit owner paths required');a.plan,a.out=a.plan.resolve(),a.out.resolve()
    need(a.out.parent.is_dir() and not a.out.exists(),'fresh output required')
    if a.prepare_only:
        # Preparation's output is a separate receipt folder, never the registered owner folder.
        base,plan,_,_,specs=validate_plan(SimpleNamespace(plan=a.plan,plan_sha256=a.plan_sha256))
        a.out.mkdir();base.durable(a.out/'prepared.json',dict(schema='C1-profile-owner-prepare-v1',status='PREPARED',owner_plan_sha256=a.plan_sha256,
            specs=specs,sources=plan['execution']['pins'],runtime_weights_verified=False),exclusive=True);return 0
    return run(a)


if __name__=='__main__':raise SystemExit(main())
