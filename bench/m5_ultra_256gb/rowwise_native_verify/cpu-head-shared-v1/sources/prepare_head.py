"""Build a CPU-validated head-reuse plan and exact commands; no model/device launch."""
import argparse, importlib.abc, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; R=HERE.parents[2]
sys.path.insert(0,str(R))
from bench.m5_ultra_256gb.rowwise_native_verify import qualify as q
from bench.m5_ultra_256gb.exact_verifier_bridge import qualify as bridge


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    for name in ('out','child','audit-out'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if not a.run:
        q.need(not any((a.out,a.child,a.audit_out)),'explicit --run required');print('DESCRIPTION ONLY: CPU plan validation and exact command preparation.');return 0
    q.need(all((a.out,a.child,a.audit_out)),'all explicit output paths required')
    for path in (a.out,a.child,a.audit_out):q.need(path.is_absolute() and path.resolve()==path and not path.exists(),'fresh canonical output paths')
    q.need(len({a.out,a.child,a.audit_out})==3 and a.out.parent.is_dir(),'distinct outputs and existing preparation parent')
    old=R/'.artifacts/m5-ultra-256gb/long-decode-graph-v1-inputs/qualification-plan.json'
    plan=dict(schema='Rowwise-native-verifier-plan-v2',qualifier_sha256=q.sha(q.__file__),
        candidate=dict(path=str(HERE/'candidate.py'),sha256=q.CANDIDATE_SHA),bridge=dict(path=str(q.BRIDGE),sha256=q.BRIDGE_SHA),
        head_projection=dict(path=str(q.HEAD),sha256=q.HEAD_SHA),e024_plan=dict(path=str(old),sha256=bridge.OLD_PLAN_SHA),deadline_s=1800)
    _,_,_,_,_,_,inherited=q.validate(plan)
    a.out.mkdir()
    def write(name,value):
        with (a.out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    write('plan.json',plan);path=a.out/'plan.json';digest=q.sha(path);python=inherited['execution']['python']
    write('commands.json',dict(child=[python,'-I','-B',str(HERE/'qualify.py'),'--plan',str(path),'--plan-sha256',digest,'--out',str(a.child)],
        audit=[python,'-I','-B',str(HERE/'audit.py'),'--child',str(a.child),'--plan-sha256',digest,'--out',str(a.audit_out)],
        environment=inherited['execution']['environment'],scope='Root-owned cold/resource/lifecycle execution required; these commands have not been run.'))
    write('preparation.json',dict(status='CPU_VALIDATED',plan_sha256=digest,head_source=plan['head_projection'],
        source_pins={str(f):q.sha(f) for f in sorted(HERE.glob('*.py'))},model=inherited['model'],model_manifest=inherited['model_manifest'],
        source_manifest=inherited['source_manifest'],fixtures=inherited['fixtures'],runtime_pins=inherited['execution']['pins'],
        expected=dict(cases=9,prefixes=42,raw_files=96,output_tensor_pairs=5418,head_calls=9),device_started=False))
    print(json.dumps(dict(status='CPU_VALIDATED',plan=str(path),plan_sha256=digest)));return 0
if __name__=='__main__':raise SystemExit(main())
