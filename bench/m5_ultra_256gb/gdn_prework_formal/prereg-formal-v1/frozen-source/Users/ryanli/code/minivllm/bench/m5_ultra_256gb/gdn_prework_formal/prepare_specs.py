"""CPU-only all-12 actual formal.prepare/validate compatibility check.

Default describes; --run requires actual GDN HTTP independent PASS and a
source-matching complete focused CPU receipt. No model/runtime payload hashing,
subprocess, service, network, server, or owner/controller preparation is allowed.
Every attempted spec and failure is retained in a new output directory. This is
not execution registration or performance qualification.
"""
import argparse
import contextlib
import datetime
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path('/Users/ryanli/code/minivllm')
ART=ROOT/'.artifacts/m5-ultra-256gb'
BENCH=ROOT/'bench/m5_ultra_256gb'
FORMAL=BENCH/'gdn_prework_formal/formal.py'
TEST=FORMAL.with_name('test_formal_cpu.py')
FORMAL_SHA='e3fd332884975426487853108098933283e4b98705671cfe8dbccd37f91bfb7a'
TEST_SHA='8cc2d1f7a20fd8e621ae2c21841f9298dba779d94c1013719a6c194d4c81f838'
BASE_REG=BENCH/'pareto_realistic/prereg-native-formal-v1'
BASE_PLAN_SHA='7b7a563ee832c1f91c2f5c745ec0b3087ef25dae6d8974cbfa6e0cb9a23687bb'
BASE_PREPARED_SHA='eac787529e3d704b0f7a777f9fae54a5562233fbc5f4e3500bf83da5d06eee13'
ROUTE_REG=BENCH/'gdn_prework_http_cycle/prereg-http-v1'
OWNER=ART/'gdn-http-v1-owner'
JOURNAL=ART/'gdn-http-v1-cold-journal/journal.json'
EXECUTION=ART/'gdn-http-v1-execution/final.json'
ANALYSIS=ART/'gdn-http-v1-analysis'
DEFAULT_OUT=ART/'gdn-http-formal-prepare-v1'
ARCHIVE=ROOT/'docs/experiments/m5-ultra-256gb/E029-gdn-prework-http-v1'
CYCLE_PLAN_SHA='5d7bf34bec68dca784ebc449e660d14bcc63a6ecd2ffd7cae63981c22ebe3024'
FORBIDDEN={'mlx','omlx','mlx_lm','mlx_vlm','torch','numpy','tutorial_code','monolith'}
SUITE='bench.m5_ultra_256gb.gdn_prework_formal.test_formal_cpu'


def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1<<20):digest.update(block)
    return digest.hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def ref(path):return dict(path=str(path.resolve()),sha256=sha(path))
def same(a,b,msg):need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),msg)


def closed_gate():
    execution,journal,summary=[read(p) for p in (EXECUTION,JOURNAL,ANALYSIS/'execution-summary.json')]
    need(execution['journal_state']==journal['state']=='RESTORED'
         and type(execution['recovery_returncode']) is int and execution['recovery_returncode']==0
         and type(execution['owner_returncode']) is int and execution['owner_returncode']==0,
         'actual successful owner and restored driver required')
    need(summary['schema']=='GDN-http-closed-audits-v1' and summary['status']=='PASS'
         and summary['restoration_qualified'] is summary['numerical_qualified'] is summary['http_history_qualified'] is True
         and summary['performance_qualified'] is False and summary['sources_unchanged'] is True
         and summary['inputs_unchanged'] is True and summary['ended_at'] and 'error' not in summary,
         'complete actual independent audit execution')
    need(summary['owner_plan_sha256']==CYCLE_PLAN_SHA==sha(ROUTE_REG/'owner-plan.json'), 'exact registered GDN HTTP owner plan')
    for label in ('restoration','c1','c8','cycle'):
        row=summary['jobs'][label]
        need(type(row['returncode']) is int and row['returncode']==0 and row['audit_status']=='PASS'
             and row['ended_at'],'actual audit CLI failed: '+label)
        output=row['output'];need(sha(output['path'])==output['sha256']
            and Path(output['path']).stat().st_size==output['bytes'],'actual audit output changed')
    restore,cycle=[read(ANALYSIS/name) for name in ('restoration-audit.json','independent-cycle.json')]
    need(restore['status']==cycle['status']=='PASS' and restore['qualified'] is cycle['qualified'] is True
         and cycle['restoration_verified'] is cycle['independent_resource_verified'] is cycle['http_history_qualified'] is True
         and restore['journal_sha256']==cycle['journal_sha256']==sha(JOURNAL)
         and restore['owner_plan_sha256']==cycle['owner_plan_sha256']==CYCLE_PLAN_SHA,
         'independent restored/source/resource cycle gate')
    same(cycle['restoration'],restore,'exact independent restoration within cycle')
    archive=read(ARCHIVE/'summary.json');verified=read(ARCHIVE/'archive-verification.json')
    manifest=read(ARCHIVE/'manifest.json');stream=read(ARCHIVE/'archive-stream.json')
    need(archive['schema']=='GDN-http-archive-summary-v1' and archive['owner_plan_sha256']==CYCLE_PLAN_SHA
         and archive['owner_status']==archive['audit_status']=='PASS'
         and archive['restoration_qualified'] is archive['http_history_qualified'] is True
         and archive['performance_qualified'] is False,'successful immutable GDN HTTP archive required')
    need(verified['schema']=='GDN-http-archive-verification-v1' and verified['status']=='PASS'
         and verified['all_regular_members_decoded_verified'] is verified['all_original_files_rehashed'] is True
         and verified['all_tree_members_rechecked'] is True
         and verified['archive']==stream['archive'] and verified['members']==len(manifest['files']),
         'actual complete archive verification required (no tar rehash here)')
    sidecars={'restored-journal.json':JOURNAL,'restoration-audit.json':ANALYSIS/'restoration-audit.json',
        'audit-execution-summary.json':ANALYSIS/'execution-summary.json',
        **{name:ANALYSIS/name for name in ('independent-c1.json','independent-c8.json','independent-cycle.json')}}
    for name,path in sidecars.items():
        archived=ARCHIVE/name;digest=sha(path)
        need(sha(archived)==digest,'exact archive sidecar bytes: '+name)
        row=manifest['files']['archive-sidecars/'+name]
        need(row['sha256']==digest and row['bytes']==archived.stat().st_size,'archive manifest sidecar: '+name)
    return cycle


def gate(cpu_path):
    cpu=read(cpu_path)
    need(cpu['status']=='PASS' and cpu['failures']==cpu['errors']==cpu['skipped']==0
         and cpu['source_unchanged'] is True and cpu['denied_operations']==[]
         and cpu['forbidden_modules_loaded']==[], 'actual guarded CPU receipt must PASS without denied operations')
    rows=[x for x in cpu['suites'] if x['module']==SUITE]
    need(len(rows)==1 and rows[0]['expected_tests']==rows[0]['discovered_tests']==19,
         'full 19-method GDN formal suite required')
    for p,digest in ((FORMAL,FORMAL_SHA),(TEST,TEST_SHA)):
        need(sha(p)==digest==cpu['source_before'][str(p)]['sha256']==cpu['source_after'][str(p)]['sha256'],
             'CPU receipt must bind final actual source: '+str(p))
    return cpu,closed_gate()


def build(args):
    out=args.out.resolve()
    need(out.parent==ART and not out.exists(), 'fresh direct artifact output required')
    need(args.cpu_receipt is not None and args.cpu_receipt.is_absolute(), 'explicit actual CPU receipt required')
    out.mkdir();(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'inputs').mkdir();(out/'specs').mkdir()
    record=dict(schema='E024-gdn-http-formal-actual-prepare-v1',status='RUNNING',started_at=now(),
        argv=[sys.executable,*sys.argv],device_run=False,execution_registered=False,performance_qualified=False,
        runtime_weights_verified=False,steps_ledger_opened=False,specs=[],denied_operations=[],source_before={},source_after={},
        scope='CPU spec construction against real closed evidence; no inference, server, service, owner/controller, or numerical/performance grant.')
    inputs={}; guard=None
    def capture(path,name):
        path=Path(path); raw=path.read_bytes();target=out/'inputs'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        inputs[str(path)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),copy=str(target.relative_to(out)))
    def deny(kind,detail):
        record['denied_operations'].append(dict(kind=kind,detail=str(detail)));raise RuntimeError('forbidden CPU prepare operation: '+kind)
    def hook(event,arguments):
        if event in ('subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec','os.spawn','os.kill','os.killpg','socket.connect','socket.bind'):
            deny(event,arguments)
        if event=='open' and arguments and isinstance(arguments[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(arguments[0])
            if path.endswith(('.safetensors','.gguf','.bin','steps.jsonl')):deny('model-payload/steps-ledger open',path)
    class Imports(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in FORBIDDEN:deny('backend import',fullname)
            return None
    save(out/'result.json',record)
    try:
        gate(args.cpu_receipt)
        capture(args.cpu_receipt,'guarded-cpu-result.json')
        for path,name in ((EXECUTION,'gdn-execution-final.json'),(JOURNAL,'gdn-restored-journal.json'),
            (ANALYSIS/'execution-summary.json','gdn-audit-execution-summary.json'),
            (ANALYSIS/'restoration-audit.json','gdn-restoration-audit.json'),
            (ANALYSIS/'independent-cycle.json','gdn-independent-cycle.json'),
            (BASE_REG/'receipt.json','baseline-registration-receipt.json'),
            (BASE_REG/'owner-plan.json','baseline-owner-plan.json'),
            (ROUTE_REG/'owner-plan.json','gdn-owner-plan.json'),
            *[(ARCHIVE/name,'archive/'+name) for name in ('summary.json','archive-verification.json','manifest.json','archive-stream.json','restored-journal.json','restoration-audit.json','audit-execution-summary.json','independent-c1.json','independent-c8.json','independent-cycle.json')]):capture(path,name)
        baseline=read(BASE_REG/'owner-plan.json');baseline_receipt=read(BASE_REG/'receipt.json')
        need(sha(BASE_REG/'owner-plan.json')==BASE_PLAN_SHA==baseline_receipt['plan_sha256'], 'original formal preregistration plan')
        prepared_ref=baseline_receipt['cpu_prepare']; prepared_path=Path(prepared_ref['path'])
        need(sha(prepared_path)==prepared_ref['sha256']==BASE_PREPARED_SHA,'original source-pinned formal prepared specs')
        capture(prepared_path,'baseline-prepared-specs.json');prepared=read(prepared_path)
        need(prepared['status']=='PREPARED' and prepared['owner_plan_sha256']==BASE_PLAN_SHA,
             'actual original CPU prepared record')
        original={str(s['concurrency']):s for s in prepared['formal_specs']}
        need(set(original)=={'1','8'} and len(prepared['formal_specs'])==2, 'original C1/C8 spec pair')
        route_plan=read(ROUTE_REG/'owner-plan.json')
        need(sha(ROUTE_REG/'owner-plan.json')==read(ANALYSIS/'independent-cycle.json')['owner_plan_sha256'], 'closed cycle actual owner plan')
        same(route_plan['qualification'],baseline['qualification'],'same original D model/workload qualification')
        bindings={};captured={}
        for C in (1,8):
            folder=OWNER/'phases'/f'c{C}-replay';child=folder/'qualification'
            bindings[str(C)]=dict(audit=ref(ANALYSIS/f'independent-c{C}.json'),replay_plan=ref(folder/'plan.json'),
                child_directory=str(child),cycle_audit=ref(ANALYSIS/'independent-cycle.json'))
            capture(ANALYSIS/f'independent-c{C}.json',f'c{C}-independent-numerical.json')
            capture(folder/'plan.json',f'c{C}-gdn-replay-plan.json')
            replay=read(folder/'plan.json');cap=Path(replay['capture']['spec']['path'])
            need(sha(cap)==replay['capture']['spec']['sha256'],'actual source-bound capture spec')
            capture(cap,f'c{C}-capture-spec.json');captured[str(C)]=read(cap)
        guard=Imports();sys.meta_path.insert(0,guard);sys.addaudithook(hook)
        spec=importlib.util.spec_from_file_location('_actual_gdn_formal_preparation',FORMAL)
        formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal)
        sources={**formal.source_pins(),str(TEST):TEST_SHA,str(Path(__file__).resolve()):sha(__file__)}
        for path,digest in sources.items():
            need(sha(path)==digest,'actual source changed before prepare')
            record['source_before'][path]=dict(sha256=digest,bytes=Path(path).stat().st_size)
            target=out/'frozen-source'/path.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes())
        save(out/'result.json',record)
        with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            for phase in formal.TRIALS:
                repeat,C,arm=phase.split('-');repeat=int(repeat[1:]);C=int(C[1:]);base=original[str(C)];cap=captured[str(C)]
                same(base['qualification'],baseline['qualifications'][str(C)],'original actual numerical binding')
                same(base['transport'],cap['transport'],'original formal versus actual capture transport')
                prepared=formal.prepare(base['base_plan'],C,repeat,arm=arm,
                    baseline_qualification=base['qualification'],gdn_qualification=bindings[str(C)],
                    environment=cap['environment'],transport=cap['transport'],deadlines=base['deadlines'])
                path=out/'specs'/(phase+'.json');save(path,prepared)
                item=dict(phase=phase,status='PREPARED_NOT_VALIDATED',spec=ref(path));record['specs'].append(item);save(out/'result.json',record)
                formal.validate(prepared)
                need(prepared['phase']==phase and prepared['observer_enabled'] is False, 'actual reconstructed phase/observer mode')
                for name,count in (('warm',8),('formal',64)):
                    need(len(prepared['jobs'][name])==count,'fixed warm/formal counts')
                    for i,row in enumerate(prepared['jobs'][name]):
                        same({k:v for k,v in row.items() if k!='request_id'},
                             {k:v for k,v in base['jobs'][name][i].items() if k!='request_id'},'unchanged full D request body/fixture/path')
                item.update(status='PASS',warm_requests=8,formal_requests=64,
                    command=prepared['command'],identity=prepared['identity'],sources=prepared['sources'],
                    source_scope='Original and new source pins verified; no runtime/weights payload verification.')
                save(out/'result.json',record)
        need(len(record['specs'])==12 and all(s['status']=='PASS' for s in record['specs']),'all 12 actual specs must reconstruct')
        for path,before in record['source_before'].items():record['source_after'][path]=dict(sha256=sha(path),bytes=Path(path).stat().st_size)
        record['source_unchanged']=record['source_before']==record['source_after']
        record['inputs_unchanged']=all(sha(path)==meta['sha256'] for path,meta in inputs.items())
        need(record['source_unchanged'] and record['inputs_unchanged'] and not record['denied_operations'],'source/input/guard invariant')
        record.update(status='PASS',prepared_specs=12,warm_requests=96,formal_requests=768,total_requests=864)
    except BaseException as exc:
        record.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        if guard in sys.meta_path:sys.meta_path.remove(guard)
        for path in record['source_before']:
            try:record['source_after'][path]=dict(sha256=sha(path),bytes=Path(path).stat().st_size)
            except BaseException as exc:record['source_after'][path]=dict(error=str(exc))
        record['source_unchanged']=bool(record['source_before']) and record['source_before']==record['source_after']
        record['inputs_unchanged']=bool(inputs) and all(Path(path).is_file() and sha(path)==meta['sha256'] for path,meta in inputs.items())
        if not record['source_unchanged'] or not record['inputs_unchanged']:record['status']='FAIL'
        record['forbidden_modules_loaded']=sorted(n for n in sys.modules if n.split('.')[0] in FORBIDDEN)
        if record['forbidden_modules_loaded']:record['status']='FAIL'
        record.update(finished_at=now(),inputs=inputs);save(out/'result.json',record)
        inventory={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()}
        save(out/'artifact-manifest.json',dict(schema='cpu-prepare-artifact-manifest-v1',files=inventory))
    print(json.dumps(dict(status=record['status'],out=str(out),prepared_specs=record.get('prepared_specs'),error=record.get('error'))))
    return 0 if record['status']=='PASS' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--cpu-receipt',type=Path);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if not args.run:
        print(json.dumps(dict(action='describe',device_run=False,output=str(args.out),requires='--run --cpu-receipt ABS_PASS_RESULT; actual four audits and immutable archive PASS, original frozen specs',specs=12)));return 0
    return build(args)


if __name__=='__main__':raise SystemExit(main())
