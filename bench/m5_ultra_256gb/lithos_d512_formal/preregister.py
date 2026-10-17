"""Explicit CPU-only registration of six frozen Lithos formal trials.

Writes only a fresh prereg-formal-v1 directory. No subprocess, HTTP, service,
listener, tokenizer/model import or model/pack payload hashing. The root owner
must separately perform read-only controller preparation, retain its real
PREPARED journal SHA, and create receipt.json before executing the cycle.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREREG = HERE/'prereg-formal-v1'
OLD = HERE.parent/'lithos_d512_retry/prereg-qualification-v2'
ARCHIVE = ROOT/'docs/experiments/m5-ultra-256gb/E023-lithos-d512-qualification-v2'
ARTIFACTS = ROOT/'.artifacts/m5-ultra-256gb'
PATHS = {name: ARTIFACTS/('lithos-d512-formal-v1-'+name)
         for name in ('owner','cold-journal','execution','analysis','cpu-prepare')}
OLD_PLAN_SHA = 'e72792b61039414d0c16d9f5f3216619aed62d2d3c23536fe491d409383ab354'
FIXED = {
    'contract.py': '2c5b221608e8c53b39116b2e13b247b9fb78d83c74cbee069179efbea9210b32',
    'formal.py': 'fca180b4296e352e7b75db517ac897fe56d25ceea313d31b6a6c091a515c73ff',
    'owner.py': '49dfa344b0abca24ba3ec84a14b44018751cb57353b82fd16197ca4bd49c9b70',
    'controller.py': '5c13010c5645e93eb099592e32ee6192f8a2ec7ca81c5ab2ed67c06f7e4273c2',
    'audit_restore.py': '41d260e63ce13c5c5912217ae74169e5c3cd118af32f3667aefffc3637a2d66f',
    'audit_resources.py': '398ff7a52dba307418f4c2725f33988b1571d65b11e3f46c9e97759f3a167c9e',
    'audit_formal.py': '768dc970dca201a700c94e07e17c5fdd399c903f444272ea150eb6336575c9c2',
    'execute.py': 'f18ec943335bc149b5f96e7567d8e9aae71a8895f6092ebcfcb6dc685acdc250',
}
DEADLINES = dict(ready=180, request=600, warm=720, formal=2400, overall=3600,
                 close=10, term=15, kill=15, resource_interval=5)
AUDITS = {
    'audit': ('independent-qualification.json', 'df64cd94a1ec47dc4f5dd8f749e61066eba17d126abc0860ad929b1703f16189'),
    'resource_audit': ('resource-summary.json', 'f171bcbb8f0f709ececc7b9b35a1f25f5e0e4b3194ed5d542799a8fd5e38930e'),
    'restoration_audit': ('restoration-audit.json', '269a9a85b50674077fee8dc464a0edc4a0531fb13e5386d95dbc8cc57ba66e43'),
}
PREDECESSOR_SHA = '302a54499f03d2cac02981a5ad06f9369a255aa5d78be90243736205529be6af'
PROPOSAL_SHA = 'e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8'
CONTRACT = 'cold-restart-persisted-visible-v1'


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def ref(path): return dict(path=str(path), sha256=sha(path))
def write(path, value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
def load(name):
    path=HERE/name; need(sha(path)==FIXED[name], 'unfrozen registration dependency: '+name)
    spec=importlib.util.spec_from_file_location('_lithos_preregister_'+path.stem,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def prepare():
    # Every source is independently finalized before any output is created.
    for name, digest in FIXED.items():
        need(len(digest)==64 and all(c in '0123456789abcdef' for c in digest), 'source not frozen: '+name)
        need(sha(HERE/name)==digest, 'frozen source changed: '+name)
    need(not PREREG.exists() and all(not p.exists() for p in PATHS.values()), 'fresh prereg and five output paths required')
    need(sha(OLD/'owner-plan.json')==OLD_PLAN_SHA, 'original qualified owner plan changed')
    old=read(OLD/'owner-plan.json'); qowner=ARTIFACTS/'lithos-d512-qualification-v2-owner'
    need(sha(qowner/'owner-plan.json')==OLD_PLAN_SHA, 'actual qualified owner plan copy')
    qualification=dict(owner_directory=str(qowner), owner_plan=ref(qowner/'owner-plan.json'))
    for key,(name,digest) in AUDITS.items():
        need(sha(ARCHIVE/name)==digest, 'archived independent audit changed: '+name)
        qualification[key]=ref(ARCHIVE/name)
    need(sha(ARCHIVE/'restored-journal.json')==PREDECESSOR_SHA, 'immutable restored predecessor changed')
    predecessor=dict(**ref(ARCHIVE/'restored-journal.json'),audit=qualification['restoration_audit'])
    need(read(ARCHIVE/'restored-journal.json')['replacement']['pid']==28531, 'registered actual predecessor PID')
    http=copy.deepcopy(old['http']);http['deadlines']=dict(DEADLINES)
    need(http['port']==18881 and http['api_key']==http['environment']['LITHOS_METAL_API_KEY'], 'same localhost key/port')
    owner=load('owner.py');controller=load('controller.py')
    plan=owner.plan_template(qualification=copy.deepcopy(old['qualification']),
        qualifications={str(C):copy.deepcopy(qualification) for C in (1,8)},
        cold_restart=dict(journal_path=str(PATHS['cold-journal']),controller=ref(HERE/'controller.py'),
            approved_contract=CONTRACT,proposal_plan_sha256=PROPOSAL_SHA,predecessor=predecessor),
        http=http,verification=copy.deepcopy(old['verification']))
    need(plan['verification']==old['verification'] and plan['qualification']==old['qualification'], 'unchanged full dependency/workload inventory')
    PREREG.mkdir()
    try:
        write(PREREG/'owner-plan.json',plan)
        plan_ref=ref(PREREG/'owner-plan.json')
        args=SimpleNamespace(plan=PREREG/'owner-plan.json',plan_sha256=plan_ref['sha256'],out=PATHS['owner'])
        _,_,_,_,specs=owner.validate_plan(args)
        need(len(specs)==6 and all(len(s['jobs']['warm'])==8 and len(s['jobs']['formal'])==64 for s in specs), 'six actual 8+64 specs')
        # This is source validation only, never controller.prepare or System().
        lifecycle_pins=controller.source_pins(plan)
        python=plan['verification']['python'];common=['--journal',str(PATHS['cold-journal']),
            '--plan-sha256',PROPOSAL_SHA]
        commands=dict(
            cpu_prepare=[python,'-I','-B',str(HERE/'owner.py'),'--plan',plan_ref['path'],'--plan-sha256',plan_ref['sha256'],
                         '--out',str(PATHS['cpu-prepare']),'--prepare-only'],
            readonly_service_prepare=[python,'-I','-B',str(HERE/'controller.py'),'prepare',*common,
                '--owner-directory',str(PATHS['owner']),'--owner-plan',plan_ref['path'],'--owner-plan-sha256',plan_ref['sha256']],
            execute_after_receipt_and_commit=[python,'-I','-B',str(HERE/'execute.py')],
            restore_audit=[python,'-I','-B',str(HERE/'audit_restore.py'),'--journal',str(PATHS['cold-journal']),
                '--out',str(PATHS['analysis']/'restoration-audit.json')],
            resource_audit=[python,'-I','-B',str(HERE/'audit_resources.py'),'--owner',str(PATHS['owner']),
                '--out',str(PATHS['analysis']/'resource-summary.json')],
            formal_audit=[python,'-I','-B',str(HERE/'audit_formal.py'),'--owner',str(PATHS['owner']),
                '--journal',str(PATHS['cold-journal']),'--plan-sha256',plan_ref['sha256'],
                '--out',str(PATHS['analysis']/'independent-formal.json')])
        write(PREREG/'commands.json',commands)
        write(PREREG/'preparation.json',dict(schema='Lithos-D512-formal-prereg-prepare-v1',status='CPU_PREPARED',
            owner_plan=plan_ref,prepare_source=ref(Path(__file__)),paths={k:str(v) for k,v in PATHS.items()},
            trial_specs=specs,deadline_seconds=plan['deadline_seconds'],original_owner_plan=ref(OLD/'owner-plan.json'),
            runtime_weights_pack_verified=False,service_action=False,
            scope='Small source/metadata checks only; runtime/model/pack payloads checked by actual owner before/after. No PREPARED journal or execution receipt fabricated.'))
        pins={}
        def add(items):
            for name,digest in items.items():
                need(name not in pins or pins[name]==digest, 'source pin conflict: '+name)
                need(sha(name)==digest, 'source changed during registration: '+name); pins[name]=digest
        # Retain all already reviewed transitive lifecycle/audit helpers. This is
        # a small source/evidence inventory, distinct from the five payload groups.
        add(read(OLD/'receipt.json')['source_pins'])
        add(lifecycle_pins);add(plan['execution']['pins'])
        for s in specs:add(s['sources'])
        add({str(HERE/name):digest for name,digest in FIXED.items()})
        add({str(p):sha(p) for p in [Path(__file__),HERE/'README.md',*sorted(HERE.glob('test_*.py')),
             PREREG/'owner-plan.json',PREREG/'commands.json',PREREG/'preparation.json']})
        write(PREREG/'source-manifest.json',dict(schema='Lithos-D512-formal-source-manifest-v1',source_pins=pins,
            owner_plan=plan_ref,qualification=qualification,predecessor=predecessor,
            payload_inventory_sha256=hashlib.sha256(json.dumps(plan['verification'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            runtime_model_pack_rehashed=False,
            scope='Freeze execution/audit code and immutable inputs. Root adds actual read-only PREPARED journal digest to receipt.json; this file is not permission or a device result.'))
    except BaseException as exc:
        write(PREREG/'preparation-failure.json',dict(status='FAIL',type=type(exc).__name__,error=str(exc)))
        raise
    return dict(status='CPU_PREPARED',owner_plan=plan_ref,deadline_seconds=plan['deadline_seconds'],
        trials=plan['trials'],warm_requests=48,formal_requests=384,source_manifest=ref(PREREG/'source-manifest.json'),
        runtime_weights_pack_verified=False,service_action=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true',required=True,help='explicitly create fresh CPU registration; never starts a child/service')
    parser.parse_args(argv)
    print(json.dumps(prepare(),sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
