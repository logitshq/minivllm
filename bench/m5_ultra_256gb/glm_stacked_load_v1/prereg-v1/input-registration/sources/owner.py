"""E036 checkpoint admission adapter; original v3 child/monitor/reap unchanged."""
import importlib.util as _importlib
from pathlib import Path as _Path
_path=_Path(__file__).resolve().with_name('_reuse.py')
_spec=_importlib.spec_from_file_location('_glm_stacked_reuse',_path)
_adapter=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_adapter)

_v3=_adapter.install(globals(),'glm_flash_cycle_v3/owner.py')
FIXED=dict(FIXED);FIXED['../glm_flash/load_probe.py']='559993d5fedb6714c3909b1db981e31de8479d6b878e2295051fc2c8b62041f2'
LEGACY_MATCHES=(*LEGACY_MATCHES,'/glm_stacked_load_v1/load_probe.py')
def helper(relative):
    path=HERE/'load_probe.py' if relative=='../glm_flash/load_probe.py' else (HERE/relative).resolve()
    return module(path,FIXED[relative])
def source_pins(plan):
    pins={str((HERE/'load_probe.py' if k=='../glm_flash/load_probe.py' else (HERE/k).resolve())):h for k,h in FIXED.items()}
    pins.update(_adapter.pins());pins.update(_reuse.support_pins());pins[str(MONITOR_PATH)]=MONITOR_SHA
    pins[str(Path(__file__).resolve())]=sha(__file__)
    c=plan['cold_restart']['controller'];pins[c['path']]=c['sha256']
    child=read(reference(plan['probe_plan']))
    for path,h in child['source_pins'].items():
        need(path not in pins or pins[path]==h,'conflicting child source');pins[path]=h
    need(sha(MONITOR_PATH)==MONITOR_SHA,'unchanged dual monitor')
    return pins
def verify_inventory(plan,probe,check):
    receipt=read(reference(plan['verification']));child=read(reference(plan['probe_plan']))
    manifest=probe.checkpoint_evidence(child)
    need(set(receipt)=={'schema','snapshot','repack_receipt','header_verification','payload_scope','shards','runtime'} and
         receipt['schema']=='GLM-derived-model-runtime-inventory-v1' and receipt['snapshot']==str(probe.SNAPSHOT) and
         receipt['payload_scope']==manifest['payload_scope'] and
         all(receipt[k]==child[k] for k in ('repack_receipt','header_verification')),'real derived inventory scope')
    need(set(receipt['shards'])==set(manifest['shards']) and len(receipt['shards'])==58,'58 real derived shards')
    for name,row in receipt['shards'].items():
        check();expected=manifest['shards'][name]
        need(file_record(row) and row['sha256']==expected['sha256'] and
             row['stat']['bytes']==expected['identity']['size'] and stamp(probe.SNAPSHOT/name)==row['stat'],'derived shard stat')
    runtime=receipt['runtime'];need(type(runtime) is dict and runtime,'explicit supplied runtime inventory')
    required={str(p.resolve()) for p in probe.REQUIRED_SOURCES if p.is_relative_to(probe.SITE)}
    required.update(str(p.resolve()) for p in (probe.VENV/'bin/python',Path(sys.executable),
        probe.SITE/'mlx/core.cpython-313-darwin.so',probe.SITE/'mlx/lib/libmlx.dylib',probe.SITE/'mlx/lib/libjaccl.dylib'))
    need(required<=set(runtime),'same actual interpreter/vendor/native runtime closure plus MTP/sanitizer')
    for name,row in runtime.items():
        check();path=Path(name)
        need(path.is_absolute() and str(path.resolve())==name and file_record(row) and
             stamp(path)==row['stat'] and sha(path)==row['sha256'] and stamp(path)==row['stat'],'runtime changed: '+name)
    metadata=probe.model_metadata()
    need(metadata['shard_stats']=={k:r['stat'] for k,r in receipt['shards'].items()},'matching actual derived stat evidence')
    return dict(schema='GLM-derived-owner-model-runtime-proof-v1',status='PASS',inventory=plan['verification'],
        snapshot=str(probe.SNAPSHOT),payload_scope=receipt['payload_scope'],weight_payload_SHA_rechecked=False,
        shard_count=58,runtime_files=len(runtime),runtime_SHA_rechecked=True,metadata=metadata,
        scope='E036 producer byte proof and independent header proof; current header/stat and supplied runtime hashes. No new payload SHA, loader equivalence or recursive closure claim.')
_old_receipts="""    for k, (filename, digest) in probe.FIXED_RECEIPTS.items():
        need(child[k] == dict(path=str(ROOT/'.artifacts/m5-ultra-256gb/glm-flash-download-v1'/filename),sha256=digest), 'original model receipt')
        reference(child[k])"""
_adapter.variant(globals(),_v3._validate_source,[(_old_receipts,'    probe.checkpoint_evidence(child)')])
_validated_original_contract=validate_plan
validate_plan=_adapter.bind(_v3.validate_plan,globals())
if __name__=='__main__':raise SystemExit(main())
