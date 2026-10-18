"""Explicit GLM CPU input inventory builder; no service/backend or weight hashing.

Only --run hashes listed small sources/runtime binaries, reads fixed download
receipts and stats the 62 shards. Original full payload SHA values are retained
provenance, never presented as a new payload check. Imports are stdlib only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
PROBE_SHA='cd9df050384af83d92f0321d95f5bba70dca349defbedf0bb7d1d1e8b57385fc'
TOTAL_BYTES=203992076296


def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_bytes())
def stamp(path):
    s=Path(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
def digest(path,check=lambda:None):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1<<20):check();h.update(chunk)
    return h.hexdigest()
def ref(path):return dict(path=str(Path(path).resolve()),sha256=digest(path))
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def probe_module():
    path=HERE.parent/'glm_flash/load_probe.py';need(digest(path)==PROBE_SHA,'frozen original GLM probe changed')
    sp=importlib.util.spec_from_file_location('glm_input_original_probe',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def inventory(probe,outer_python,check):
    """Supplied reviewed set; no recursive Python/native dependency claim."""
    refs={k:dict(path=str(probe.REPO/'.artifacts/m5-ultra-256gb/glm-flash-download-v1'/name),sha256=h)
          for k,(name,h) in probe.FIXED_RECEIPTS.items()}
    for r in refs.values():check();need(digest(r['path'],check)==r['sha256'],'fixed model receipt changed')
    download=read(refs['download_receipt']['path']);header=read(refs['header_inventory']['path'])
    manifest=read(probe.HERE/'metadata-v1/manifest.json')
    expected={r['name']:dict(bytes=r['bytes'],sha256=r['sha256']) for r in manifest['shards']}
    need(manifest['revision']==probe.REVISION and len(expected)==62 and sum(r['bytes'] for r in expected.values())==TOTAL_BYTES,
         'fixed 62-shard manifest')
    need(download['status']=='PASS_PINNED_DOWNLOAD_AND_SHA_ONLY' and download['revision']==probe.REVISION
         and download['snapshot']==str(probe.SNAPSHOT) and len(download['shards'])==62
         and {r['name']:dict(bytes=r['bytes'],sha256=r['sha256']) for r in download['shards']}==expected
         and all(r['sha256_verified'] is True for r in download['shards']), 'retained full-download provenance')
    need(header['status']=='PASS_HEADER_INVENTORY_ONLY' and header['snapshot']==str(probe.SNAPSHOT),'fixed header receipt')
    shards={}
    for name,row in expected.items():
        check();need(Path(name).name==name and name.endswith('.safetensors'),'simple original shard filename')
        stat=stamp(probe.SNAPSHOT/name);need(stat['bytes']==row['bytes'],'shard size changed')
        shards[name]=dict(sha256=row['sha256'],stat=stat)  # Deliberately never digest(shard).
    paths={p.resolve() for p in probe.REQUIRED_SOURCES if p.is_relative_to(probe.SITE)}
    paths.update(p.resolve() for p in (probe.VENV/'bin/python',Path(outer_python),Path(sys.executable),
        probe.SITE/'mlx/core.cpython-313-darwin.so',probe.SITE/'mlx/lib/libmlx.dylib',probe.SITE/'mlx/lib/libjaccl.dylib'))
    runtime={}
    for path in sorted(paths):
        check();need(path.is_absolute() and path.is_file() and not path.is_relative_to(probe.SNAPSHOT.resolve()),'real runtime file, not model payload')
        before=stamp(path);h=digest(path,check);need(stamp(path)==before,'runtime changed during input preparation')
        runtime[str(path)]=dict(sha256=h,stat=before)
    metadata=probe.model_metadata()  # Fixed small metadata and shard stat only.
    need(metadata['shard_stats']=={k:r['stat'] for k,r in shards.items()},'shard stat changed during preparation')
    return dict(schema='GLM-model-runtime-inventory-v1',snapshot=str(probe.SNAPSHOT),download_receipt=refs['download_receipt'],
        payload_scope='retained-download-full-SHA256-and-registered-stat;no-new-payload-hash',shards=shards,runtime=runtime),refs,metadata


def prepare(args):
    out=Path(args.out).resolve();need(not out.exists() and out.parent.is_dir(),'new input directory required')
    prompt=Path(args.prompt).resolve();raw=prompt.read_bytes();raw.decode('utf-8',errors='strict')
    need(0<len(raw)<=4096 and b'\0' not in raw and hashlib.sha256(raw).hexdigest()==args.prompt_sha256,'fixed bounded plain prompt')
    outer=Path(args.outer_python).resolve();need(outer.is_file(),'actual outer Python executable')
    start=time.monotonic()
    def check():need(time.monotonic()-start<=300,'CPU input inventory preparation deadline')
    probe=probe_module();value,refs,metadata=inventory(probe,outer,check)
    pins={}
    for path in probe.REQUIRED_SOURCES:
        check();pins[str(path)]=probe.sha(path)
    plan=dict(schema='GLM-official-load-plan-v1',snapshot=str(probe.SNAPSHOT),prompt=dict(path=str(prompt),sha256=args.prompt_sha256),
        **refs,source_pins=pins,environment=probe.ENVIRONMENT,deadlines=dict(work_seconds=1800,request_seconds=180,shutdown_seconds=60))
    probe.source_check(plan);check();out.mkdir()
    write(out/'probe-plan.json',plan);write(out/'model-runtime-inventory.json',value);write(out/'model-metadata.json',metadata)
    result=dict(schema='GLM-cpu-input-preparation-v1',status='CPU_INPUTS_ONLY',probe_plan=ref(out/'probe-plan.json'),
        verification=ref(out/'model-runtime-inventory.json'),builder=ref(__file__),probe=ref(HERE.parent/'glm_flash/load_probe.py'),
        outer_python=str(outer),runtime_files=len(value['runtime']),shards=62,weight_payload_SHA_rechecked=False,
        source_pins=pins,scope='Listed supplied runtime hashes plus retained download hashes/new shard stat; not complete dependency closure.',
        service_prepared=False,model_loaded=False)
    write(out/'preparation.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    p.add_argument('--prompt',type=Path);p.add_argument('--prompt-sha256');p.add_argument('--outer-python',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args(argv)
    if not a.run:
        need(not any((a.prompt,a.prompt_sha256,a.outer_python,a.out)),'explicit --run needed')
        print('DESCRIPTION_ONLY: explicit prompt SHA and outer Python; no model payload hash, device or service action.');return 0
    need(all((a.prompt,a.prompt_sha256,a.outer_python,a.out)),'complete explicit CPU input arguments')
    prepare(a);return 0


if __name__=='__main__':raise SystemExit(main())
