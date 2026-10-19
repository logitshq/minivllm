"""CPU-only actual E036 input preparation. No payload hash, service or model import."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROBE_SHA='559993d5fedb6714c3909b1db981e31de8479d6b878e2295051fc2c8b62041f2'
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while raw:=f.read(1<<20):h.update(raw)
 return h.hexdigest()
def stamp(p):
 s=Path(p).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def probe_module():
 p=HERE/'load_probe.py';need(sha(p)==PROBE_SHA,'fixed derived probe')
 s=importlib.util.spec_from_file_location('_derived_probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prepare(a):
 out=Path(a.out).resolve();need(not out.exists() and out.parent.is_dir(),'fresh inputs')
 prompt=Path(a.prompt).resolve();raw=prompt.read_bytes();raw.decode('utf-8')
 need(0<len(raw)<=4096 and b'\0' not in raw and sha(prompt)==a.prompt_sha256,'bounded pinned prompt')
 outer=Path(a.outer_python).absolute();need(outer.is_file(),'real outer interpreter')
 probe=probe_module();metadata=probe.model_metadata();manifest=probe.checkpoint_module().manifest()
 pins={str(p):probe.sha(p) for p in probe.REQUIRED_SOURCES}
 paths={p.resolve() for p in probe.REQUIRED_SOURCES if p.is_relative_to(probe.SITE)}
 paths.update(p.resolve() for p in (probe.VENV/'bin/python',outer,Path(sys.executable),probe.SITE/'mlx/core.cpython-313-darwin.so',probe.SITE/'mlx/lib/libmlx.dylib',probe.SITE/'mlx/lib/libjaccl.dylib'))
 started=time.monotonic();runtime={}
 for path in sorted(paths):
  need(time.monotonic()-started<300 and path.is_file() and not path.is_relative_to(probe.SNAPSHOT),'bounded actual runtime-only inventory')
  st=stamp(path);h=sha(path);need(stamp(path)==st,'stable runtime');runtime[str(path)]=dict(sha256=h,stat=st)
 plan=dict(schema='GLM-official-load-plan-v1',snapshot=str(probe.SNAPSHOT),prompt=ref(prompt),
   **probe.FIXED_EVIDENCE,source_pins=pins,environment=probe.ENVIRONMENT,deadlines=dict(work_seconds=1800,request_seconds=180,shutdown_seconds=60))
 inventory=dict(schema='GLM-derived-model-runtime-inventory-v1',snapshot=str(probe.SNAPSHOT),
   **probe.FIXED_EVIDENCE,payload_scope=manifest['payload_scope'],
   shards={k:dict(sha256=v['sha256'],stat=metadata['shard_stats'][k]) for k,v in manifest['shards'].items()},runtime=runtime)
 need(probe.model_metadata()==metadata,'metadata stable during preparation');probe.source_check(plan)
 out.mkdir();write(out/'probe-plan.json',plan);write(out/'model-runtime-inventory.json',inventory);write(out/'model-metadata.json',metadata)
 result=dict(status='CPU_INPUTS_ONLY',probe_plan=ref(out/'probe-plan.json'),verification=ref(out/'model-runtime-inventory.json'),
   source_pins=pins,shards=58,runtime_files=len(runtime),weight_payload_SHA_rechecked=False,service_prepared=False,model_loaded=False)
 write(out/'preparation.json',result);return result
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
 for name in ('prompt','outer-python','out'):p.add_argument('--'+name,type=Path)
 p.add_argument('--prompt-sha256');a=p.parse_args(argv)
 if not a.run:
  need(not any((a.prompt,a.outer_python,a.out,a.prompt_sha256)),'explicit --run required');print('DESCRIPTION_ONLY: 58-shard derived metadata/runtime inputs; no service or model.');return 0
 need(all((a.prompt,a.outer_python,a.out,a.prompt_sha256)),'complete explicit inputs');prepare(a);return 0
if __name__=='__main__':raise SystemExit(main())
