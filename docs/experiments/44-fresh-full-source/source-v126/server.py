"""V126 unified HTTP executable selecting an immutable native startup profile.

Default is exact125 C1. --enable-full-curve enables only declared C2/C4 ordinary
batch configuration. No dynamic migration or serial queue masquerading as batch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from entry_policy import choose
D=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__,add_help=False)
    ap.add_argument('--config',type=Path,default=D/'profiles.json')
    ap.add_argument('--concurrency',type=int,choices=[1,2,4],default=1)
    ap.add_argument('--enable-full-curve',action='store_true')
    ap.add_argument('--cpu-plan',action='store_true')
    args,rest=ap.parse_known_args()
    route=choose(args.concurrency,enable_full_curve=args.enable_full_curve)
    profiles=json.loads(args.config.read_text());selected=profiles[route]
    if args.cpu_plan:
        print(json.dumps({'status':'CPU_PLAN_GPU_UNRUN','route':route,'concurrency':args.concurrency,
            'scheduler_max_seqs':selected['max_seqs'],'physical_capacity':selected['physical_capacity'],
            'native_adapter':selected['native_adapter'],'dynamic_migration':False,'GPU_used':False}))
        return 0
    pins=json.loads((D/'source-pins.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for path,digest in pins['files'].items():
        assert sha(path)==digest,path
    assert args.config.resolve()==(D/'profiles.json').resolve()
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
    entry=D/selected['entry'];assert sha(entry)==selected['entry_sha256']
    if route=='direct_c1':
        assert args.concurrency==1
        assert sha(selected['runtime_config'])==selected['runtime_config_sha256']
        delegated=[*rest,'--config',selected['runtime_config']]
    else:
        assert not any(k.startswith(('DYLD_','METAL_')) for k in os.environ)
        assert sha(selected['source_manifest'])==selected['source_manifest_sha256']
        delegated=[*rest,'--package-root',selected['package_root'],'--source-manifest',selected['source_manifest'],
            '--batch','4','--gates','on','--client-batch',str(args.concurrency)]
    # Selection is fixed before any model import/request; each point starts fresh.
    sys.argv=[str(entry),*delegated]
    runpy.run_path(str(entry),run_name='__main__')
    return 0
if __name__=='__main__':raise SystemExit(main())
