"""Engine-specific preimport identity, then unchanged pinned official entry."""
import argparse,hashlib,importlib.metadata,json,os,runpy,sys
from pathlib import Path
D=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--engine',choices=['omlx','lithos'],required=True);ap.add_argument('--receipt',type=Path,required=True);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
reg=json.loads((D/'registry.json').read_text())[a.engine];pins=json.loads((D/'pins.json').read_text())
assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
assert sys.executable==reg['actual_environment']['python_executable']
for p,h in reg['source_native_files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
controls={k:os.environ.get(k) for k in ['MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER','MLX_BFS_MAX_WIDTH','MLX_METAL_FAST_SYNCH']};assert controls==pins['child_controls']
assert not [k for k in os.environ if k.startswith(('OMLX_','DYLD_','METAL_'))] and not [k for k in os.environ if k.startswith('MLX_') and k not in controls]
with a.receipt.open('x') as f:json.dump({'engine':a.engine,'pid':os.getpid(),'ppid':os.getppid(),'python_executable':sys.executable,'python_version':sys.version,'source_native_tree_sha256':reg['tree_sha256'],'files_checked':len(reg['source_native_files']),'controls':controls,'mlx_not_imported':True},f,indent=2)
if a.engine=='omlx':
 assert cmd[:2]==reg['reference_launch'][:2];sys.argv=cmd;runpy.run_path(cmd[0],run_name='__main__')
else:
 assert cmd[:3]==reg['reference_launch'][:3];sys.argv=[cmd[2],*cmd[3:]];runpy.run_module(cmd[2],run_name='__main__',alter_sys=True)
