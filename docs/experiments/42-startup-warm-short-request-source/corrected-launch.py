"""Correct the historical V123 README parent environment preflight. Inert by default.
No raw environment values or actual process argv are printed or archived.
Root-only GPU execution requires --execute-gpu and a fresh absolute --out.
"""
from pathlib import Path
import argparse,json,os,sys
P=Path('/private/tmp/minivllm-engine-optimization-20261014')
PY='/Users/ryanli/code/minivllm/.venv/bin/python'
OWNER=P/'v123-startup-ordinary-warm-build1/owner.py'
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-gpu',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
 if not a.execute_gpu:
  print(json.dumps({'status':'INERT_CPU_ONLY','GPU_used':False,'plan':'remove all keys returned by frozen common.selected(env), then exec original owner; child controls set by owner'}));return
 assert a.out is not None and a.out.is_absolute() and not a.out.exists(),'fresh absolute output required'
 sys.path.insert(0,str(P/'m4-final-native15-official-attempt2'))
 import common as c
 env=dict(os.environ)
 for key in c.selected(env):env.pop(key,None)
 assert not c.selected(env)
 print(json.dumps({'parent_selected':[],'scope':'no environment values printed'}),flush=True)
 os.execve(PY,[PY,'-B',str(OWNER),'--out',str(a.out)],env)
if __name__=='__main__':main()
