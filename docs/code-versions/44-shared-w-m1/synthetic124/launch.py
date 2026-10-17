"""Root-owned GPU worker exec; default inert. Never archives raw environment."""
from pathlib import Path
import argparse,importlib.util,json,os
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v124_localqual_launch_support',D/'localqual.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
def environment(inherited):
 env=dict(inherited)
 for key in q.selected(env):env.pop(key,None)
 assert not q.selected(env)
 env.update(q.CONTROLS)
 assert q.selected(env)==q.CONTROLS
 return env
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-gpu',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args();q.identity()
 if not a.execute_gpu:
  print(json.dumps({'status':'INERT_CLEAN_LAUNCH_READY','GPU_used':False,'controls':q.CONTROLS,'parent_selected_policy':'remove every selected key, then set exact child controls'}));return
 assert a.out is not None and a.out.is_absolute() and not a.out.exists(),'fresh absolute output required'
 env=environment(os.environ)
 print(json.dumps({'status':'EXEC_ROOT_OWNED_LOCALQUAL','selected_keys_count':len(q.selected(env)),'raw_environment_logged':False}),flush=True)
 os.execve(q.PYTHON,[q.PYTHON,'-B',str(D/'localqual.py'),'--execute-gpu','--out',str(a.out)],env)
if __name__=='__main__':main()
