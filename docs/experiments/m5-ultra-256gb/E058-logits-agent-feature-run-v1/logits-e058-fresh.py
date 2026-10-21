from pathlib import Path
import os,sys,json,subprocess,time,shutil
name,attempt,prompt_path=sys.argv[1:]
root=Path('/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/logits-build-e058-2026-10-21-v1');folder=root/name
profile={'timing':'logits-muse','implementation':'logits-deepseek','independent_audit':'logits-kimi'}[name]
thread=next(json.loads(l)['thread_id'] for l in (folder/'stdout.jsonl').read_text().splitlines() if json.loads(l).get('type')=='thread.started')
out=folder/('attempt-'+attempt);out.mkdir(exist_ok=False)
for p in folder.glob('*.py'):shutil.copyfile(p,out/p.name)
env=dict(os.environ);env['CODEX_HOME']=str(folder/'codex-home')
cmd=['/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex','exec','--strict-config','-p',profile,'--skip-git-repo-check','--disable','apps','--disable','plugins','--disable','multi_agent','-s','workspace-write','-C',str(folder),'--json','-o',str(out/'final.md'),Path(prompt_path).read_text()]
(out/'command.json').write_text(json.dumps(cmd,indent=2));start=time.monotonic()
with (out/'stdout.jsonl').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
 p=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr)
 try:rc=p.wait(timeout=480)
 except subprocess.TimeoutExpired:p.terminate();rc=p.wait(timeout=15)
result={'profile':profile,'prior_thread':thread,'fresh_context':True,'exit_code':rc,'elapsed_s':time.monotonic()-start};(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
