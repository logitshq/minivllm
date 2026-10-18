"""One queued offline CPU check after the current formal experiment closes.

No service/model actions. The reviewed child independently checks actual
restoration, audit and archive receipts before loading any fake tests.
"""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb'
OUT=A/'gdn-http-prepared-cpu-v1-queue';RESULT=A/'gdn-http-prepared-cpu-v1'
CLOSURE=A/'route-http-formal-v2-closure/status.json'
SCRIPT=Path('/tmp/run-gdn-http-prepared-cpu-v1.py')
DIGEST='22e9ef7a46e1a2ea76940abc8e5152088e124d5f060f02c2ee63c1eeb51b35cc'
PY=R/'.artifacts/omlx-baseline/.venv/bin/python'
def need(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def save(value):
    target=OUT/'status.json';temporary=OUT/'status.next.json'
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(target)
def main():
    need(sys.argv[1:]==['--run'],'explicit --run required')
    need(not OUT.exists() and not RESULT.exists(),'fresh queue and actual test output only')
    need(sha(SCRIPT)==DIGEST,'reviewed CPU runner changed')
    OUT.mkdir();(OUT/'runner.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'cpu-runner.py').write_bytes(SCRIPT.read_bytes())
    value=dict(status='WAITING_FOR_CLOSED_FORMAL',started_at=time.time(),service_actions=False,
        cpu_tests_started=False,cpu_tests_completed=False,performance_qualified=False)
    save(value);deadline=time.monotonic()+43200
    try:
        while True:
            need(time.monotonic()<deadline,'formal closure did not finish within 12 hours')
            try:closure=read(CLOSURE)
            except (FileNotFoundError,json.JSONDecodeError):closure={}
            if closure.get('ended_at'):
                need(closure.get('status') in ('PASS','EXPERIMENT_FAILED_EVIDENCE_RETAINED'),
                     'formal closure failed; CPU runner not started')
                break
            time.sleep(30)
        need(sha(SCRIPT)==DIGEST,'reviewed CPU runner changed while waiting')
        command=[str(PY),'-I','-B',str(SCRIPT),'--run','--out',str(RESULT)]
        value.update(status='RUNNING_OFFLINE_CPU',cpu_tests_started=True,command=command,
                     source_sha256=DIGEST,closure_sha256=sha(CLOSURE),child_started_at=time.time());save(value)
        with (OUT/'cpu.stdout').open('xb') as stdout,(OUT/'cpu.stderr').open('xb') as stderr:
            child=subprocess.run(command,cwd=R,stdout=stdout,stderr=stderr,check=False,timeout=1800)
        value.update(returncode=child.returncode,child_ended_at=time.time())
        result=read(RESULT/'result.json')
        value.update(cpu_tests_completed=True,test_status=result['status'],tests=result['tests'],
                     result_sha256=sha(RESULT/'result.json'))
        need(child.returncode==0 and result['status']=='PASS' and result['tests']==16,
             'CPU check did not pass; preserve complete result for review')
        need((RESULT/'artifact-manifest.json').is_file(),'completed CPU inventory required')
        value['status']='PASS'
    except BaseException as error:
        value.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        value['ended_at']=time.time();save(value)
    print(json.dumps(value),flush=True)
    return int(value['status']!='PASS')
if __name__=='__main__':raise SystemExit(main())
