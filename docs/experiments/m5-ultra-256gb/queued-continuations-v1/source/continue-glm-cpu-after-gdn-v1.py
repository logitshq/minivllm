"""One queued GLM fake-only CPU check after the actual GDN CPU process exits.

The reviewed child independently checks current formal restoration/archive and
the previous CPU parent's wait result. No GLM model/service operation occurs.
"""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb'
OUT=A/'glm-cycle-prepared-cpu-v1-queue';RESULT=A/'glm-cycle-prepared-cpu-v1'
PREVIOUS=A/'gdn-http-prepared-cpu-v1-queue/status.json'
SCRIPT=Path('/tmp/run-glm-cycle-prepared-cpu-v1.py')
DIGEST='1c03870860aaf1670c8f687ab26d04b72daf454c3be05ef4562b9f4f32046489'
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
    need(sha(SCRIPT)==DIGEST,'reviewed GLM CPU runner changed')
    OUT.mkdir();(OUT/'runner.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'cpu-runner.py').write_bytes(SCRIPT.read_bytes())
    value=dict(status='WAITING_FOR_GDN_CPU',started_at=time.time(),service_actions=False,
        cpu_tests_started=False,cpu_tests_completed=False,performance_qualified=False)
    save(value);deadline=time.monotonic()+43200
    try:
        while True:
            need(time.monotonic()<deadline,'previous CPU queue did not close within 12 hours')
            try:previous=read(PREVIOUS)
            except (FileNotFoundError,json.JSONDecodeError):previous={}
            if previous.get('ended_at'):
                need(previous.get('status') in ('PASS','FAIL') and previous.get('cpu_tests_completed') is True
                     and type(previous.get('returncode')) is int and previous.get('child_ended_at'),
                     'previous CPU process did not return a terminal result; GLM checks not started')
                break
            time.sleep(30)
        need(sha(SCRIPT)==DIGEST,'reviewed GLM CPU runner changed while waiting')
        raw=PREVIOUS.read_bytes();(OUT/'previous-cpu-queue.json').write_bytes(raw)
        command=[str(PY),'-I','-B',str(SCRIPT),'--run','--out',str(RESULT)]
        value.update(status='RUNNING_OFFLINE_CPU',cpu_tests_started=True,command=command,
                     source_sha256=DIGEST,previous_queue_sha256=hashlib.sha256(raw).hexdigest(),child_started_at=time.time());save(value)
        with (OUT/'cpu.stdout').open('xb') as stdout,(OUT/'cpu.stderr').open('xb') as stderr:
            child=subprocess.run(command,cwd=R,stdout=stdout,stderr=stderr,check=False,timeout=1800)
        value.update(returncode=child.returncode,child_ended_at=time.time())
        result=read(RESULT/'result.json')
        value.update(cpu_tests_completed=True,test_status=result['status'],tests=result['tests'],
                     result_sha256=sha(RESULT/'result.json'))
        need(child.returncode==0 and result['status']=='PASS' and result['tests']==33,
             'GLM CPU checks did not pass; preserve complete result for review')
        need((RESULT/'artifact-manifest.json').is_file(),'completed GLM CPU inventory required')
        value['status']='PASS'
    except BaseException as error:
        value.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        value['ended_at']=time.time();save(value)
    print(json.dumps(value),flush=True)
    return int(value['status']!='PASS')
if __name__=='__main__':raise SystemExit(main())
