"""One fresh oMLX target-only VLM-route D512 C1/C8 qualification trial, never formal timing.

prepare() reads small frozen inputs only; run(spec,out,checkpoint) explicitly
owns one direct server. The outer owner supplies settled resource/service probes,
full Python/runtime/model pins and restoration. No service control occurs
here. Imported/help paths never import a model or start a process/listener.
"""
import argparse
import asyncio
import base64
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

HERE=Path(__file__).resolve().parent
BASE_PLAN=HERE.parent/'pareto_realistic/prereg-qualification-v1/plan.json'
BASE_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
FIXED={
    '../pareto_realistic/native_cleanup.py':'6c292a9d3cb28c047d1dc92ac0fec4abe19fa8df84c172ef61d2f6ce41157ca6',
    '../pareto_realistic/client.py':'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
    '../pareto_realistic/http_capture_v2.py':'536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3',
}
LOCAL=('serve_probe_v2.py','join_v2.py')
UVICORN_SOURCE=HERE.parents[2]/'.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/uvicorn/server.py'
UVICORN_SHA='c1be9d133467a42ab88d7168e408ef5a90257661a901899426214a3d1fc2d026'



def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def load(name):
    path=(HERE/name).resolve();need(name in FIXED or name in LOCAL,'fixed small source only')
    if name in FIXED:need(sha(path)==FIXED[name],'frozen trial source changed: '+name)
    package='_omlx_retry_trial_sources'
    if package not in sys.modules:
        module=types.ModuleType(package);module.__path__=[str(HERE)];sys.modules[package]=module
    spec=importlib.util.spec_from_file_location(package+'.'+path.stem,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def pinned(ref):
    need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute() and sha(ref['path'])==ref['sha256'],'artifact pin')
    return Path(ref['path'])


def prepare(out,concurrency,port,key,*,deadlines,environment):
    """Pure fixed launch/spec; no model, runtime-library or model-payload hashing."""
    p=load('serve_probe_v2.py');need(sha(BASE_PLAN)==BASE_SHA,'frozen D workload plan')
    base=read(BASE_PLAN);fixtures=read(pinned(base['fixtures']))['fixtures'];model=read(pinned(base['model_manifest']))
    launch=p.specification(out,concurrency,port,key,observed=True)
    official=launch['command'][launch['command'].index('--')+1:]
    model_dir=next(iter(launch['symlinks'].values()))
    need(model_dir==base['model'] and model['model']==model_dir,'same Qwen3.8 target')
    need(set(deadlines)=={'ready','request','group','close','term','kill','overall','resource_interval'}
         and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in deadlines.values()),'finite explicit budgets')
    need(deadlines['overall']>=deadlines['ready']+deadlines['group'] and deadlines['resource_interval']<=5
         and deadlines['term']==deadlines['kill']==15,'bounded original TERM/KILL15 and probe interval<=5')
    need(type(environment) is dict and all(type(k) is type(v) is str for k,v in environment.items()),'explicit string environment')
    allowed={'PATH','HOME','TMPDIR','LANG','LC_ALL','USER','LOGNAME'}|set(launch['environment'])
    need(set(environment)<=allowed and all(environment.get(k)==v for k,v in launch['environment'].items()),
         'exact oMLX flags; only explicit system environment additions')
    tokenizer={str(Path(model_dir)/name):digest for name,digest in model['files'].items()
        if name in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja')}
    need(str(Path(model_dir)/'tokenizer.json') in tokenizer and str(Path(model_dir)/'tokenizer_config.json') in tokenizer,'pinned tokenizer files')
    sources={str((HERE/name).resolve()):digest for name,digest in FIXED.items()}
    sources.update({str((HERE/name).resolve()):sha(HERE/name) for name in LOCAL})
    sources.update({str(p.ROOT/name):digest for name,digest in p.PINS.items()})
    sources[str(Path(__file__).resolve())]=sha(__file__)
    sources[str(UVICORN_SOURCE)]=UVICORN_SHA
    for path,digest in {**sources,**tokenizer}.items():need(sha(path)==digest,'small source/tokenizer changed: '+path)
    model_name='Qwen3.8-27B-nvfp4'
    need(base['fixtures']==dict(path=str(p.FIXTURES),sha256=p.FIXTURES_SHA),'probe full fixture identity')
    return dict(schema='E023-omlx-D512-trial-spec-v1',output_directory=str(Path(out).resolve()),concurrency=concurrency,
        port=port,api_key=key,command=launch['command'],cwd=launch['cwd'],environment=dict(environment),
        original_settings_plan_sha256=launch['original_settings_plan_sha256'],official_args=official,files=launch['files'],symlinks=launch['symlinks'],
        deadlines=dict(deadlines),sources=sources,tokenizer_pins=tokenizer,model_directory=model_dir,served_model=model_name,
        base_plan=dict(path=str(BASE_PLAN),sha256=BASE_SHA),fixtures_ref=base['fixtures'],fixtures=fixtures,
        jobs=p.jobs_for(fixtures,concurrency),observer_enabled=True,
        scope='Observed route/raw/count/text qualification only; outer runtime/weights/resources/restore audit required; seal is not EOF/process close.')


def validate(spec):
    need(spec.get('schema')=='E023-omlx-D512-trial-spec-v1','fixed qualification trial spec')
    expected=prepare(spec['output_directory'],spec['concurrency'],spec['port'],spec['api_key'],
                     deadlines=spec['deadlines'],environment=spec['environment'])
    need(json.dumps(spec,sort_keys=True,allow_nan=False)==json.dumps(expected,sort_keys=True,allow_nan=False),'trial spec changed')


def ready(raw,spec):
    if raw.get('status')!=200:return False
    data=json.loads(base64.b64decode(raw['body_base64'],validate=True))
    need(data.get('object')=='list' and type(data.get('data')) is list and len(data['data'])==1,
         'one official model-list entry')
    row=data['data'][0]
    need(row.get('id')==spec['served_model'] and row.get('object')=='model' and row.get('owned_by')=='omlx',
         'actual official visible model; loaded engine is checked separately')
    return True


def materialize(spec,out):
    """Only registered new-directory settings and one local target symlink."""
    out=Path(out).resolve(); records={}
    expected={str(out/'state/settings.json'),str(out/'state/model_settings.json')}
    need(set(spec['files'])==expected and set(spec['symlinks'])=={str(out/'models'/spec['served_model'])},
         'fixed local materialization paths')
    for name,value in spec['files'].items():
        path=Path(name);path.parent.mkdir(parents=True,exist_ok=True);need(not path.exists(),'settings path already exists')
        write(path,value);records[name]=dict(sha256=sha(path),bytes=path.stat().st_size)
    for name,target in spec['symlinks'].items():
        path=Path(name);need(str(Path(target).resolve())==spec['model_directory'],'fixed target symlink')
        path.parent.mkdir(parents=True,exist_ok=True);need(not path.exists() and not path.is_symlink(),'model path already exists')
        path.symlink_to(target);records[name]=dict(symlink_target=target,resolved=str(path.resolve()))
    write(out/'materialized-inputs.json',dict(files=records,scope='Exact initial configuration; official may persist expanded settings.'))


def request_seal(path):
    """Read a bounded tail, never infer closure from expected request count."""
    path=Path(path)
    if not path.is_file():return None
    with path.open('rb') as stream:
        size=stream.seek(0,2);stream.seek(max(0,size-32768));raw=stream.read()
    rows=raw.splitlines(keepends=True)
    if size>32768:rows=rows[1:]  # first line can be a partial large batch.
    seal=None
    for line in rows:
        if not line.endswith(b'\n'):continue
        event=json.loads(line)
        if event.get('event')=='seal':seal=event
        elif seal is not None and event.get('event')!='close':raise ValueError('observation after seal')
    return seal


class System:
    def __init__(self):self.base=load('../pareto_realistic/http_capture_v2.py').System()
    def __getattr__(self,name):return getattr(self.base,name)
    def spawn(self,command,environment,stdout,stderr,*,cwd):
        return subprocess.Popen(command,cwd=cwd,env=environment,stdout=stdout,stderr=stderr)
    async def readiness(self,spec):
        import httpx
        started=time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=2,trust_env=False,follow_redirects=False) as client:
                response=await client.get(f"http://127.0.0.1:{spec['port']}/v1/models",headers={'Authorization':'Bearer '+spec['api_key']})
                return dict(start_monotonic=started,end_monotonic=time.monotonic(),status=response.status_code,
                    headers=dict(response.headers),body_base64=base64.b64encode(response.content).decode())
        except (httpx.TransportError,OSError) as exc:
            return dict(start_monotonic=started,end_monotonic=time.monotonic(),error=dict(type=type(exc).__name__,message=str(exc)))
    def decoder(self,spec):
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(Path(spec['model_directory'])/'tokenizer.json'))
        return lambda ids:tokenizer.decode(ids,skip_special_tokens=False)
    async def seal(self,path):return request_seal(path)
    async def group(self,spec,output):
        client=load('../pareto_realistic/native_cleanup.py');d=spec['deadlines']
        return await client.run_group(f"http://127.0.0.1:{spec['port']}",spec['jobs'],spec['concurrency'],phase='qualification',
            qualification=None,headers={'Authorization':'Bearer '+spec['api_key']},request_timeout_s=d['request'],
            group_timeout_s=d['group'],cleanup_timeout_s=d['close'],output_dir=output)


async def run(spec,out,checkpoint,*,backend=None):
    """One direct process; checkpoint must settle any probe before propagating cancellation."""
    need(callable(checkpoint),'external settled resource/service checkpoint required');validate(spec)
    out=Path(out).resolve();need(str(out)==spec['output_directory'],'registered output directory')
    spec=copy.deepcopy(spec);out.mkdir(parents=True,exist_ok=False)
    cap=load('../pareto_realistic/http_capture_v2.py');system=backend or System();d=spec['deadlines'];port=spec['port']
    result=dict(schema='E023-omlx-D512-trial-result-v1',status='INCOMPLETE',concurrency=spec['concurrency'],
        route_count_text_qualified=False,performance_qualified=False,full_state_qualified=False,server_reaped=False,
        cleanup_complete=False,spawn_attempted=False,popen_returned=False,errors=[],started_at=time.time())
    write(out/'spec.json',spec);write(out/'launch.json',dict(command=spec['command'],cwd=spec['cwd'],environment=spec['environment']))
    write(out/'complete.json',result);child=task=None;term_sent=False;stdout=stderr=None;phase='preflight'
    events=(out/'events.jsonl').open('x')
    def event(label,**fields):
        events.write(json.dumps(dict(label=label,time=time.time(),monotonic=time.monotonic(),**fields))+'\n');events.flush()
    def error(exc,where):result['errors'].append(dict(phase=where,type=type(exc).__name__,message=str(exc)))
    async def check(label,pid):
        value=checkpoint(label,pid)
        if inspect.isawaitable(value):await value
        event('checkpoint',phase=label,pid=pid)
    async def group():
        try:
            value=await system.group(spec,out/'client');write(out/'group.json',value);return value
        except BaseException as exc:
            if hasattr(exc,'result'):write(out/'group.json',exc.result)
            raise
    try:
        async with asyncio.timeout(d['overall']):
            await check('pre-spawn',None)
            materialize(spec,out)
            raw=await system.port(port);write(out/'port-before.json',raw);need(cap.no_process(raw),'port occupied/unknown')
            stdout=(out/'server.stdout').open('xb');stderr=(out/'server.stderr').open('xb');phase='spawn'
            write(out/'spawn-intent.json',dict(command=spec['command'],cwd=spec['cwd'],time=time.time()))
            result['spawn_attempted']=True
            child=system.spawn(spec['command'],spec['environment'],stdout,stderr,cwd=spec['cwd']);result['popen_returned']=True
            write(out/'child.json',dict(pid=child.pid,time=time.time()));await check('spawned',child.pid)
            identity=await system.identity(child.pid);write(out/'child-identity.json',identity)
            need(identity['returncode']==0 and identity['stdout'].split()[:2]==[str(child.pid),str(os.getpid())],'direct server identity')
            phase='readiness'
            async with asyncio.timeout(d['ready']):
                sampled=time.monotonic()
                while True:
                    need(child.poll() is None,'server exited before ready')
                    raw=await system.readiness(spec);event('readiness',raw=raw)
                    if ready(raw,spec):
                        listener=await system.port(port,child.pid);write(out/'listener.json',listener)
                        need(cap.owned_listener(listener,child.pid,port),'owned localhost listener');write(out/'ready.json',raw);break
                    if time.monotonic()-sampled>=d['resource_interval']:
                        await check('starting',child.pid);sampled=time.monotonic()
                    await asyncio.sleep(.1)
            await check('ready',child.pid);phase='qualification';task=asyncio.create_task(group(),name='omlx-qualification')
            while not task.done():
                await asyncio.wait([task],timeout=d['resource_interval'])
                need(child.poll() is None,'server exited during requests');await check('qualification',child.pid)
            task.result();await check('after-qualification',child.pid)
            phase='request-seal'
            async with asyncio.timeout(d['close']):
                while True:
                    need(child.poll() is None,'server exited before request seal')
                    seal=await system.seal(out/'probe/observer.jsonl')
                    if seal is not None:write(out/'request-seal.json',seal);break
                    await asyncio.sleep(.05)
            raw=await system.readiness(spec);write(out/'ready-after.json',raw);need(ready(raw,spec),'server identity changed')
    except BaseException as exc:error(exc,phase)
    finally:
        async def finish_owned():
            nonlocal term_sent
            try:
                if task is not None and not task.done():task.cancel();await asyncio.wait([task],timeout=d['close']+1)
                if task is not None and not task.done():task.cancel();error(TimeoutError('client unresolved'),'client-cleanup')
            except BaseException as exc:error(exc,'client-cleanup')
            finally:
                if child is not None:
                    try:
                        if child.poll() is None:
                            event('server-term',pid=child.pid)
                            try:
                                child.terminate();term_sent=True;await system.wait(child,d['term'])
                            except subprocess.TimeoutExpired:event('server-term-timeout',pid=child.pid)
                            except BaseException as exc:error(exc,'server-term')
                            if child.poll() is None:event('server-kill',pid=child.pid);child.kill()
                        await system.wait(child,d['kill'])
                        absent=await system.absent(child.pid);write(out/'child-absence.json',absent)
                        port_after=await system.port(port);write(out/'port-after.json',port_after)
                        need(type(child.returncode) is int and cap.no_process(absent) and cap.no_process(port_after),'strict server/port absence')
                        write(out/'release.json',dict(pid=child.pid,returncode=child.returncode,waited_reaped=True,
                            strict_absence=True,port_absent=True,term_sent=term_sent,time=time.time()));result['server_reaped']=True
                        result['cleanup_complete']=task is None or (task.done() and (out/'group.json').is_file()
                            and read(out/'group.json')['cleanup']['complete'] is True)
                    except BaseException as exc:error(exc,'server-cleanup')
        cleaner=asyncio.create_task(finish_owned(),name='omlx-owned-cleanup')
        end=asyncio.get_running_loop().time()+d['close']+d['term']+d['kill']+12
        while not cleaner.done() and asyncio.get_running_loop().time()<end:
            try:await asyncio.wait([cleaner],timeout=max(0,end-asyncio.get_running_loop().time()))
            except asyncio.CancelledError as exc:error(exc,'cancel-during-cleanup')
        if not cleaner.done():cleaner.cancel();result['cleanup_complete']=False;error(TimeoutError('owned cleanup deadline'),'cleanup')
        elif cleaner.cancelled():result['cleanup_complete']=False;error(RuntimeError('cleanup task cancelled'),'cleanup')
        elif cleaner.exception() is not None:result['cleanup_complete']=False;error(cleaner.exception(),'cleanup')
        result['cleanup_task_resolved']=cleaner.done();result['client_task_resolved']=task is None or task.done()
        try:
            await check('after-reap' if result['server_reaped'] else 'cleanup-incomplete',None if result['server_reaped'] or child is None else child.pid)
            validate(spec);result['small_pins_rechecked']=True
            if result['server_reaped'] and result['cleanup_complete'] and not result['errors']:
                need(child.returncode==0 or (child.returncode==-15 and term_sent),
                     'qualification accepts normal exit or own explained SIGTERM only')
                join=load('join_v2.py');decode_text=system.decoder(spec)
                with (out/'probe/observer.jsonl').open('rb') as lines:
                    checked=join.join(read(out/'group.json'),lines,spec['jobs'],{f['fixture_id']:f for f in spec['fixtures']},
                        decode_text,spec['sources'][str(HERE/'serve_probe_v2.py')],
                        expected_official_args=spec['official_args'],expected_model_path=spec['model_directory'],expected_pid=child.pid)
                write(out/'raw-join.json',checked)
                need(read(out/'request-seal.json')==checked['seal'],'observed pre-TERM seal matches closed ledger')
                result['route_count_text_qualified']=True
                result['request_sealed']=True;result['observer_closed']=checked['observer_closed']
        except BaseException as exc:error(exc,'postflight')
        if stdout is not None:stdout.close()
        if stderr is not None:stderr.close()
        events.close();result['ended_at']=time.time()
        result['status']='PASS' if result['route_count_text_qualified'] and not result['errors'] else 'FAIL'
        result['artifacts']={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size)
            for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'complete.json'}
        write(out/'complete.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--spec-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    spec=read(pinned(dict(path=str(a.spec.resolve()),sha256=a.spec_sha256)));validate(spec)
    a.out.mkdir(parents=True,exist_ok=False);write(a.out/'prepared.json',dict(status='PREPARED',spec_sha256=a.spec_sha256,
        scope='CPU spec check only; actual trial requires external resource/service checkpoint.'));return 0


if __name__=='__main__':raise SystemExit(main())
