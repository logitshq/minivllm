"""One fresh Lithos target-only D512 C1/C8 qualification trial, never formal timing.

prepare() reads small frozen inputs only; run(spec,out,checkpoint) explicitly
owns one direct server. The outer owner supplies settled resource/service probes,
full Python/runtime/model/pack pins and restoration. No service control occurs
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
    'serve_probe.py':'c6f4915a83d15cf5c527d71e73d4e54683c0de3881de4c2352b9e0d0abc14703',
    'join.py':'2d4dfa12c9672a7a9deff7a962ea18dd671f53c4e7010eed3589fa4a0826cd0d',
    '../pareto_realistic/lithos_client.py':'13a629f096434bc6c422803bf7bfd8992e24396cb122880b493e893cc5ecd2e9',
    '../pareto_realistic/client.py':'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
    '../pareto_realistic/http_capture_v2.py':'536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3',
}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def load(name):
    path=(HERE/name).resolve();need(sha(path)==FIXED[name],'frozen trial source changed: '+name)
    package='_lithos_retry_trial_sources'
    if package not in sys.modules:
        module=types.ModuleType(package);module.__path__=[str(HERE)];sys.modules[package]=module
    spec=importlib.util.spec_from_file_location(package+'.'+path.stem,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def pinned(ref):
    need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute() and sha(ref['path'])==ref['sha256'],'artifact pin')
    return Path(ref['path'])


def prepare(out,concurrency,port,key,*,deadlines,environment):
    """Pure fixed launch/spec; no model, runtime-library or pack-payload hashing."""
    p=load('serve_probe.py');need(sha(BASE_PLAN)==BASE_SHA,'frozen D workload plan')
    base=read(BASE_PLAN);fixtures=read(pinned(base['fixtures']))['fixtures'];model=read(pinned(base['model_manifest']))
    launch=p.specification(out,concurrency,port,key,observed=True)
    official=launch['command'][launch['command'].index('--')+1:]
    model_dir=official[official.index('--model')+1];pack=official[official.index('--pack')+1]
    need(model_dir==base['model'] and model['model']==model_dir,'same Qwen3.8 target')
    need(set(deadlines)=={'ready','request','group','close','term','kill','overall','resource_interval'}
         and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in deadlines.values()),'finite explicit budgets')
    need(deadlines['overall']>=deadlines['ready']+deadlines['group'] and deadlines['resource_interval']<=5
         and deadlines['term']==deadlines['kill']==15,'bounded original TERM/KILL15 and probe interval<=5')
    need(type(environment) is dict and all(type(k) is type(v) is str for k,v in environment.items()),'explicit string environment')
    allowed={'PATH','HOME','TMPDIR','LANG','LC_ALL','USER','LOGNAME'}|set(launch['environment'])
    need(set(environment)<=allowed and all(environment.get(k)==v for k,v in launch['environment'].items()),
         'exact Lithos flags/API key; only explicit system environment additions')
    tokenizer={str(Path(model_dir)/name):digest for name,digest in model['files'].items()
        if name in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja')}
    need(str(Path(model_dir)/'tokenizer.json') in tokenizer and str(Path(model_dir)/'tokenizer_config.json') in tokenizer,'pinned tokenizer files')
    sources={str((HERE/name).resolve()):digest for name,digest in FIXED.items()}
    sources.update({str(p.ROOT/name):digest for name,digest in p.PINS.items()})
    sources[str(Path(__file__).resolve())]=sha(__file__)
    for path,digest in {**sources,**tokenizer}.items():need(sha(path)==digest,'small source/tokenizer changed: '+path)
    model_name=Path(model_dir).name
    return dict(schema='E023-lithos-D512-trial-spec-v1',output_directory=str(Path(out).resolve()),concurrency=concurrency,
        port=port,api_key=key,command=launch['command'],cwd=launch['cwd'],environment=dict(environment),
        original_launch_plan_sha256=launch['original_launch_plan_sha256'],official_args=official,pack_dir=pack,
        deadlines=dict(deadlines),sources=sources,tokenizer_pins=tokenizer,model_directory=model_dir,served_model=model_name,
        base_plan=dict(path=str(BASE_PLAN),sha256=BASE_SHA),fixtures_ref=base['fixtures'],fixtures=fixtures,
        jobs=p.jobs_for(fixtures,concurrency,model_name),observer_enabled=True,
        scope='Observed route/raw/count/text qualification only; outer runtime/weights/pack/resources/restore audit required.')


def validate(spec):
    need(spec.get('schema')=='E023-lithos-D512-trial-spec-v1','fixed qualification trial spec')
    expected=prepare(spec['output_directory'],spec['concurrency'],spec['port'],spec['api_key'],
                     deadlines=spec['deadlines'],environment=spec['environment'])
    need(json.dumps(spec,sort_keys=True,allow_nan=False)==json.dumps(expected,sort_keys=True,allow_nan=False),'trial spec changed')


def ready(raw,spec):
    if raw.get('status')!=200:return False
    data=json.loads(base64.b64decode(raw['body_base64'],validate=True))
    need(set(data)=={'object','data'} and data['object']=='list' and type(data['data']) is list and len(data['data'])==1,'official model-list shape')
    row=data['data'][0]
    need(set(row)=={'id','object','created','owned_by','context_window'} and row['id']==spec['served_model']
         and row['object']=='model' and row['owned_by']=='lithos-metal' and type(row['created']) is int and row['created']>0
         and type(row['context_window']) is int and row['context_window']==2048,'actual official model/capacity')
    return True


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
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(spec['model_directory'],local_files_only=True,trust_remote_code=False)
        return lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    async def group(self,spec,output):
        client=load('../pareto_realistic/lithos_client.py');d=spec['deadlines']
        return await client.run_group(f"http://127.0.0.1:{spec['port']}",spec['jobs'],spec['concurrency'],phase='qualification',
            qualification=None,headers={'Authorization':'Bearer '+spec['api_key']},request_timeout_s=d['request'],
            group_timeout_s=d['group'],cleanup_timeout_s=d['close'],output_dir=output)


async def run(spec,out,checkpoint,*,backend=None):
    """One direct process; checkpoint must settle any probe before propagating cancellation."""
    need(callable(checkpoint),'external settled resource/service checkpoint required');validate(spec)
    out=Path(out).resolve();need(str(out)==spec['output_directory'],'registered output directory')
    spec=copy.deepcopy(spec);out.mkdir(parents=True,exist_ok=False)
    cap=load('../pareto_realistic/http_capture_v2.py');system=backend or System();d=spec['deadlines'];port=spec['port']
    result=dict(schema='E023-lithos-D512-trial-result-v1',status='INCOMPLETE',concurrency=spec['concurrency'],
        route_count_text_qualified=False,performance_qualified=False,full_state_qualified=False,server_reaped=False,
        cleanup_complete=False,spawn_attempted=False,popen_returned=False,errors=[],started_at=time.time())
    write(out/'spec.json',spec);write(out/'launch.json',dict(command=spec['command'],cwd=spec['cwd'],environment=spec['environment']))
    write(out/'complete.json',result);child=task=None;stdout=stderr=None;phase='preflight'
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
            await check('ready',child.pid);phase='qualification';task=asyncio.create_task(group(),name='lithos-qualification')
            while not task.done():
                await asyncio.wait([task],timeout=d['resource_interval'])
                need(child.poll() is None,'server exited during requests');await check('qualification',child.pid)
            task.result();await check('after-qualification',child.pid)
            raw=await system.readiness(spec);write(out/'ready-after.json',raw);need(ready(raw,spec),'server identity changed')
    except BaseException as exc:error(exc,phase)
    finally:
        async def finish_owned():
            try:
                if task is not None and not task.done():task.cancel();await asyncio.wait([task],timeout=d['close']+1)
                if task is not None and not task.done():task.cancel();error(TimeoutError('client unresolved'),'client-cleanup')
            except BaseException as exc:error(exc,'client-cleanup')
            finally:
                if child is not None:
                    try:
                        if child.poll() is None:
                            event('server-term',pid=child.pid)
                            try:child.terminate();await system.wait(child,d['term'])
                            except subprocess.TimeoutExpired:event('server-term-timeout',pid=child.pid)
                            except BaseException as exc:error(exc,'server-term')
                            if child.poll() is None:event('server-kill',pid=child.pid);child.kill()
                        await system.wait(child,d['kill'])
                        absent=await system.absent(child.pid);write(out/'child-absence.json',absent)
                        port_after=await system.port(port);write(out/'port-after.json',port_after)
                        need(type(child.returncode) is int and cap.no_process(absent) and cap.no_process(port_after),'strict server/port absence')
                        write(out/'release.json',dict(pid=child.pid,returncode=child.returncode,waited_reaped=True,
                            strict_absence=True,port_absent=True,time=time.time()));result['server_reaped']=True
                        result['cleanup_complete']=task is None or (task.done() and (out/'group.json').is_file()
                            and read(out/'group.json')['cleanup']['complete'] is True)
                    except BaseException as exc:error(exc,'server-cleanup')
        cleaner=asyncio.create_task(finish_owned(),name='lithos-owned-cleanup')
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
                join=load('join.py');decode_text=system.decoder(spec)
                with (out/'probe/observer.jsonl').open('rb') as lines:
                    checked=join.join(read(out/'group.json'),lines,spec['jobs'],{f['fixture_id']:f for f in spec['fixtures']},
                        decode_text,FIXED['serve_probe.py'],expected_official_args=spec['official_args'],expected_pack_dir=spec['pack_dir'])
                write(out/'raw-join.json',checked)
                log=join.generation_log_window((out/'server.stderr').read_bytes(),len(spec['jobs']),
                                               require_setup_zero=False,expected_logger='monolith.serve')
                write(out/'generation-log-check.json',log);result['route_count_text_qualified']=True
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
