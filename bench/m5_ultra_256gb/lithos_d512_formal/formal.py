"""One fresh observer-OFF Lithos D512 trial: eight warm then 64 formal jobs.

Only explicit run() owns a direct official server. The outer owner supplies
settled resource/service checkpoints, full dependency pins and restoration.
No model/tokenizer/server imports occur at module load or --help. Count/log
eligibility remains separate from independent resource/performance approval.
"""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
FIXED={
    '../lithos_d512_retry/trial.py':'32d1bdca8236c70129ba5113bfadbad377c8c925e40a925bafd027e8ecff5ebe',
    '../lithos_d512_retry/serve_probe.py':'c6f4915a83d15cf5c527d71e73d4e54683c0de3881de4c2352b9e0d0abc14703',
    '../lithos_d512_retry/join.py':'2d4dfa12c9672a7a9deff7a962ea18dd671f53c4e7010eed3589fa4a0826cd0d',
    '../pareto_realistic/lithos_client.py':'13a629f096434bc6c422803bf7bfd8992e24396cb122880b493e893cc5ecd2e9',
    '../pareto_realistic/client.py':'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
    '../pareto_realistic/http_capture_v2.py':'536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3',
    'contract.py':'2c5b221608e8c53b39116b2e13b247b9fb78d83c74cbee069179efbea9210b32',
}
MAX_LOG_BYTES=16 << 20


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1 << 20):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def write(path,value):
    """Durable small evidence, including phase boundaries before creating POST tasks."""
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def load(name):
    path=(HERE/name).resolve();need(sha(path)==FIXED[name],'frozen formal source changed: '+name)
    spec=importlib.util.spec_from_file_location('_lithos_formal_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source_pins():
    return {**{str((HERE/name).resolve()):value for name,value in FIXED.items()},str(Path(__file__).resolve()):sha(__file__)}


def prepare(out,concurrency,repeat,port,key,*,qualification,deadlines,environment):
    trial=load('../lithos_d512_retry/trial.py')
    need(type(concurrency) is int and concurrency in (1,8) and type(repeat) is int and 1<=repeat<=3,'registered C1/C8 repeat1..3')
    need(set(deadlines)=={'ready','request','warm','formal','close','term','kill','overall','resource_interval'}
         and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in deadlines.values()),'finite explicit formal budgets')
    need(deadlines['overall']>=sum(deadlines[k] for k in ('ready','warm','formal'))
         and deadlines['overall']<=3600 and deadlines['term']==deadlines['kill']==15
         and deadlines['close']<=15 and deadlines['resource_interval']<=5,'original absolute/cleanup/resource limits')
    inherited={k:v for k,v in deadlines.items() if k not in ('warm','formal')};inherited['group']=deadlines['formal']
    seed=trial.prepare(out,concurrency,port,key,deadlines=inherited,environment=environment)
    probe=load('../lithos_d512_retry/serve_probe.py')
    launch=probe.specification(out,concurrency,port,key,observed=False)
    need(launch['command'][1:4]==['-B','-m','monolith.serve'] and launch['observer_enabled'] is False,'original observer-OFF official CLI')
    data=load('contract.py').prepare(seed['base_plan'],qualification,concurrency,repeat,seed['served_model'])
    need(data['fixtures']==seed['fixtures'] and data['model_directory']==seed['model_directory']
         and data['served_model']==seed['served_model'] and data['pack_dir']==seed['pack_dir'],'qualified original fixtures/model/pack')
    sources={**seed['sources'],**data['sources'],**source_pins()}
    for path,digest in sources.items():need(sha(path)==digest,'small formal dependency changed: '+path)
    return dict(**{k:v for k,v in data.items() if k not in ('sources','concurrency','repeat')},schema='E023-lithos-D512-formal-spec-v1',
        output_directory=str(Path(out).resolve()),concurrency=concurrency,repeat=repeat,port=port,api_key=key,
        command=launch['command'],cwd=launch['cwd'],environment=dict(environment),deadlines=dict(deadlines),
        qualification_binding=copy.deepcopy(qualification),observer_enabled=False,official_args=seed['official_args'],
        original_launch_plan_sha256=launch['original_launch_plan_sha256'],sources=sources,tokenizer_pins=seed['tokenizer_pins'],
        log_contract=dict(max_bytes=MAX_LOG_BYTES,logger='__main__',warm=dict(count=8,cache_zero=True,setup_zero=False),
                          formal=dict(count=64,cache_zero=True,setup_zero=True),per_wire_request_id_bound=False),
        scope='Observed qualified route/count semantics, new observer-OFF finite workload. Outer runtime/resources/reap/restoration audit required.')


def validate(spec):
    need(spec.get('schema')=='E023-lithos-D512-formal-spec-v1','formal spec schema')
    expected=prepare(spec['output_directory'],spec['concurrency'],spec['repeat'],spec['port'],spec['api_key'],
        qualification=spec['qualification_binding'],deadlines=spec['deadlines'],environment=spec['environment'])
    need(json.dumps(spec,sort_keys=True,allow_nan=False)==json.dumps(expected,sort_keys=True,allow_nan=False),'formal spec changed')


class LogWindows:
    """Same open stderr identity, immutable prefixes, exact quiet phase windows."""
    def __init__(self,out,stream,parser):
        self.out,self.stream,self.parser=Path(out),stream,parser
        self.boundaries={};self.windows={};self.identity=None
    def boundary(self,label):
        self.stream.flush();opened=os.fstat(self.stream.fileno());path=self.out/'server.stderr'
        current=path.stat();identity=(opened.st_dev,opened.st_ino)
        need((current.st_dev,current.st_ino)==identity,'stderr rotated/replaced')
        need(self.identity is None or self.identity==identity,'stderr identity changed')
        need(opened.st_size==current.st_size and 0<=current.st_size<=MAX_LOG_BYTES,'stderr size/race/bound')
        with path.open('rb') as source:
            st=os.fstat(source.fileno());raw=source.read(MAX_LOG_BYTES+1)
            need((st.st_dev,st.st_ino)==identity and len(raw)==current.st_size,'stderr changed during boundary')
        need(not raw or raw.endswith(b'\n'),'stderr boundary splits a line')
        for previous in self.boundaries.values():
            need(len(raw)>=previous['offset'] and hashlib.sha256(raw[:previous['offset']]).hexdigest()==previous['prefix_sha256'],
                 'stderr prefix truncated or rewritten')
        row=dict(label=label,path='server.stderr',device=identity[0],inode=identity[1],offset=len(raw),
            prefix_sha256=hashlib.sha256(raw).hexdigest(),wall_s=time.time(),monotonic_s=time.monotonic(),perf_counter_s=time.perf_counter())
        self.identity=identity;self.boundaries[label]=row;write(self.out/(label+'-log-boundary.json'),row)
        return row,raw
    def start(self,phase):self.boundary(phase+'-start')
    def close(self,phase,group):
        cleanup=group['cleanup']
        need(cleanup['complete'] is True and not cleanup['pending_tasks'] and not cleanup['errors'], 'group not drained before log boundary')
        end,raw=self.boundary(phase+'-end');start=self.boundaries[phase+'-start']
        rows=group['requests'];need(rows and start['perf_counter_s']<=min(r['start_perf_counter_s'] for r in rows)
            and end['perf_counter_s']>=max(r['eof_perf_counter_s'] for r in rows)
            and end['perf_counter_s']>=cleanup['end_perf_counter_s'],'log boundaries do not bracket POST/DONE/drain')
        window=raw[start['offset']:end['offset']];name=phase+'-generation.log'
        with (self.out/name).open('xb') as stream:stream.write(window);stream.flush();os.fsync(stream.fileno())
        checked=self.parser(window,8 if phase=='warm' else 64,require_setup_zero=phase=='formal',expected_logger='__main__')
        receipt=dict(schema='E023-lithos-formal-log-window-v1',phase=phase,start=start,end=end,
            artifact=dict(path=name,sha256=hashlib.sha256(window).hexdigest(),bytes=len(window)),check=checked)
        write(self.out/(phase+'-log-check.json'),receipt);self.windows[phase]=receipt
        return receipt
    def finalize(self,*,complete):
        end,raw=self.boundary('after-reap')
        for phase,window in self.windows.items():
            data=raw[window['start']['offset']:window['end']['offset']]
            need(data==(self.out/window['artifact']['path']).read_bytes()
                 and hashlib.sha256(data).hexdigest()==window['artifact']['sha256'],'retained log window changed')
        result=dict(status='PASS',complete_windows=list(self.windows),final=end,all_prefixes_unchanged=True)
        if complete:
            need(list(self.windows)==['warm','formal'],'both quiet windows required')
            result['all_generations']=self.parser(raw,72,require_setup_zero=False,expected_logger='__main__')
        write(self.out/'log-final-check.json',result);return result


class System:
    def __init__(self):self.base=load('../lithos_d512_retry/trial.py').System()
    def __getattr__(self,name):return getattr(self.base,name)
    async def group(self,spec,phase,output):
        client=load('../pareto_realistic/lithos_client.py');contract=load('contract.py');d=spec['deadlines']
        return await client.run_group(f"http://127.0.0.1:{spec['port']}",spec['jobs'][phase],spec['concurrency'],phase=phase,
            qualification=contract.count_contract(spec,phase),headers={'Authorization':'Bearer '+spec['api_key']},
            request_timeout_s=d['request'],group_timeout_s=d[phase],cleanup_timeout_s=d['close'],output_dir=output)


async def run(spec,out,checkpoint,backend=None):
    need(callable(checkpoint),'external settled resource/service checkpoint required');validate(spec)
    out=Path(out).resolve();need(str(out)==spec['output_directory'],'registered formal output directory')
    spec=copy.deepcopy(spec);out.mkdir(parents=True,exist_ok=False)
    cap=load('../pareto_realistic/http_capture_v2.py');trial=load('../lithos_d512_retry/trial.py')
    contract=load('contract.py');join=load('../lithos_d512_retry/join.py')
    system=backend or System();d=spec['deadlines'];port=spec['port']
    result=dict(schema='E023-lithos-D512-formal-result-v1',status='INCOMPLETE',concurrency=spec['concurrency'],repeat=spec['repeat'],
        count_eligible=False,performance_qualified=False,full_state_qualified=False,server_reaped=False,cleanup_complete=False,
        spawn_attempted=False,popen_returned=False,errors=[],groups={},log_windows={},started_at=time.time())
    write(out/'spec.json',spec);write(out/'launch.json',dict(command=spec['command'],cwd=spec['cwd'],environment=spec['environment']))
    write(out/'complete.json',result);child=task=None;stdout=stderr=None;logs=None;phase='preflight';active_group=None;checked={}
    events=(out/'events.jsonl').open('x')
    def event(label,**fields):
        events.write(json.dumps(dict(label=label,time=time.time(),monotonic=time.monotonic(),**fields))+'\n');events.flush()
    def error(exc,where):result['errors'].append(dict(phase=where,type=type(exc).__name__,message=str(exc)))
    def ref(path):return dict(path=str(path.relative_to(out)),sha256=sha(path))
    async def check(label,pid):
        value=checkpoint(label,pid)
        if inspect.isawaitable(value):await value
        event('checkpoint',phase=label,pid=pid)
    async def group(name):
        try:
            value=await system.group(spec,name,out/name);write(out/(name+'-group.json'),value);return value
        except BaseException as exc:
            if hasattr(exc,'result'):write(out/(name+'-group.json'),exc.result)
            raise
    try:
        async with asyncio.timeout(d['overall']):
            await check('pre-spawn',None)
            raw=await system.port(port);write(out/'port-before.json',raw);need(cap.no_process(raw),'formal port occupied/unknown')
            need(spec['observer_enabled'] is False and spec['command'][1:4]==['-B','-m','monolith.serve'],'official observer OFF required')
            stdout=(out/'server.stdout').open('xb');stderr=(out/'server.stderr').open('xb');phase='spawn'
            logs=LogWindows(out,stderr,join.generation_log_window)
            write(out/'spawn-intent.json',dict(command=spec['command'],cwd=spec['cwd'],time=time.time()));result['spawn_attempted']=True
            child=system.spawn(spec['command'],spec['environment'],stdout,stderr,cwd=spec['cwd']);result['popen_returned']=True
            write(out/'child.json',dict(pid=child.pid,time=time.time()));await check('spawned',child.pid)
            identity=await system.identity(child.pid);write(out/'child-identity.json',identity)
            need(identity['returncode']==0 and identity['stdout'].split()[:2]==[str(child.pid),str(os.getpid())],'direct formal server identity')
            phase='readiness'
            async with asyncio.timeout(d['ready']):
                sampled=time.monotonic()
                while True:
                    need(child.poll() is None,'formal server exited before ready')
                    raw=await system.readiness(spec);event('readiness',raw=raw)
                    if trial.ready(raw,spec):
                        listener=await system.port(port,child.pid);write(out/'listener.json',listener)
                        need(cap.owned_listener(listener,child.pid,port),'owned formal localhost listener');write(out/'ready.json',raw);break
                    if time.monotonic()-sampled>=d['resource_interval']:
                        await check('starting',child.pid);sampled=time.monotonic()
                    await asyncio.sleep(.1)
            await check('ready',child.pid)
            for phase in ('warm','formal'):
                active_group=phase;logs.start(phase)
                task=asyncio.create_task(group(phase),name='lithos-'+phase)
                while not task.done():
                    await asyncio.wait([task],timeout=d['resource_interval'])
                    need(child.poll() is None,'formal server exited during requests');await check(phase,child.pid)
                value=task.result();checked[phase]=contract.check_group(value,spec,phase)
                need(checked[phase]['status']=='PASS' and checked[phase]['count_eligible'] is True,'raw count/transport check failed')
                write(out/(phase+'-raw-check.json'),checked[phase]);logs.close(phase,value)
                result['groups'][phase]=ref(out/(phase+'-group.json'));result['log_windows'][phase]=ref(out/(phase+'-log-check.json'))
                await check('after-'+phase,child.pid)
                raw=await system.readiness(spec);write(out/('ready-after-'+phase+'.json'),raw);need(trial.ready(raw,spec),'formal model identity changed')
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
                        after=await system.port(port);write(out/'port-after.json',after)
                        need(type(child.returncode) is int and cap.no_process(absent) and cap.no_process(after),'strict server/port absence')
                        write(out/'release.json',dict(pid=child.pid,returncode=child.returncode,waited_reaped=True,
                            strict_absence=True,port_absent=True,time=time.time()));result['server_reaped']=True
                        path=out/(str(active_group)+'-group.json')
                        result['cleanup_complete']=task is None or (task.done() and path.is_file() and read(path)['cleanup']['complete'] is True)
                    except BaseException as exc:error(exc,'server-cleanup')
        cleaner=asyncio.create_task(finish_owned(),name='lithos-formal-owned-cleanup')
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
            if logs is not None:
                logs.finalize(complete=not result['errors'] and list(checked)==['warm','formal']);result['logs_rechecked']=True
            if result['server_reaped'] and result['cleanup_complete'] and not result['errors'] and list(checked)==['warm','formal']:
                result.update(count_eligible=True,point=checked['formal']['point'])
                write(out/'point.json',dict(**result['point'],count_eligible=True,performance_qualified=False))
        except BaseException as exc:error(exc,'postflight')
        if stdout is not None:stdout.close()
        if stderr is not None:stderr.close()
        events.close();result['ended_at']=time.time()
        result['status']='PASS' if result['count_eligible'] and not result['errors'] else 'FAIL'
        result['artifacts']={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size)
            for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'complete.json'}
        write(out/'complete.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--spec-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    need(sha(a.spec)==a.spec_sha256,'formal spec pin');spec=read(a.spec);validate(spec)
    a.out.mkdir(parents=True,exist_ok=False);write(a.out/'prepared.json',dict(status='PREPARED',spec_sha256=a.spec_sha256,
        scope='CPU spec check only. Actual run requires the external resource/service owner.'));return 0


if __name__=='__main__':raise SystemExit(main())
