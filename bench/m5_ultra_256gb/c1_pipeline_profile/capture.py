"""Single C1 diagnostic capture, default inert; never a Pareto point.

prepare() binds an existing qualified combined C1 formal specification. run()
requires an external resource/lifecycle checkpoint and owns only its direct
server. Eight warm + eight admission-probe requests, then authenticated drain.
No service control or backend/model import occurs at module import or CLI.
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
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
SERVER=HERE/'http_server.py'
FACTORY=HERE.parent/'gdn_graph_combined_formal/formal.py'
FACTORY_SHA='4b567f8ed6090175f3639aa109584ca26c1426165c03de3167c3ebde74891a89'
MODES=('off','runner','full')
STAGES=('warm','diagnostic')
CLIENT_PHASE={'warm':'warm','diagnostic':'admission_probe'}


def need(ok,message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def same(a,b): return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def write(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)


def load(path,digest,name):
    need(sha(path)==digest,'actual pinned source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def factory(): return load(FACTORY,FACTORY_SHA,'_c1_profile_formal_factory')
def native(): return factory().donor()


def pinned(ref):
    need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute() and sha(ref['path'])==ref['sha256'],'exact absolute input pin')
    return Path(ref['path'])


def prepare(formal_spec,mode,*,output_directory,drain_token,server_sha256,drain_seconds=30,worker_join_seconds=10):
    """Small CPU metadata preparation; full runtime/model/physical admission is external.

All three modes must use the SAME formal_spec, port, deadlines, candidate flags
and jobs. Only mode, fresh output path and owner-generated nonce vary. No source
pin is guessed: the owner supplies the actual diagnostic entry SHA at sealing.
"""
    need(mode in MODES,'explicit OFF/runner/full diagnostic mode')
    need(type(drain_token) is str and len(drain_token)>=32 and drain_token.isascii()
         and all(33<=ord(c)<=126 for c in drain_token),'ASCII nonce with at least32 printable characters')
    need(type(drain_seconds) in (int,float) and math.isfinite(drain_seconds) and 0<drain_seconds<=120
         and type(worker_join_seconds) in (int,float) and math.isfinite(worker_join_seconds)
         and 0<worker_join_seconds<drain_seconds,'bounded drain and worker-join budgets')
    f=factory(); base=read(pinned(formal_spec)); f.validate(base)
    need(base['concurrency']==1 and base['identity']['combined_candidates']['route_decode_graph_enabled'] is True,
         'existing qualified route-ON C1 specification required')
    output=Path(output_directory)
    need(output.is_absolute() and output.resolve()==output,'canonical fresh output location')
    entry=load(SERVER,server_sha256,'_c1_profile_server_contract')
    combined=entry.load(entry.COMBINED,'_c1_profile_combined_contract'); combined_pins=combined.source_pins()
    server_pins={**combined_pins,**{str(p):h for p,h in entry.FIXED.items()},str(SERVER):server_sha256}
    sources=dict(base['sources'])
    for path,digest in {**server_pins,str(FACTORY):FACTORY_SHA,str(Path(__file__).resolve()):sha(__file__)}.items():
        need(path not in sources or sources[path]==digest,'conflicting original/diagnostic source pin')
        sources[path]=digest
    need(all(sha(p)==h for p,h in sources.items()),'actual diagnostic source closure')
    command=list(base['command'])
    need(command[3]==str(entry.COMBINED) and not any(flag in command for flag in
         ('--qualification-trace','--route-events','--dispatch-events','--diagnostic-out','--drain-token','--profile')),
         'original unobserved combined command')
    command[3]=str(SERVER)
    command+=['--profile',mode,'--diagnostic-out',str(output/'server'),'--drain-token',drain_token,
              '--warm-requests','8','--worker-join-seconds',str(float(worker_join_seconds))]
    keys=('configuration','source_files','source_manifest_sha256','manifest_all_sources_match','package_root_loaded',
          'model_snapshot_path','model_config_sha256','eos_ids','drafter','normal_eos','prefix_cache',
          'step_observer','qualification_trace_enabled')
    identity={k:copy.deepcopy(base['identity'][k]) for k in keys}
    identity.update(combined_entry=combined.identity_fields(SimpleNamespace(phase='formal',route_decode_graph='on',gdn_prework=base['arm']),
        combined_pins,installed=True,route_observer=False),benchmark_phase='diagnostic',
        diagnostic=entry.contract(mode,8),drain_token_sha256=hashlib.sha256(drain_token.encode()).hexdigest(),
        server_source_path=str(SERVER),server_source_sha256=server_sha256)
    jobs=dict(warm=copy.deepcopy(base['jobs']['warm']),diagnostic=copy.deepcopy(base['jobs']['formal'][:8]))
    need(all(len(v)==8 and len({j['fixture_id'] for j in v})==8 for v in jobs.values()),'same original D8 in both groups')
    need([j['body'] for j in jobs['warm']]==[j['body'] for j in jobs['diagnostic']], 'identical warm/diagnostic bodies')
    d=base['deadlines']; deadlines={k:d[k] for k in ('ready','request','warm','close','term','kill','overall','resource_interval')}
    deadlines.update(diagnostic=d['warm'],drain=drain_seconds,worker_join=worker_join_seconds)
    need(deadlines['overall']>=deadlines['ready']+deadlines['warm']+deadlines['diagnostic']+drain_seconds,
         'overall covers readiness, both eight-request groups and natural drain')
    return dict(schema='C1-pipeline-diagnostic-capture-spec-v1',formal_spec=copy.deepcopy(formal_spec),mode=mode,
        output_directory=str(output),drain_token=drain_token,command=command,environment=copy.deepcopy(base['environment']),
        transport=copy.deepcopy(base['transport']),deadlines=deadlines,identity=identity,sources=sources,server_sources=server_pins,
        qualification=copy.deepcopy(base['qualification']),fixtures=copy.deepcopy(base['fixtures']),tokenizer=copy.deepcopy(base['tokenizer']),
        jobs=jobs,concurrency=1,repeat=base['repeat'],performance_qualified=False,numerical_qualified=False,
        scope='Diagnostic overhead comparison only; spans overlap and are not exclusive GPU time. External admission/reap/restoration required.')


def validate(spec):
    need(spec.get('schema')=='C1-pipeline-diagnostic-capture-spec-v1','diagnostic capture specification')
    actual=prepare(spec['formal_spec'],spec['mode'],output_directory=spec['output_directory'],drain_token=spec['drain_token'],
        server_sha256=spec['sources'][str(SERVER)],drain_seconds=spec['deadlines']['drain'],worker_join_seconds=spec['deadlines']['worker_join'])
    need(same(spec,actual),'diagnostic specification changed')


def count_contract(spec,stage): return native().count_contract(spec,stage)


def check_group(group,spec,stage,decode_text):
    """Reuse the frozen eight-request raw gate through an explicit READ-ONLY view.

The diagnostic raw group remains admission_probe, including its original bytes,
job/body hashes, timings and formal_point_eligible=False. Only the validator's
phase/queue selector is warm, to select its identical eight-request count gate.
"""
    need(stage in STAGES and group['phase']==CLIENT_PHASE[stage] and group['summary']['formal_point_eligible'] is False,
         'original nonformal diagnostic phase')
    view={**group,'phase':'warm'}
    spec_view={**spec,'jobs':{'warm':spec['jobs'][stage]}}
    checked=native().check_group(view,spec_view,'warm',decode_text)
    return dict(status='PASS',raw_requests_checked=checked['raw_requests_checked'],count_verified=True,
        stage=stage,raw_client_phase=group['phase'],validation_view_phase='warm',diagnostic_metrics=checked['point'],
        numerical_qualified=False,performance_qualified=False)


def closed_receipts(out,spec,pid,groups):
    """Bind actual local drain rows to the sixteen remotely received raw DONEs."""
    n=native(); join=n.load('http_join'); directory=Path(out)/'server'
    identity=read(directory/'identity.json'); closure=read(directory/'closure.json')
    need(same(identity,spec['identity']),'closed identity differs from actual ready contract')
    need(closure['schema']=='C1-pipeline-diagnostic-http-v1' and closure['status']=='RECORDED' and closure['pid']==pid
         and closure['worker_stopped'] is closure['handlers_stopped'] is True and closure['worker_failure'] is None
         and closure['errors']==[] and same(closure['sources'],spec['server_sources'])
         and closure['performance_qualified'] is closure['numerical_qualified'] is False,'actual clean server closure')
    drain=closure['drain']; rows=drain['requests']
    need(drain['phase']=='DRAINED' and drain['expected_requests']==drain['warm_requests']==8
         and drain['attempts']==drain['normal_done_flushes']==16 and len(rows)==16
         and drain['active_handlers']==drain['active_completions']==0 and drain['errors']==[], 'fixed fully drained16 requests')
    accepted,stopped=drain['accepted_at_ns'],drain['stopped_at_ns']
    need(type(accepted) is type(stopped) is int and 0<accepted<=stopped,'actual accepted/worker-stop clocks')
    wire=[]; previous_done=None
    for stage in STAGES:
        for row in groups[stage]['requests']:
            rid,_,_=join.raw_completion(row)
            start,end=row['start_perf_counter_s'],row['done_perf_counter_s']
            need(math.isfinite(start) and math.isfinite(end) and start<end
                 and (previous_done is None or start>=previous_done),'serial actual C1 POST to DONE')
            previous_done=end; wire.append((stage,rid))
    need(len(wire)==16 and len({rid for _,rid in wire})==16,'all16 unique actual response UUIDs')
    for i,(row,(stage,rid)) in enumerate(zip(rows,wire),1):
        need(row['ordinal']==i and row['phase']==stage and row['rid']==rid,'server/client actual request order and UUID')
        clocks=[row[k] for k in ('entered_ns','done_flush_ns','handler_exit_ns')]
        need(all(type(x) is int and x>0 for x in clocks) and clocks[0]<=clocks[1]<=clocks[2]<=accepted,'normal local DONE/handler closure clocks')
    profile_path=directory/'profile.json'
    if spec['mode']=='off':
        need(closure['profile_status'] is None and not profile_path.exists(),'OFF has no profile payload')
    else:
        profile=read(profile_path)
        need(closure['profile_status']==profile['status']=='RECORDED' and profile['schema']=='C1-pipeline-profile-v1'
             and profile['overflow'] is False and profile['event_count']==len(profile['events'])>0
             and same(profile['sources'],spec['server_sources'])
             and profile['performance_qualified'] is profile['numerical_qualified'] is False,'closed retained profile/source')
        for i,event in enumerate(profile['events']):
            need(event['sequence']==i and event.get('error') is None and type(event['start_ns']) is type(event['end_ns']) is int
                 and event['start_ns']<=event['end_ns'],'complete ordered profile events')
    return dict(status='RECORDED',requests=16,closure_sha256=sha(directory/'closure.json'),
        profile_sha256=sha(profile_path) if profile_path.exists() else None,
        numerical_qualified=False,performance_qualified=False,
        scope='Local terminal flush/handler stop joined with remote raw DONE; no assertion all ahead GPU work completed before drain.')


class System:
    def __init__(self): self.base=native().System()
    def __getattr__(self,name): return getattr(self.base,name)
    async def group(self,spec,stage,output):
        client=native().load('native_cleanup'); d=spec['deadlines']
        return await client.run_group(f"http://127.0.0.1:{spec['transport']['port']}",spec['jobs'][stage],1,
            phase=CLIENT_PHASE[stage],qualification=count_contract(spec,stage),request_timeout_s=d['request'],
            group_timeout_s=d[stage],cleanup_timeout_s=d['close'],output_dir=output)
    async def drain(self,port,token,seconds,output):
        import httpx
        raw=dict(method='POST',url=f'http://127.0.0.1:{port}/_diagnostic/drain',request_body_bytes=0,
                 request_headers={'X-Diagnostic-Drain-Token':token},start_monotonic=time.monotonic(),close_complete=False)
        client=httpx.AsyncClient(timeout=seconds,trust_env=False,follow_redirects=False); failure=None
        try:
            async with asyncio.timeout(seconds):
                response=await client.post(raw['url'],headers=raw['request_headers'],content=b'')
                raw.update(status=response.status_code,headers=dict(response.headers),body_base64=base64.b64encode(response.content).decode())
        except BaseException as exc:
            failure=exc; raw['error']=dict(type=type(exc).__name__,message=str(exc))
        finally:
            closer=asyncio.create_task(client.aclose()); end=asyncio.get_running_loop().time()+5
            while not closer.done() and asyncio.get_running_loop().time()<end:
                try: await asyncio.wait([closer],timeout=max(0,end-asyncio.get_running_loop().time()))
                except asyncio.CancelledError as exc:
                    failure=failure or exc; raw['close_cancelled']=True
            if not closer.done(): closer.cancel(); raw['close_error']='client close deadline'
            elif closer.cancelled(): raw['close_error']='client close cancelled'
            elif closer.exception() is not None: raw['close_error']=repr(closer.exception())
            else: raw['close_complete']=True
            raw['end_monotonic']=time.monotonic(); write(output,raw)
        if failure is not None: raise failure
        return raw


async def run(spec,out,*,checkpoint,backend=None):
    """Mandatory outer callback must settle any in-flight resource probes on cancel."""
    need(callable(checkpoint),'external resource/service owner checkpoint required'); validate(spec)
    out=Path(out).resolve(); need(str(out)==spec['output_directory'],'actual registered output'); out.mkdir(parents=True,exist_ok=False)
    spec=copy.deepcopy(spec); system=backend or System(); cap=native().load('http_capture_v2')
    d=spec['deadlines']; port=spec['transport']['port']; child=task=None; stdout=stderr=None; stage='preflight'; groups={}
    control_resolved=True
    result=dict(schema='C1-pipeline-diagnostic-capture-result-v1',status='INCOMPLETE',mode=spec['mode'],started_at=time.time(),
        count_verified=False,natural_exit=False,forced_stop=False,server_reaped=False,cleanup_complete=False,errors=[],
        numerical_qualified=False,performance_qualified=False,scope=spec['scope'])
    write(out/'spec.json',spec); write(out/'complete.json',result)
    write(out/'launch.json',dict(command=spec['command'],environment=spec['environment']))
    events=(out/'events.jsonl').open('x')
    def event(label,**fields): events.write(json.dumps(dict(label=label,time=time.time(),**fields))+'\n'); events.flush()
    def error(exc,where): result['errors'].append(dict(stage=where,type=type(exc).__name__,message=str(exc)))
    async def check(label,pid):
        value=checkpoint(label,pid)
        if inspect.isawaitable(value): await value
        event('checkpoint',stage=label,pid=pid)
    async def group(name):
        try:
            raw=await system.group(spec,name,out/name); write(out/(name+'-group.json'),raw); return raw
        except BaseException as exc:
            if hasattr(exc,'result'): write(out/(name+'-group.json'),exc.result)
            raise
    try:
        decode_text=system.decoder(spec)
        async with asyncio.timeout(d['overall']):
            await check('pre-spawn',None)
            before=await system.port(port); write(out/'port-before.json',before); need(cap.no_process(before),'port occupied/unknown')
            stdout=(out/'server.stdout').open('xb'); stderr=(out/'server.stderr').open('xb')
            stage='spawn'; write(out/'spawn-intent.json',dict(command=spec['command'],time=time.time()))
            child=system.spawn(spec['command'],spec['environment'],stdout,stderr)
            write(out/'child.json',dict(pid=child.pid,time=time.time())); await check('spawned',child.pid)
            identity=await system.identity(child.pid); write(out/'child-identity.json',identity)
            need(identity['returncode']==0 and identity['stdout'].split()[:2]==[str(child.pid),str(os.getpid())],'actual direct server child')
            stage='readiness'
            async with asyncio.timeout(d['ready']):
                sampled=time.monotonic()
                while True:
                    need(child.poll() is None,'server exited before ready')
                    raw=await system.readiness(port); event('readiness',raw=raw)
                    if cap.ready_identity(raw,spec['identity']):
                        listener=await system.port(port,child.pid); write(out/'listener.json',listener)
                        need(cap.owned_listener(listener,child.pid,port),'owned localhost listener'); write(out/'ready.json',raw); break
                    if time.monotonic()-sampled>=d['resource_interval']:
                        await check('starting',child.pid); sampled=time.monotonic()
                    await asyncio.sleep(.1)
            await check('ready',child.pid)
            for stage in STAGES:
                task=asyncio.create_task(group(stage),name='diagnostic-'+stage)
                while not task.done():
                    await asyncio.wait([task],timeout=d['resource_interval'])
                    need(child.poll() is None,'server exited during requests'); await check(stage,child.pid)
                groups[stage]=task.result(); checked=check_group(groups[stage],spec,stage,decode_text)
                write(out/(stage+'-raw-check.json'),checked); await check('after-'+stage,child.pid)
            result['count_verified']=True
            stage='drain'; end=asyncio.get_running_loop().time()+d['drain']; attempt=0
            async def drain_check():
                await check('draining',child.pid)
                now=asyncio.get_running_loop().time()
                need(now<end,'drain checkpoint exceeded absolute deadline')
                return now
            sampled=await drain_check()
            while True:
                if asyncio.get_running_loop().time()-sampled>=d['resource_interval']:
                    sampled=await drain_check()
                remaining=end-asyncio.get_running_loop().time(); need(remaining>0 and child.poll() is None,'drain deadline/early server exit')
                control_resolved=False
                raw=await system.drain(port,spec['drain_token'],min(2,remaining),out/f'drain-{attempt:03d}.json'); attempt+=1
                control_resolved=raw.get('close_complete') is True
                need(raw['close_complete'] is True and 'error' not in raw and 'close_error' not in raw,'drain client incomplete')
                # POST and client-close time consume the same absolute budget,
                # including the final 200; a closed child cannot bypass it.
                need(asyncio.get_running_loop().time()<end,'drain response/close exceeded absolute deadline')
                if raw['status']==200:
                    need(json.loads(base64.b64decode(raw['body_base64'],validate=True))==dict(status='DRAIN_ACCEPTED',performance_qualified=False),
                         'exact accepted drain response'); result['drain_accepted']=True; break
                need(raw['status']==409,'unexpected drain response')
                # DONE receive can precede the server handler finally. Only this
                # idempotent rejected control request is retried; never inference.
                await asyncio.sleep(min(.1,max(0,end-asyncio.get_running_loop().time())))
            sampled=await drain_check()
            while child.poll() is None:
                need(asyncio.get_running_loop().time()<end,'natural drain exit deadline')
                if asyncio.get_running_loop().time()-sampled>=d['resource_interval']:
                    sampled=await drain_check()
                await asyncio.sleep(min(.1,max(0,end-asyncio.get_running_loop().time())))
            need(asyncio.get_running_loop().time()<end,'natural drain exit observed after absolute deadline')
            await system.wait(child,d['kill']); need(child.returncode==0,'diagnostic natural exit failed')
            result['natural_exit']=True
    except BaseException as exc: error(exc,stage)
    finally:
        async def finish_owned():
            try:
                if task is not None and not task.done(): task.cancel(); await asyncio.wait([task],timeout=d['close']+1)
                if task is not None and not task.done(): task.cancel(); error(TimeoutError('client unresolved'),'client-cleanup')
                if task is not None and task.done() and not task.cancelled(): task.exception()  # Retrieve retained partial-group failure.
            except BaseException as exc: error(exc,'client-cleanup')
            finally:
                if child is not None:
                    try:
                        if child.poll() is None:
                            result['forced_stop']=True; event('server-term',pid=child.pid)
                            try: child.terminate(); await system.wait(child,d['term'])
                            except subprocess.TimeoutExpired: event('server-term-timeout',pid=child.pid)
                            except BaseException as exc: error(exc,'server-term')
                            if child.poll() is None: event('server-kill',pid=child.pid); child.kill()
                        await system.wait(child,d['kill'])
                        absent=await system.absent(child.pid); write(out/'child-absence.json',absent)
                        port_after=await system.port(port); write(out/'port-after.json',port_after)
                        need(type(child.returncode) is int and cap.no_process(absent) and cap.no_process(port_after),'strict server/port absence')
                        write(out/'release.json',dict(pid=child.pid,returncode=child.returncode,waited_reaped=True,strict_absence=True,
                            port_absent=True,natural_exit=result['natural_exit'],forced_stop=result['forced_stop'],time=time.time()))
                        result['server_reaped']=True
                    except BaseException as exc: error(exc,'server-cleanup')
        cleaner=asyncio.create_task(finish_owned(),name='diagnostic-owned-cleanup')
        end=asyncio.get_running_loop().time()+d['close']+d['term']+d['kill']+12
        while not cleaner.done() and asyncio.get_running_loop().time()<end:
            try: await asyncio.wait([cleaner],timeout=max(0,end-asyncio.get_running_loop().time()))
            except asyncio.CancelledError as exc: error(exc,'cancel-during-cleanup')
        if not cleaner.done(): cleaner.cancel(); error(TimeoutError('owned cleanup deadline'),'cleanup')
        elif cleaner.cancelled(): error(RuntimeError('cleanup task cancelled'),'cleanup')
        elif cleaner.exception() is not None: error(cleaner.exception(),'cleanup')
        result['cleanup_task_resolved']=cleaner.done(); result['drain_client_resolved']=control_resolved
        result['client_task_resolved']=(task is None or task.done()) and control_resolved
        result['cleanup_complete']=result['server_reaped'] and cleaner.done() and not cleaner.cancelled() and result['client_task_resolved']
        try:
            await check('after-reap' if result['server_reaped'] else 'cleanup-incomplete',None if result['server_reaped'] or child is None else child.pid)
            validate(spec); result['small_pins_rechecked']=True
        except BaseException as exc: error(exc,'postflight')
        if stdout is not None: stdout.close()
        if stderr is not None: stderr.close()
        if result['natural_exit'] and result['server_reaped'] and not result['errors']:
            try: result['closed_server']=closed_receipts(out,spec,child.pid,groups)
            except BaseException as exc: error(exc,'closed-receipts')
        events.close(); result['ended_at']=time.time()
        result['status']='PASS' if (result['count_verified'] and result['natural_exit'] and not result['forced_stop']
            and result['cleanup_complete'] and 'closed_server' in result and not result['errors']) else 'FAIL'
        result['artifacts']={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size)
            for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'complete.json'}
        write(out/'complete.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.parse_args(argv)
    print(json.dumps(dict(schema='C1-pipeline-diagnostic-capture-description-v1',modes=list(MODES),warm=8,diagnostic=8,
        default_action='describe only; external owner calls prepare/validate/run',performance_qualified=False,
        normal_stop='authenticated drain then natural exit; TERM/KILL is failure cleanup'),indent=2)); return 0


if __name__=='__main__': raise SystemExit(main())
