"""Single-phase diagnostic owner glue, not a new cold lifecycle/controller.

No standalone launch CLI. The registered outer owner supplies its existing
durable spawn tracker, bounded resource probe and strict reap function. All run
in this same process, so the owned PID is the actual model server. Unknown
launch and unsettled cleanup never authorize restoration or a following mode.
"""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
CAPTURE=HERE/'capture.py'
MODES=('off','runner','full')


def need(ok,message):
    if not ok: raise ValueError(message)


def write(path,value,*,exclusive=False):
    path=Path(path)
    if exclusive:
        with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    else:
        tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load_capture(ref):
    need(type(ref) is dict and set(ref)=={'path','sha256'} and ref['path']==str(CAPTURE),
         'actual adjacent capture source reference required')
    need(hashlib.sha256(CAPTURE.read_bytes()).hexdigest()==ref['sha256'],'registered capture source changed')
    spec=importlib.util.spec_from_file_location('_c1_profile_owned_capture',CAPTURE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def prepare_modes(formal_spec,output,nonces,*,capture_source,server_sha256,
                  order=MODES,drain_seconds=30,worker_join_seconds=10,_capture=None):
    """CPU-only recipe: one exact C1 combined-ON spec, three fresh processes.

    The outer plan must freeze this order/spec set and all source/runtime/model
    pins before cold prepare. This neither registers a cycle nor stops services.
    _capture is only a fake-test dependency; production leaves it unset.
    """
    need(type(order) in (list,tuple) and len(order)==3 and set(order)==set(MODES),
         'each diagnostic mode exactly once in the registered order')
    need(type(nonces) is dict and set(nonces)==set(MODES)
         and all(type(n) is str for n in nonces.values()) and len(set(nonces.values()))==3,
         'three distinct owner-generated drain nonces')
    root=Path(output)
    need(root.is_absolute() and root==root.resolve(),'canonical fresh outer output path')
    capture=_capture or load_capture(capture_source); specs=[]
    for mode in order:
        spec=capture.prepare(formal_spec,mode,output_directory=root/'phases'/mode/'trial',
            drain_token=nonces[mode],server_sha256=server_sha256,
            drain_seconds=drain_seconds,worker_join_seconds=worker_join_seconds)
        flags=spec['identity']['combined_entry']['combined_candidates']
        need(spec['concurrency']==1 and flags['route_decode_graph_enabled'] is flags['gdn_prework_enabled'] is True
             and flags['identity_refill_enabled'] is False,'same qualified combined-ON C1 configuration')
        if specs:
            need(all(spec[k]==specs[0][k] for k in ('formal_spec','jobs','environment','transport','deadlines',
                                                  'fixtures','tokenizer','qualification')),
                 'modes changed underlying workload or configuration')
        specs.append(spec)
    return dict(schema='C1-pipeline-owner-recipe-v1',order=list(order),specs=specs,
        warm_requests_per_process=8,diagnostic_requests_per_process=8,
        capture_source=copy.deepcopy(capture_source),numerical_qualified=False,performance_qualified=False,
        execution_registered=False,scope='Source-prepared recipe only; no cold journal or device result.')


async def _settled_call(fn,*args,**kwargs):
    """Same shield/settle policy as frozen native owner checkpoints.

    The supplied callback must use existing bounded subprocess/read/wait code.
    A Python thread cannot be safely abandoned on cancellation. The registered
    outer hard owner deadline remains mandatory; no fresh timeout grants PASS.
    """
    task=asyncio.create_task(asyncio.to_thread(fn,*args,**kwargs));interrupted=False
    while not task.done():
        try: await asyncio.shield(task)
        except asyncio.CancelledError: interrupted=True
        except BaseException: break  # task.result below propagates the original failure.
    return task.result(),interrupted


def accepted(result,mode):
    need(result.get('schema')=='C1-pipeline-diagnostic-capture-result-v1' and result.get('mode')==mode
         and result.get('status')=='PASS' and result.get('errors')==[], 'diagnostic capture failed')
    need(all(result.get(k) is True for k in ('count_verified','natural_exit','server_reaped','cleanup_complete',
         'cleanup_task_resolved','client_task_resolved','drain_client_resolved','small_pins_rechecked'))
         and result.get('forced_stop') is False,'natural drained capture and settled clients required')
    need(result.get('numerical_qualified') is result.get('performance_qualified') is False,
         'diagnostic cannot grant numerical or performance qualification')
    closed=result.get('closed_server',{})
    need(closed.get('status')=='RECORDED' and type(closed.get('requests')) is int and closed['requests']==16
         and closed.get('numerical_qualified') is closed.get('performance_qualified') is False,
         'sixteen actual remotely completed requests joined to natural server closure')


async def run_mode(spec,out,*,capture_source,registered_spawn,resource_probe,strict_reap,
                   backend=None,_capture=None):
    """Called by the cold outer owner itself, never by a subprocess wrapper.

    registered_spawn(mode, thunk): existing durable-intent/actual-Popen tracker.
    resource_probe(label, pid, cleanup=bool): existing bounded STOPPED/absence,
        resource guard; called with '<mode>:<capture label>'.
    strict_reap(mode): existing TERM/KILL/wait + exact ps/port absence; writes
        <phase>/release.json. It must retain a returned PID before identity or
        durable receipts can fail. Outer finally retries failed reap as usual.
    Callbacks are mandatory, synchronous, bounded and source-pinned externally.
    No callback result alone replaces the actual inner/outer release files.
    """
    need(all(callable(f) for f in (registered_spawn,resource_probe,strict_reap)),
         'registered spawn, settled resource and strict reap callbacks required')
    mode=spec.get('mode');need(mode in MODES,'registered diagnostic phase')
    out=Path(out);phase=out.parent
    need(out.is_absolute() and out==out.resolve() and str(out)==spec['output_directory']
         and phase.is_dir() and not out.exists(),'fresh capture under an existing registered phase directory')
    capture=_capture or load_capture(capture_source);capture.validate(spec)
    system=backend or capture.System();started=time.monotonic();child=None;entered=False;result=None;errors=[]
    current=asyncio.current_task();need(current is not None and current.cancelling()==0,'already-cancelled owner task')
    state=dict(schema='C1-pipeline-owned-phase-v1',status='INCOMPLETE',mode=mode,owner_pid=os.getpid(),
        capture_source=copy.deepcopy(capture_source),actual_server_pid=None,popen_entered=False,popen_returned=False,
        unknown_launch=False,outer_reaped=False,numerical_qualified=False,performance_qualified=False,
        full_runtime_model_verified=False,restoration_verified=False,errors=errors)
    write(phase/'glue-start.json',state,exclusive=True)
    def fail(exc,where): errors.append(dict(stage=where,type=type(exc).__name__,message=str(exc)))
    class Tracked:
        def __getattr__(self,name): return getattr(system,name)
        def spawn(self,command,environment,stdout,stderr):
            def actual():
                nonlocal child,entered
                need(not entered,'only one server Popen attempt per phase')
                entered=True
                child=system.spawn(command,environment,stdout,stderr)
                # Preserve the actual object before the outer identity/fsync path.
                need(type(child.pid) is int and child.pid>1 and child.pid!=os.getpid(),'returned server PID')
                return child
            returned=registered_spawn(mode,actual)
            need(child is not None and returned is child,'outer tracker returned a different/missing Popen')
            return returned
    async def checkpoint(label,pid):
        need(pid is None or (child is not None and pid==child.pid),'resource PID must be the actual direct server')
        _,interrupted=await _settled_call(resource_probe,mode+':'+label,pid,
            cleanup=label in ('after-reap','cleanup-incomplete'))
        if interrupted: raise asyncio.CancelledError()
    try:
        result=await capture.run(spec,out,checkpoint=checkpoint,backend=Tracked())
        accepted(result,mode)
    except BaseException as exc: fail(exc,'capture')
    finally:
        if child is not None:
            try:
                _,interrupted=await _settled_call(strict_reap,mode)
                if interrupted: fail(asyncio.CancelledError(),'cancel-during-reap')
                outer=json.loads((phase/'release.json').read_bytes())
                need(outer.get('phase')==mode and outer.get('pid')==child.pid
                     and type(outer.get('returncode')) is int and outer['returncode']==child.returncode
                     and outer.get('waited_reaped') is outer.get('strict_absence') is True,
                     'actual outer direct-server release required')
                state['outer_reaped']=True
                if result is not None and result.get('status')=='PASS':
                    inner=json.loads((out/'release.json').read_bytes())
                    need(inner.get('pid')==child.pid and inner.get('returncode')==outer['returncode']==0
                         and inner.get('waited_reaped') is inner.get('strict_absence') is inner.get('port_absent') is True
                         and inner.get('natural_exit') is True and inner.get('forced_stop') is False,
                         'inner/outer natural release mismatch')
            except BaseException as exc: fail(exc,'outer-reap')
        if current.cancelling(): fail(asyncio.CancelledError(),'owner-cancellation-observed')
        try:
            if _capture is None: load_capture(capture_source)
        except BaseException as exc: fail(exc,'capture-source-recheck')
        state.update(actual_server_pid=None if child is None else child.pid,popen_entered=entered,
            popen_returned=child is not None,unknown_launch=entered and child is None,
            elapsed_wall_s=time.monotonic()-started)
        state['status']='PASS' if result is not None and result.get('status')=='PASS' and state['outer_reaped'] and not errors else 'FAIL'
        state['capture_status']=None if result is None else result.get('status')
        write(phase/'glue-result.json',state)
    return state


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(dict(schema='C1-pipeline-owner-glue-description-v1',modes=list(MODES),
        default_action='describe only; no execution CLI',warm=8,diagnostic=8,
        lifecycle_registered=False,performance_qualified=False),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
