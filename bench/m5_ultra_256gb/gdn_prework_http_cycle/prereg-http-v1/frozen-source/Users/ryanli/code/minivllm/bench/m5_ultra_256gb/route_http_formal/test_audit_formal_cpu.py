"""Fake-only independent audit regressions; never model/performance evidence.

The transport fixture uses frozen native serialization, queue and parsing over
httpx.MockTransport and the actual route formal run(). Qualification metadata
is explicitly mocked; no socket, service, model or Metal launch is permitted.
"""
import ast
import asyncio
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from bench.m5_ultra_256gb.route_http_formal import audit_formal as a, formal
from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
from bench.m5_ultra_256gb.pareto_realistic import test_audit_native_formal_cpu as oldaudit
from bench.m5_ultra_256gb.pareto_realistic import native_server as server


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False)+'\n')
def ref(root,path):return dict(path=str(path.relative_to(root)),sha256=a.sha(path))
def cmd_port(port):return ['/usr/sbin/lsof','-nP','-a',f'-iTCP:{port}','-sTCP:LISTEN','-Fpn']
def cmd_pid(pid):return ['/bin/ps','-p',str(pid),'-o','pid=']
def raw(command,inner=False):return dict(command=command,returncode=1,stdout='',**({'stderr':''} if inner else {}))


def lifecycle(root):
    root.mkdir();source=dict(path=str(a.HERE/'owner.py'),sha256=a.OWNER_SHA)
    cold=dict(journal_path=str(root/'journal'),controller=dict(sha256=a.CONTROLLER_SHA),predecessor={'mock':'closed'})
    plan=dict(schema='E024-route-http-formal-owner-plan-v1',trials=list(a.PHASES),execution=dict(owner=source),cold_restart=cold,
              http=dict(transport=dict(port=18181)))
    write(root/'owner-plan.json',plan);digest=a.sha(root/'owner-plan.json')
    stopped=dict(state='STOPPED',owner_directory=str(root),owner_plan=dict(sha256=digest),owner_source=source,
        id='fake-cycle',removed_at=10,trials=list(a.PHASES),predecessor=cold['predecessor'],controller_sha256=a.CONTROLLER_SHA)
    write(root/'stopped-journal.json',stopped)
    attempt=dict(schema='cold-service-owner-attempt-v1',owner_pid=900,owner_identity=dict(pid=900),owner_source=source,
        owner_plan_sha256=digest,journal_id='fake-cycle',removed_at=10,created_at=11,journal_path=cold['journal_path'],
        controller_sha256=a.CONTROLLER_SHA,stopped_journal_sha256=a.sha(root/'stopped-journal.json'))
    write(root/'attempt.json',attempt);bound={k:attempt[k] for k in
        ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
    bound['attempt_sha256']=a.sha(root/'attempt.json');write(root/'spawn-intent.json',bound)
    children=[]
    for i,phase in enumerate(a.PHASES):
        folder=root/'phases'/phase;trial=folder/'trial';pid=1000+i;start=20+i*10
        write(folder/'intent.json',dict(phase=phase,time=start,**bound))
        identity=dict(pid=pid,ppid=900)
        write(folder/'child.json',dict(phase=phase,pid=pid,owner_pid=900,identity=identity,
            ownership='direct-Popen-returned',spawned_at=start+1))
        write(trial/'spawn-intent.json',dict(time=start,command=['CPU-FAKE']))
        write(trial/'child.json',dict(pid=pid,time=start+1))
        write(trial/'release.json',dict(pid=pid,returncode=-15,waited_reaped=True,strict_absence=True,port_absent=True,time=start+5))
        write(folder/'child-absence-00.json',raw(cmd_pid(pid)));write(folder/'port-absence-00.json',raw(cmd_port(18181)))
        write(folder/'release.json',dict(phase=phase,pid=pid,returncode=-15,waited_reaped=True,strict_absence=True,
              time=start+6,absence=ref(root,folder/'child-absence-00.json')))
        for name,command in (('port-before.json',cmd_port(18181)),('child-absence.json',cmd_pid(pid)),('port-after.json',cmd_port(18181))):
            write(trial/name,raw(command,True))
        write(trial/'complete.json',dict(artifacts={p.name:dict(sha256=a.sha(p),bytes=p.stat().st_size)
                                                   for p in trial.iterdir() if p.is_file()}))
        children.append(dict(phase=phase,pid=pid,identity=identity,**{name:ref(root,folder/(name+'.json')) for name in ('intent','child','release')}))
    write(root/'children.json',dict(schema='cold-service-owner-children-v1',children=children))
    write(root/'release.json',dict(schema='cold-service-owner-release-v2',children=children,children_sha256=a.sha(root/'children.json'),
        all_children_reaped=True,time=150,**bound))
    return plan,dict(owner_pid=900,owner_plan_sha256=digest)


class Backend(oldaudit.UniqueBackend):
    async def group(self,spec,phase,output):
        self.phases.append(phase);fixtures={f['prompt_text']:f for f in spec['fixtures']};counter=0
        async def handler(request):
            nonlocal counter
            i=counter;counter+=1;body=json.loads(request.content);fixture=fixtures[body['prompt']]
            eos=spec['identity']['eos_ids'][0];tokens=[10,eos] if i%3==0 else [eos] if i%3==1 else [11]*512
            rid=f'{self.child.pid*10000+(1000 if phase=="formal" else 1)+i:032x}'
            terminal=server.completion(server.Job(rid,fixture['prompt_ids'],512),tokens,set(spec['identity']['eos_ids']))
            text=old.decode(terminal['benchmark_committed_output_ids'])
            delta=old.wire(dict(id='cmpl-'+rid,choices=[dict(index=0,text=text,finish_reason=None)])) if text else b''
            return httpx.Response(200,stream=old.Stream(delta+old.wire(terminal)+old.wire('[DONE]')),
                                  headers={'content-type':'text/event-stream'})
        return await formal.load('native_cleanup').run_group(f"http://127.0.0.1:{spec['transport']['port']}",
            spec['jobs'][phase],spec['concurrency'],phase=phase,qualification=formal.count_contract(spec,phase),
            request_timeout_s=2,group_timeout_s=2,cleanup_timeout_s=.1,transport=httpx.MockTransport(handler),output_dir=output)


async def wire_fixture(root):
    owner=root/'wire-owner';owner.mkdir();specs=[];index=[];stages=[]
    route= formal.module(formal.SERVER);pins=route.source_pins()
    for i,stage in enumerate(a.PHASES):
        r,c,arm=stage.split('-');C=int(c[1:]);repeat=int(r[1:]);original=old.prepare(root/('metadata-'+stage),C)
        native=formal.donor();spec=native.prepare(original['base_plan'],C,repeat,qualification=original['qualification'],
            environment=original['environment'],transport=original['transport'],deadlines=original['deadlines'])
        spec['command'],spec['identity']=formal.select_entry(spec,arm,route,pins)
        spec.update(schema='E024-route-http-formal-spec-v1',arm=arm,phase=stage,observer_enabled=False,
            baseline_qualification=spec['qualification'],route_qualification=dict(cycle_audit={'path':'CPU-FAKE','sha256':'not-evidence'}))
        specs.append(spec);folder=owner/'phases'/stage;folder.mkdir(parents=True)
        backend=Backend(spec);backend.child.pid=30000+i
        with patch.object(formal,'validate',return_value=None):result=await formal.run(spec,folder/'trial',checkpoint=lambda *args:None,backend=backend)
        if result['status']!='PASS':raise AssertionError(result['errors'])
        child=dict(phase=stage,pid=backend.child.pid,owner_pid=os.getpid(),identity=dict(pid=backend.child.pid,ppid=os.getpid()),
                   ownership='direct-Popen-returned',spawned_at=backend.spawned_at)
        release=dict(phase=stage,pid=child['pid'],returncode=backend.child.returncode,waited_reaped=True,strict_absence=True,time=time.time())
        write(folder/'child.json',child);write(folder/'release.json',release)
        index.append(dict(phase=stage,pid=child['pid']))
        stages.append(dict(phase=stage,arm=arm,status='PASS',result=ref(owner,folder/'trial/complete.json'),
            point=ref(owner,folder/'trial/point.json'),numerical_audit=spec['qualification']['audit']))
    write(owner/'children.json',dict(schema='cold-service-owner-children-v1',children=index));write(owner/'formal-specs.json',specs)
    write(owner/'points.json',dict(schema='E024-route-http-formal-points-v1',trials=stages,performance_qualified=False))
    wrapper=dict(trials=list(a.PHASES),http=dict(environment=specs[0]['environment'],transport=specs[0]['transport']))
    return owner,wrapper,specs,dict(owner_pid=os.getpid(),phases=stages)


class BindingTests(unittest.TestCase):
    def test_import_and_incomplete_never_import_backend_or_open_ledger(self):
        original=builtins.__import__
        def guard(name,*args,**kwargs):
            if name.split('.')[0] in ('mlx','mlx_lm','mlx_vlm','omlx','tokenizers'):raise AssertionError(name)
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guard):runpy.run_path(a.__file__)
        with tempfile.TemporaryDirectory() as tmp:
            result=a.audit(Path(tmp)/'owner',Path(tmp)/'journal','missing')
        self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['performance_qualified'] or result['http_raw_opened'])

    def test_exact_outer_three_fields_do_not_fabricate_stderr(self):
        row=raw(cmd_pid(123));a.outer_absence(row,cmd_pid(123))
        for bad in (dict(row,stderr=''),dict(row,returncode=True),dict(row,stdout='123'),dict(row,command=cmd_pid(456))):
            with self.assertRaises(ValueError):a.outer_absence(bad,cmd_pid(123))

    def test_all12_actual_direct_children_and_separate_inner_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/'owner';plan,complete=lifecycle(root)
            _,_,lives,proof=a.owner_binding(root,plan,complete)
            self.assertEqual([x['phase'] for x in lives],list(a.PHASES));self.assertEqual(len(proof),12)
            self.assertEqual(len({x['pid'] for x in lives}),12)

    def test_missing_extra_wrong_pid_wait_and_stale_inner_absence_reject(self):
        for mutation in ('wrong-pid','wait','unknown','prefix','inner-command','inner-stderr','cross-time'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve()/'owner';plan,complete=lifecycle(root);phase=a.PHASES[1];folder=root/'phases'/phase
                index=a.read(root/'children.json');release=a.read(root/'release.json')
                if mutation in ('wrong-pid','wait','cross-time'):
                    p=folder/'release.json';row=a.read(p)
                    if mutation=='wrong-pid':row['pid']+=1
                    elif mutation=='wait':row['waited_reaped']=False
                    else:row['time']=10
                    write(p,row);index['children'][1]['release']=ref(root,p)
                elif mutation=='unknown':write(root/'phases/unlisted/intent.json',{'unknown':True})
                elif mutation=='prefix':index['children'].pop()
                else:
                    p=folder/'trial/child-absence.json';row=a.read(p)
                    if mutation=='inner-command':row['command']=cmd_pid(999)
                    else:row['stderr']='probe failed'
                    write(p,row);top=a.read(folder/'trial/complete.json');top['artifacts'][p.name]=dict(sha256=a.sha(p),bytes=p.stat().st_size)
                    write(folder/'trial/complete.json',top)
                write(root/'children.json',index);release.update(children=index['children'],children_sha256=a.sha(root/'children.json'));write(root/'release.json',release)
                with self.assertRaises(ValueError):a.owner_binding(root,plan,complete)

    def test_adapter_copy_preserves_same_basename_sources_and_exact_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();pins={};meta={};folder=root/'preflight'
            for parent in ('native','route'):
                source=root/parent/'formal.py';source.parent.mkdir();source.write_text(parent)
                copy_path=Path('frozen-adapters')/source.relative_to('/');target=folder/copy_path
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
                pins[str(source)]=a.sha(source);meta[str(source)]=dict(sha256=a.sha(source),bytes=source.stat().st_size,copy=str(copy_path))
            write(folder/'adapter-pins.json',meta);a.adapter_copies(folder,pins)
            bad=copy.deepcopy(meta);bad[next(iter(bad))]['copy']='frozen-adapters/formal.py';write(folder/'adapter-pins.json',bad)
            with self.assertRaises(ValueError):a.adapter_copies(folder,pins)

    def test_native157_preparation_changes_only_new_adapter_copy_path_check(self):
        oldtree=ast.parse((a.NATIVE/'audit_http_owner_v2.py').read_bytes());newtree=ast.parse(Path(a.__file__).read_bytes())
        oldfn=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='preparation_pins')
        newfn=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name=='preparation_pins')
        class Replace(ast.NodeTransformer):
            def visit_For(self,node):
                if isinstance(node.target,ast.Tuple) and [n.id for n in node.target.elts if isinstance(n,ast.Name)]==['source','h'] and 'wrapper' in ast.unparse(node.iter):
                    return ast.parse("adapter_copies(folder,wrapper['execution']['pins'])").body[0]
                return self.generic_visit(node)
        self.assertEqual(ast.dump(Replace().visit(oldfn),include_attributes=False),ast.dump(newfn,include_attributes=False))


class WireTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.owner,self.wrapper,self.specs,self.complete=await wire_fixture(self.root)
    def check(self):return a.audit_trials(self.owner,self.wrapper,self.specs,self.complete,old.decode)

    def test_real_frozen_client_864requests_and12_unapproved_points(self):
        children,points=self.check();self.assertEqual(len(children),12);self.assertEqual(sum(p['requests'] for p in points),768)
        self.assertEqual(sum(p['warm_requests_checked'] for p in points),96);self.assertEqual(len(a.paired_summary(points)['pairs']),6)
        self.assertTrue(all(not p['actual_compiled_invocations_observed'] for p in points))
        self.assertTrue(all(not a.read(self.owner/'phases'/p['phase']/'trial/point.json')['performance_qualified'] for p in points))

    def test_full_prompt_eos_usage_text_and_raw_clock_corruption(self):
        spec=self.specs[0];group=a.read(self.owner/'phases'/a.PHASES[0]/'trial/formal-group.json')
        for mutation in ('prompt','eos','usage','text','clock'):
            bad=copy.deepcopy(group);row=bad['requests'][0]
            if mutation=='prompt':old.rewrite_terminal(row,lambda t:t['benchmark_prompt_ids'].__setitem__(0,777))
            elif mutation=='eos':old.rewrite_terminal(row,lambda t:t.__setitem__('benchmark_terminal_eos_id',None))
            elif mutation=='usage':old.rewrite_terminal(row,lambda t:t['usage'].__setitem__('completion_tokens',True))
            elif mutation=='text':row['answer']='forged'
            else:row['done_perf_counter_s']+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):formal.check_group(bad,spec,'formal',old.decode)

    def test_wire_metrics_ignore_derived_count_summary_and_reject_deadline(self):
        group=a.read(self.owner/'phases'/a.PHASES[2]/'trial/formal-group.json');before=a.raw_metrics(group)
        group['summary']={'aggregate_output_tok_s':9999}
        for row in group['requests']:row['qualified_output_tokens']=999;row['reported_completion_tokens']=999
        self.assertEqual(a.raw_metrics(group),before)
        group['requests'][0]['absolute_deadline_perf_counter_s']+=1
        with self.assertRaises(ValueError):a.raw_metrics(group)

    def test_observer_identity_cannot_promote_installation_to_hit_or_swap_arm(self):
        for i in (0,1):
            original=self.specs[i];a.observer_identity(original,original['phase'])
            for mutation in ('trace','entry','observer','hit'):
                spec=copy.deepcopy(original)
                if mutation=='trace':spec['command'].append('--route-events')
                elif mutation=='entry':spec['command'][3]='/wrong/native.py'
                elif mutation=='observer':spec['identity']['step_observer']=True
                elif i==1:spec['identity']['route_decode_graph']['actual_compiled_invocations_observed']=True
                else:spec['identity']['route_decode_graph']={'installed':True}
                with self.subTest(arm=i,mutation=mutation),self.assertRaises(ValueError):a.observer_identity(spec,original['phase'])

    def test_unapproved_point_falseflag_and_cross_group_uuid_are_required(self):
        folder=self.owner/'phases'/a.PHASES[0]/'trial';point=a.read(folder/'point.json');point['performance_qualified']=True;write(folder/'point.json',point)
        with self.assertRaises(ValueError):self.check()
        group=a.read(folder/'formal-group.json')
        # Keep both records internally valid so the duplicate UUID itself is the rejection.
        group['requests'][1]=copy.deepcopy(group['requests'][0])
        with self.assertRaisesRegex(ValueError,'unique response UUIDs'):a.raw_metrics(group)


class ResourceAndPairTests(unittest.TestCase):
    def test_original_resource_thresholds_and_real_phase_lifetimes(self):
        helper=oldaudit.ResourceTests(methodName='test_real_warm_formal_labels_with_exact_current_original_pid')
        helper.setUp()
        try:
            result=a.resources(helper.root,helper.complete,helper.children,helper.pid,helper.created,helper.names)
            self.assertEqual(result['pressure'],1)
            helper.rows[0]['pressure_level_value']=2;helper.save()
            with self.assertRaises(ValueError):a.resources(helper.root,helper.complete,helper.children,helper.pid,helper.created,helper.names)
        finally:helper.doCleanups()

    def test_pairs_use_individual_process_ratios_and_three_process_medians(self):
        points=[]
        for phase in a.PHASES:
            r,c,arm=phase.split('-');repeat=int(r[1:]);speed=repeat*(2 if arm=='on' else 1)
            points.append(dict(phase=phase,arm=arm,repeat=repeat,concurrency=int(c[1:]),requests=64,
                aggregate_output_tok_s=speed,e2e_p50_s=10/speed,e2e_p95_s=12/speed,common_window_s=30+repeat))
        result=a.paired_summary(points)
        self.assertTrue(all(p['throughput_on_over_off']==2 for p in result['pairs']))
        self.assertEqual(result['process_medians'][1]['metrics']['aggregate_output_tok_s']['median'],4)
        bad=copy.deepcopy(points);bad[1]['repeat']=2
        with self.assertRaises(ValueError):a.paired_summary(bad)
        with self.assertRaises(ValueError):a.paired_summary(points[:-1])

if __name__=='__main__':unittest.main()
