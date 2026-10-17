"""Independent E027 stdlib raw audit; never imports producers or a runtime.

Reuses only the source-pinned independent E026 audit's raw-byte/finite/physical
and real-weight provenance primitives. No saved input is a model output oracle:
native/control/shared full raw matrices are compared directly. No speed claim.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIMITIVES = HERE.parent / 'conv_tail_owned_copy/audit_layer.py'
PRIMITIVES_SHA = '8796ad0296860763fdf0a2ca3c23dcd92d61aceb25aae1d04b4fbc123785d96f'
CANDIDATE_SHA = '7af958ddc747142cd2996920ed7d09e6ef3d48844aaceaef3612578070fc49fc'
SCREEN_SHA = 'd01f28e36b07176e73830b424f3ab65113f1263cc1f810678f2c88b50c11dacf'
LAYOUT_SHA = '880b0ea0d9d92b01c6c1d1188610761d470d77ae40a44c77a507cc0d4aa2f8c0'
SHAPES = ((48,5120),(1024,5120),(10240,5120))
ROWS = (2,4,5)
SYNTHETIC_BANKS = ('random','codes_signed_boundaries','extreme_cancellation')
SCALE_CODES = bytes((0x00,0x80,0x01,0x81,0x30,0x38,0x40,0xb8,0x7e,0xfe))
DTYPES = {'bfloat16': ('BF16',2,'uint16-le'), 'uint32': ('U32',4,'uint32-le'), 'uint8': ('U8',1,'uint8')}
PHYSICAL_SEMANTICS = 'separate-int-bounds;current=max(top,aux-current);peak=max(top,aux-peak);aux-peak>=aux-current'


def need(ok, message):
    if not ok: raise ValueError(message)


def same(a,b): return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1<<20): h.update(block)
    return h.hexdigest()


def helpers():
    need(sha(PRIMITIVES)==PRIMITIVES_SHA,'independent E026 audit source changed')
    spec=importlib.util.spec_from_file_location('e027_independent_raw_primitives',PRIMITIVES)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def expected_banks():
    result=[dict(id=f'synthetic-N{n}-{bank}',N=n,K=k,bank=bank,seed=124000+i*100+j,real_weights=False)
            for i,(n,k) in enumerate(SHAPES) for j,bank in enumerate(SYNTHETIC_BANKS)]
    result += [dict(id=f'real-{name}',N=n,K=5120,bank='random',seed=126000+i,real_weights=True,projection=name)
               for i,(name,n) in enumerate((('in_proj_a',48),('in_proj_qkv',10240)))]
    return result


def expected_cases():
    return [dict(case=f'{b["id"]}-T{t}',T=t,**b) for b in expected_banks() for t in ROWS]


def descriptor(out,row,dtype,shape,audit,cache,with_file=True):
    fields={'shape','dtype','strides','bytes','sha256','encoding','owner_id','finite'}|({'file'} if with_file else set())
    need(set(row)==fields,'raw descriptor fields')
    width=DTYPES[dtype][1]
    need(same([row['shape'],row['dtype'],row['strides'],row['bytes'],row['encoding']],
              [shape,dtype,[shape[1],1],math.prod(shape)*width,DTYPES[dtype][2]])
         and type(row['owner_id']) is int and row['owner_id']>0,'complete canonical raw shape/dtype/stride/owner')
    need(row['finite'] is (True if dtype=='bfloat16' else None),'raw finite declaration')
    need(type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}',row['sha256']),'raw digest')
    if not with_file:return None
    rel=Path(row['file']);path=(out/rel).resolve()
    need(not rel.is_absolute() and '..' not in rel.parts and path.is_relative_to(out.resolve()),'raw path escape')
    stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns,stat.st_ino,dtype,tuple(shape))
    if key not in cache:cache[key]=audit.raw_check(path,DTYPES[dtype][0],shape)
    checked=cache[key]
    need(checked['bytes']==row['bytes'] and checked['sha256']==row['sha256'] and checked['finite'],'raw bytes/SHA/nonfinite failure')
    return path


def bare(row):return {k:v for k,v in row.items() if k!='file'}


def structured_input_sha(bank,t,k):
    """Rebuild the two exact signed/edge banks without importing NumPy."""
    need(bank in SYNTHETIC_BANKS[1:],'structured bank required')
    result=hashlib.sha256()
    pattern=(0,0x8000,1,0x8001,0x5380,0xd380,0x3f80,0xbf80)
    for row in range(t):
        r=0 if row==2 else row
        if bank=='codes_signed_boundaries':
            words=[0 if i%2==0 else 0x8000 for i in range(k)]
            for j,pos in enumerate((0,15,16,31,32,511,512,k-1)):
                words[(pos+r)%k]=(0x3f80,0xbf80,0x3f00,0xbf00)[(j+r)%4]
        else:words=[pattern[(i-r)%8] for i in range(k)]
        result.update(struct.pack('<'+'H'*k,*words))
    return result.hexdigest()


def repeated_sha(pattern,size):
    h=hashlib.sha256()
    block=pattern*((1<<20)//len(pattern))
    while size:
        take=min(size,len(block));h.update(block[:take]);size-=take
    return h.hexdigest()


def compare(a,b,dtype,audit,progress,case,label):
    diff=audit.compare_raw(a,b,DTYPES[dtype][0])
    if not diff['equal']:
        progress['first_difference']=dict(case=case,label=label,reference=str(a),candidate=str(b),**diff)
    need(diff['equal'],'original-dtype byte mismatch: '+case+'/'+label)
    return diff['bytes']


def layout(out,record,operand,label,filename):
    need(set(record)=={'label','input_owner_id','shape','actual_strides_elements','ensure_row_contiguous','source_sha256','descriptor'},'GPU descriptor fields')
    need(same([record['label'],record['input_owner_id'],record['shape'],record['actual_strides_elements'],record['ensure_row_contiguous'],record['source_sha256']],
              [label,operand['owner_id'],operand['shape'],operand['strides'],False,LAYOUT_SHA]),'actual GPU layout/operand binding')
    expected=dict(file=filename,bytes=32,dtype='uint64',shape=[4],encoding='uint64-le')
    d=record['descriptor'];need(set(d)==set(expected)|{'sha256'} and same({k:d[k] for k in expected},expected),'descriptor raw contract')
    path=out/filename
    need(path.resolve().is_relative_to(out.resolve()) and path.stat().st_size==32 and sha(path)==d['sha256'],'descriptor raw path/size/SHA')
    need(list(struct.unpack('<4Q',path.read_bytes()))==operand['shape']+operand['strides'],'actual raw GPU shape/stride words')
    need(same(record,read(path.with_suffix('.json'))),'descriptor JSON binding')


def audit_banks(out,result,audit,cache,progress):
    rows=result['bank_receipts'];metas=expected_banks()
    need([r['bank'] for r in rows]==[b['id'] for b in metas],'11 bank preparation coverage')
    for row,meta in zip(rows,metas):
        need(same([row['launch_count'],row['kernel_build_count']],[3,1]),'one prepared callable reused for three T cases')
        for field in ('inputs_before','inputs_after','layouts_before','layouts_after'):need(set(row[field])=={'weight','scales'},'bank evidence set')
        for name,dtype,shape in (('weight','uint32',[meta['N'],meta['K']//8]),('scales','uint8',[meta['N'],meta['K']//16])):
            a,b=row['inputs_before'][name],row['inputs_after'][name]
            p=descriptor(out,a,dtype,shape,audit,cache);q=descriptor(out,b,dtype,shape,audit,cache)
            need(same(bare(a),bare(b)),'bank input identity/bytes changed')
            compare(p,q,dtype,audit,progress,meta['id'],name+' before/after')
            for side,operand in (('before',a),('after',b)):
                layout(out,row['layouts_'+side][name],operand,meta['id']+'/'+name+'-'+side,f'banks/{meta["id"]}/{name}-layout-{side}.bin')
    return {r['bank']:r for r in rows}


def audit_case(out,row,meta,checks,audit,cache,parameters,bank,progress):
    case,t,n,k=meta['case'],meta['T'],meta['N'],meta['K']
    need(same(row,read(out/'cases'/case/'case.json')),'case file differs from complete')
    need(same([row[key] for key in ('case','T','N','K','bank','status')],[case,t,n,k,meta['id'],'PASS']), 'case identity/status')
    need(row['inputs_unchanged'] is True and set(row['inputs_before'])==set(row['inputs_after'])=={'x','weight','scales'},'complete operand set')
    inputs={}
    for name,dtype,shape in (('x','bfloat16',[t,k]),('weight','uint32',[n,k//8]),('scales','uint8',[n,k//16])):
        before,after=row['inputs_before'][name],row['inputs_after'][name]
        inputs[name]=descriptor(out,before,dtype,shape,audit,cache)
        after_path=descriptor(out,after,dtype,shape,audit,cache,with_file=name=='x')
        need(same(bare(before),bare(after)),'input owner/metadata/content changed')
        if name=='x':compare(inputs[name],after_path,dtype,audit,progress,case,name+' before/after')
        else:
            need(same(before,bank['inputs_before'][name]) and same(after,bare(bank['inputs_after'][name])),
                 'case weight/scale hash differs from full bank before/after bytes')
    for side in ('before','after'):
        layout(out,row['layout_'+side],row['inputs_'+side]['x'],case+'/x-'+side,f'cases/{case}/x-layout-{side}.bin')
    if meta['real_weights']:
        for name in ('weight','scales'):
            need(row['inputs_before'][name]['sha256']==parameters[meta['projection']+'.'+name]['sha256'],'real tensor differs from fixed bundle')
    else:
        count=n*(k//16)
        need(row['inputs_before']['scales']['sha256']==repeated_sha(SCALE_CODES,count),'synthetic scale-code bank changed')
        if meta['bank']!='random':
            pair=(0x76543210,0xfedcba98) if meta['bank']=='codes_signed_boundaries' else (0x77777777,0xffffffff)
            need(row['inputs_before']['weight']['sha256']==repeated_sha(struct.pack('<2I',*pair),n*(k//8)*4),'structured weight bank changed')
    if meta['bank']!='random':
        need(row['inputs_before']['x']['sha256']==structured_input_sha(meta['bank'],t,k),'signed/edge X bank changed')
    else:
        # Random-bank PRNG is not reimplemented; fixed sources and saved raw X
        # provide provenance. Check the declared clipped-normal magnitude bound.
        with inputs['x'].open('rb') as f:
            raw=f.read()
        need(all(word[0]&0x7fff<=0x4080 for word in struct.iter_unpack('<H',raw)),'random input exceeds BF16 magnitude four')
    need(set(row['outputs'])=={'native','control','shared'},'complete three-output set')
    products={name:descriptor(out,value,'bfloat16',[t,n],audit,cache) for name,value in row['outputs'].items()}
    byte_count=0
    for saved,name,label in zip(checks,('control','shared'),('original-expression-control-vs-native-M1','shared-W-vs-native-M1')):
        need(same([saved['case'],saved['label'],saved['status'],saved['exact']],[case,label,'PASS',True])
             and same(saved['reference'],row['outputs']['native']) and same(saved['actual'],row['outputs'][name]),'comparison receipt/result binding')
        byte_count+=compare(products['native'],products[name],'bfloat16',audit,progress,case,label)
    order=ROWS.index(t)
    need(same(row['route'],dict(launch_count_before=order,launch_count_after=order+1,kernel_build_count=1,required_route='shared-W',fallback=False)),
         'shared route/callable reuse proof missing')
    progress['cases_checked']=progress.get('cases_checked',0)+1
    return {'case':case,'comparisons':2,'output_bytes_compared':byte_count}


def phases(out,result,pid,ppid,launch,audit):
    rows=audit.jsonl(out/'phases.jsonl')
    expected=[('global','initial')]+[(r['case'],p) for r in expected_cases() for p in ('before_case','after_case')]+[('global','final_release')]
    need(len(rows)==len(result['physical_receipts'])==len(expected)==68,'68 phase completeness')
    last_elapsed,last_peak,last_aux=-1,-1,-1
    for i,(row,pair) in enumerate(zip(rows,expected)):
        need(same([row['index'],row['case'],row['phase'],row['pid'],row['ppid'],row['status']],[i,*pair,pid,ppid,'PASS']),'phase order/identity')
        now=row['elapsed_s']
        need(type(now) in (int,float) and math.isfinite(now) and last_elapsed<=now<=result['elapsed_wall_s'],'phase time')
        last_elapsed=now
        for key in ('active_bytes','cache_bytes','peak_bytes'):need(type(row[key]) is int and row[key]>=0,'MLX counters')
        need(row['active_bytes']+row['cache_bytes']<=1<<30 and max(last_peak,row['active_bytes'])<=row['peak_bytes']<=1<<30,'MLX numerical budget')
        last_peak=row['peak_bytes']
        raw=out/f'physical-{i:03d}.raw.json';command=read(out/f'physical-{i:03d}.command.json')
        need(command['command']==['/usr/bin/footprint','-p',str(pid),'--noCategories','-f','bytes','-j',str(raw)] and command['returncode']==0,'physical command')
        receipt=dict(path=raw.name,sha256=sha(raw),pid=pid,phase=pair[1],case=pair[0])
        need(same(row['physical_receipt'],receipt) and same(result['physical_receipts'][i],receipt),'physical raw binding')
        values=audit.physical_values(read(raw),pid)
        for field in ('current_bytes','peak_bytes','top_bytes','aux_current_bytes','aux_peak_bytes'):
            need(same(row['physical_'+field],values[field]),'derived/raw physical bytes')
        need(values['aux_peak_bytes']>=last_aux,'auxiliary historical peak fell');last_aux=values['aux_peak_bytes']
    return {'count':68,'maximum_mlx_peak_bytes':last_peak,'maximum_mlx_active_bytes':max(r['active_bytes'] for r in rows),
            'maximum_aux_physical_peak_bytes':last_aux,'maximum_physical_peak_bytes':max(r['physical_peak_bytes'] for r in rows),
            'maximum_physical_current_bytes':max(r['physical_current_bytes'] for r in rows)}


def source_pins(owner,plan):
    pins=plan['execution']['pins'];before,after=read(owner/'pins-before.json'),read(owner/'pins-after.json')
    need(set(pins)==set(before)==set(after),'before/after pin completeness')
    for p,h in pins.items():
        path=Path(p);need(path.is_absolute() and re.fullmatch('[0-9a-f]{64}',h),'absolute source pin')
        a,b=before[p],after[p];copied=path.suffix in ('.py','.json','.md','.h','.cpp','.txt')
        need(a['sha256']==b['sha256']==h and same(a['stat'],b['stat'])
             and a['source_copied'] is copied and b['source_copied'] is False,'pin stat/SHA consistency')
        actual=owner/'frozen-source'/p.lstrip('/') if copied else path
        need(actual.stat().st_size==a['stat']['size'] and sha(actual)==h,'actual frozen source/runtime/payload hash')
    need(pins[str(HERE/'candidate.py')]==CANDIDATE_SHA and pins[str(HERE/'screen.py')]==SCREEN_SHA,
         'reviewed candidate/screen execution pins')
    return pins


def resources(owner,complete,launch,audit):
    rows=read(owner/'resources.json');pid,ppid=launch['pid'],launch['ppid']
    need(len(rows)==complete['samples'] and [r['label'] for r in rows if r['label']!='running']==['preflight','after-child','postflight'],'resource lifecycle coverage')
    first=rows[0];minimum_pressure=2;count=0;measured=[];races=0
    need(rows[0]['label']=='preflight' and rows[-2]['label']=='after-child' and rows[-1]['label']=='postflight','resource order')
    for i,row in enumerate(rows):
        p=row['pressure_level'];pressure=int(p['text'].strip())
        need(p['returncode']==0 and type(row['pressure_level_value']) is int and pressure==row['pressure_level_value']
             and pressure in (1,2) and pressure<=minimum_pressure,'pressure unknown/worse/rebounded')
        minimum_pressure=pressure
        vm=row['vm_stat'];page=int(re.search(r'page size of (\d+) bytes',vm['text'])[1]);free=int(re.search(r'^Pages free:\s*(\d+)\.',vm['text'],re.M)[1])*page
        need(vm['returncode']==0 and free==row['real_free_bytes'] and free>=50<<30,'real free-page budget')
        for name,value in row['vm_counters'].items():need(type(value) is int and int(re.search('^'+name+r':\s*(\d+)',vm['text'],re.M)[1])==value,'VM raw counter mismatch')
        need(row['vm_counters']['Swapouts']==first['vm_counters']['Swapouts'],'Swapouts growth/reset')
        used=float(re.search(r'used = ([\d.]+)M',row['swap']['text'])[1])
        need(row['swap']['returncode']==0 and 0<=used==row['swap_used_mb']<=first['swap_used_mb']+16,'swap growth')
        power,thermal=row['power_metadata'],row['processinfo_thermal_metadata']
        need(power['status']==thermal['status']=='known' and power['source']=='AC Power'
             and power['raw']['returncode']==thermal['raw']['returncode']==0 and 'AC Power' in power['raw']['stdout']
             and type(thermal['code']) is int and thermal['code'] in (0,1) and thermal['raw']['stdout'].strip()==str(thermal['code']),'AC/thermal raw gate')
        need(row['native_process_inventory_ok'] is True and row['process_inventory_ok'] is True and row['authorized_suspended_pids']==[74384],'inventory/suspension known')
        native=row['native_inference_processes'];service=[v for v in native if v['pid']==74384]
        need(len(service)==1 and service[0]['authorized_suspended'] is True and 'T' in service[0]['state'],'external service lease T missing')
        running=row['label']=='running';owned=pid if running else None
        for proc in native+row['known_inference_processes']:
            need(proc['pid'] in (74384,owned),'foreign inference process')
            if proc['pid']==owned:need(proc['owned'] is True and proc['ppid']==ppid,'direct ownership')
        if running:
            need(row['owned_model_pid']==pid and row['physical_receipt']==f'footprint-{count:03d}.json','physical target binding')
            receipt=read(owner/row['physical_receipt']);parsed=audit.owner_footprint(owner,receipt,pid,ppid)
            if parsed is None:
                # This flag was sampled BEFORE footprint. The child may exit
                # during the probe; the probe's later poll plus strict release
                # are the terminal proof, not a fabricated earlier observation.
                need(type(row['terminal_exit0_observed']) is bool and receipt['child_poll_after_probe']==0,
                     'exit race lacks later terminal evidence');races+=1
            else:measured.append(parsed)
            if not row['terminal_exit0_observed']:
                exact=row['exact_owned_pid_metadata']
                need(exact['status']=='known' and exact['pid']==pid and exact['ppid']==ppid and len([v for v in native if v['pid']==pid])==1,'live child inventory/identity')
            count+=1
        if i:need(row['monotonic_s']>rows[i-1]['monotonic_s'],'resource time order')
    need(count>0 and count==complete['physical_samples'] and len(list(owner.glob('footprint-[0-9][0-9][0-9].json')))==count,'physical sample coverage')
    return dict(samples=len(rows),physical_samples=count,physical_measured=len(measured),terminal_races_not_measurements=races,
                first_pressure=first['pressure_level_value'],last_pressure=rows[-1]['pressure_level_value'],swapouts_growth=0,
                minimum_free_bytes=min(r['real_free_bytes'] for r in rows),
                maximum_swap_growth_mib=max(r['swap_used_mb'] for r in rows)-first['swap_used_mb'])


def lease_check(lease,owner,owner_pid,plan,plan_sha):
    data,resume,job=read(lease/'lease.json'),read(lease/'resume.json'),read(lease/'job-0.json')
    need(data['closed'] is True and len(data['jobs'])==1 and job['command']==data['jobs'][0]
         and data['child_pid']==job['pid']==owner_pid and job['returncode']==0 and job['waited'] is True,'closed external lease owner exit')
    command=job['command'];need(len(command)==10 and command[1:4]==['-I','-B',str(HERE/'owner.py')],'fixed owner command prefix')
    need(str(Path(command[0]).resolve()) in {str(Path(p).resolve()) for p in plan['execution']['pins']},'owner Python pin')
    flags=dict(zip(command[4::2],command[5::2]))
    need(set(flags)=={'--plan','--plan-sha256','--out'} and flags['--plan-sha256']==plan_sha and flags['--out']==str(owner)
         and sha(flags['--plan'])==plan_sha and same(read(flags['--plan']),plan),'external job exact plan/out binding')
    need(data['service_pid']==74384 and data['service_identity'].endswith('omlx-server') and data['paused_at']<resume['time']<=data['deadline']
         and resume['identity_match'] is True and 'T' in resume['state_before'] and resume['state_after'] and 'T' not in resume['state_after'],'original external service restored')
    return resume


def audit_run(owner,lease,plan_sha,progress):
    audit=helpers()
    need(re.fullmatch('[0-9a-f]{64}',plan_sha),'explicit frozen plan SHA')
    result=read(owner/'complete.json');plan=read(owner/'plan.json');child=owner/'child';out=child/'screen'
    progress['recorded_owner_status']=result['status']
    need(result['schema']=='E027-shared-W-owner-v1' and result['status']=='PASS' and result['failures']==[]
         and result['performance_qualified'] is False and result['raw_audit']=='PENDING' and 0<result['elapsed_wall_s']<=480,'owner contract failed/incomplete')
    need(sha(owner/'plan.json')==plan_sha and plan['schema']=='E027-shared-W-numeric-v1'
         and same([plan['expected_cases'],plan['expected_comparisons']],[33,66])
         and same(plan['resource_policy'],audit.POLICY) and same(plan['execution']['environment'],audit.ENV),'frozen numerical plan')
    pins=source_pins(owner,plan)
    launch,release=read(child/'launch.json'),read(child/'release.json');pid,ppid=launch['pid'],result['owner_pid']
    expected_command=[plan['execution']['python'],'-I','-B',plan['execution']['screen'],'--run','--out',str(out),
                      '--bundle-manifest',plan['bundle']['path'],'--bundle-sha256',plan['bundle']['sha256']]
    need(launch['ppid']==ppid and launch['command']==expected_command and same(launch['environment'],audit.ENV) and launch['arm_seconds']==120
         and 0<launch['arm_deadline_monotonic']-launch['spawned_monotonic']<=120,'direct child command/budget')
    need(release['pid']==pid and release['returncode']==0 and release['waited_reaped'] is True and release['strict_absence'] is True
         and release['cleanup_notes']==[] and release['absence_probe']['returncode']==1 and not release['absence_probe']['stdout'].strip(),'child zero exit/reap/strict absence')
    complete=read(out/'complete.json');progress['recorded_child_status']=complete['status']
    need(same([complete['status'],complete['pid'],complete['ppid'],complete['device_run']],['PASS',pid,ppid,True])
         and complete['source_pins_rechecked'] is True and complete['bundle_rechecked'] is True
         and complete['performance_qualified'] is False and complete['full_model_qualified'] is False
         and 0<complete['elapsed_wall_s']<=120,'child complete identity/scope')
    need(read(owner/'child-receipt.json')['complete_sha256']==sha(out/'complete.json'),'owner child summary binding')
    required=dict(schema='M5-shared-w-m1-screen-v1',cases=expected_cases(),expected_cases=33,expected_comparisons=66,
                  expected_phases=68,expected_layouts=110,cache_limit_bytes=0,mlx_observation_budget_bytes=1<<30,
                  physical_observation_budget_bytes=2<<30,child_alarm_seconds=120,physical_reading_semantics=PHYSICAL_SEMANTICS,
                  quantization=dict(group_size=16,bits=4,mode='nvfp4'),full_model_qualified=False,performance_qualified=False)
    need(same({k:complete['contract'][k] for k in required},required),'fixed child workload contract')
    published=ROOT/'docs/code-versions/44-shared-w-m1';support=HERE.parent/'conv_tail_owned_copy'
    required_sources={str(published/'kernel124'/n) for n in ('kernel_header.metal','control_body.metal','kernel_body.metal','shared_impl.metal','LICENSE-MLX.txt')}
    required_sources|={str(published/'real127/layout_probe.metal'),str(support/'layer_weights.py'),str(support/'layer_screen.py'),str(HERE/'candidate.py'),str(HERE/'screen.py')}
    need(set(complete['sources'])==required_sources,'exact 10 loaded source identities required')
    for p,h in complete['sources'].items():
        need(pins.get(p)==h and sha(out/'frozen-source'/Path(p).relative_to(ROOT))==h,'actual child source pins')
    need(complete['sources'].get(str(HERE/'candidate.py'))==CANDIDATE_SHA and complete['sources'].get(str(HERE/'screen.py'))==SCREEN_SHA
         and complete['sources'].get(str(ROOT/'docs/code-versions/44-shared-w-m1/real127/layout_probe.metal'))==LAYOUT_SHA,'reviewed implementation/layout source binding')
    runtime=complete['runtime']
    need(runtime['mlx_version']=='0.32.3' and runtime['device']['device_name']=='Apple M5 Ultra' and runtime['device']['architecture']=='applegpu_g17d','M5 actual runtime identity')
    for key in ('mlx_file','numpy_file'):need(str(Path(runtime[key]).resolve()) in {str(Path(p).resolve()) for p in pins},'runtime file unpinned')
    parameters,bundle=audit.audit_bundle(Path(plan['bundle']['path']),plan['bundle']['sha256'])
    need(complete['bundle_manifest_sha256']==bundle['manifest_sha256'] and complete['bundle_sha256']==bundle['bundle_sha256']
         and same(read(out/'bundle-manifest.json'),read(plan['bundle']['path'])),'raw real-weight bundle identity')
    cache={};banks=audit_banks(out,complete,audit,cache,progress)
    cases=complete['cases'];checks=complete['checks']
    need([r['case'] for r in cases]==[r['case'] for r in expected_cases()] and len(checks)==66
         and same(checks,read(out/'checks.json')),'33 case/66 comparison raw coverage')
    summaries=[]
    for i,(row,meta) in enumerate(zip(cases,expected_cases())):
        summaries.append(audit_case(out,row,meta,checks[2*i:2*i+2],audit,cache,parameters,banks[meta['id']],progress))
    need(complete['descriptor_count']==110 and len(list(out.glob('banks/*/*-layout-*.bin')))+len(list(out.glob('cases/*/x-layout-*.bin')))==110,'110 actual GPU descriptor coverage')
    memory=phases(out,complete,pid,ppid,launch,audit)
    resource=resources(owner,result,launch,audit)
    restored=lease_check(lease,owner,ppid,plan,plan_sha)
    return dict(status='PASS',qualified=True,numerical_scope='33 local same-input matrices,66 comparisons; 27 synthetic and6 real-weight/synthetic-X',
                performance_qualified=False,full_model_qualified=False,cases=summaries,comparisons=66,prepared_banks=11,
                rows_per_implementation=sum(r['T'] for r in expected_cases()),bf16_elements_per_implementation=sum(r['T']*r['N'] for r in expected_cases()),
                readonly_case_operand_records=99,raw_files_finite_sha_checked=len(cache),
                actual_gpu_descriptors=110,memory=memory,resources=resource,release=release,service_restore=restored,bundle=bundle,
                source_pins_checked=len(pins),plan_sha256=plan_sha,
                limits='No propagated model/state/HTTP qualification. Random PRNG values are not independently regenerated; saved finite raw operands and pinned generator provide scope. Structured banks are independently regenerated. Full bank W/scales before/after bytes plus every-case readonly hashes are checked; no transient write history is claimed. Callable counts are host witnesses, not GPU compile counts or performance.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner',type=Path,required=True);p.add_argument('--lease',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv);need(not args.out.exists(),'refuse to overwrite audit evidence')
    progress={}
    try:result=audit_run(args.owner.resolve(),args.lease.resolve(),args.plan_sha256,progress)
    except Exception as error:result=dict(status='FAIL',qualified=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),partial=progress)
    result.update(audit_source_sha256=sha(__file__),independent_primitives_sha256=PRIMITIVES_SHA,performance_qualified=False,full_model_qualified=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','qualified','performance_qualified')}))
    return int(result['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
