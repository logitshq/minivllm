"""Offline, exact E028 v2 partial-run verification. Stdlib; no measured imports.

Reads every saved safetensors file, checks original bits/finite values, and
independently reconstructs the two contiguous input cases and saved B2 output.
The resource-aborted run remains unqualified even when all retained bytes pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parent
PLAN_SHA = '26ebdcf2ee6fb09b0c86a15b3bb87a7d221ba67811a4ccf382ae150d3b495b42'
CHILD, OWNER = 58999, 58987
MASK_SIGN = bytes(v & 127 for v in range(256))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def unique(pairs):
    result = {}
    for k, v in pairs:
        need(k not in result, 'duplicate JSON key: ' + k)
        result[k] = v
    return result


def same_bytes(expected, actual, label):
    if expected != actual:
        first = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), min(len(expected), len(actual)))
        raise ValueError(f'{label}: first differing byte {first}, lengths {len(expected)}/{len(actual)}')


def finite(raw, dtype):
    width = 2 if dtype == 'BF16' else 4
    # BF16/F32 exponent is all ones iff high byte sans sign is 127 and
    # the next lower byte's high bit is set. C-backed slicing scans all words.
    high = raw[width-1::width].translate(MASK_SIGN)
    pos = high.find(b'\x7f')
    while pos >= 0:
        if raw[pos*width+width-2] & 128:
            return False
        pos = high.find(b'\x7f', pos+1)
    return True


def tensors(screen, record):
    path = screen / record['path']
    need(path.is_relative_to(screen) and '..' not in Path(record['path']).parts, 'snapshot path')
    need(sha(path) == record['sha256'] and record['finite'] is True, 'snapshot SHA/finite declaration')
    data = path.read_bytes()
    need(len(data) >= 8, 'safetensors prefix')
    size = struct.unpack('<Q', data[:8])[0]
    need(0 < size <= 1 << 20 and 8+size <= len(data), 'header bound')
    header = json.loads(data[8:8+size], object_pairs_hook=unique)
    payload = memoryview(data)[8+size:]
    result, spans = {}, []
    for key, item in header.items():
        if key == '__metadata__':
            continue
        dtype, shape, (lo, hi) = item['dtype'], item['shape'], item['data_offsets']
        need(dtype in ('BF16','F32') and all(type(n) is int and n > 0 for n in shape), 'dtype/shape')
        width = 2 if dtype == 'BF16' else 4
        need(type(lo) is int and type(hi) is int and 0 <= lo <= hi <= len(payload)
             and hi-lo == width*math.prod(shape), 'payload bounds')
        raw = bytes(payload[lo:hi]); meta = record['tensors'][key]
        need(meta == dict(shape=shape,dtype='mlx.core.'+('bfloat16' if dtype=='BF16' else 'float32'),
                         encoding=dtype,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),finite=True), 'tensor metadata/SHA')
        need(finite(raw,dtype), 'nonfinite raw tensor: '+key)
        result[key] = (dtype, shape, raw); spans.append((lo,hi))
    cursor = 0
    for lo, hi in sorted(spans):
        need(lo == cursor, 'overlap/gap'); cursor = hi
    need(cursor == len(payload) and set(result) == set(record['tensors']), 'trailing bytes/keys')
    return result


def scalar(value, dtype):
    word = struct.pack('<f', value)
    return word[2:] if dtype == 'BF16' else word


def patterned(dtype, shape, seed):
    # i = 127*q+r => (i+i//127+seed)%128 == (r+seed)%128.
    # A 127-word cycle therefore reproduces every original source bit.
    words = []
    for i in range(127):
        low, sign = (i+seed)%128, (i+seed)&1
        words.append(struct.pack('<H',0x3e00|low|(sign<<15)) if dtype=='BF16'
                     else struct.pack('<I',0x3e800000|(low<<16)|(sign<<31)))
    cycle = b''.join(words); count = math.prod(shape); width = 2 if dtype=='BF16' else 4
    return cycle*(count//127) + cycle[:(count%127)*width]


def expected_sources(B):
    specs = {'old_K':('BF16',[B,4,1536,256],11),'old_V':('BF16',[B,4,1536,256],23),
             'old_conv':('BF16',[B,3,10240],37),'old_S':('F32',[B,48,128,128],41)}
    for i in range(2):
        for field, dtype, shape, seed in [('K','BF16',[1,4,1024,256],53),('V','BF16',[1,4,1024,256],67),
                                         ('conv','BF16',[3,10240],79),('S','F32',[48,128,128],97)]:
            specs[f'new{i}_{field}'] = (dtype,shape,seed+17*i)
    return specs


def output_bytes(field, source):
    raw = source['old_'+field][2]
    for i, offset in enumerate((935,954)):
        stage = source[f'new{i}_{field}'][2]
        if field in ('K','V'):
            # Per-head time truncation followed by zero padding to L1536.
            head_stride = 1024*256*2
            stage = b''.join(stage[h*head_stride:h*head_stride+offset*256*2] + bytes((1536-offset)*256*2) for h in range(4))
        raw += stage
    return raw


def numeric(screen, child):
    rows = child['cases']
    need([(r['id'],r['status']) for r in rows] == [('identity-2-to-4','PASS'),('identity-4-to-6','FAIL')], 'case coverage')
    paths, array_count, payload_bytes = set(), 0, 0
    for row, B in zip(rows,(2,4)):
        name, snaps = row['id'], row['snapshots']
        need(row == read(screen/'cases'/name/'case.json'), 'case copy')
        labels = [f'{arm}-inputs-{suffix}' for arm in ('OFF','ON') for suffix in ('before','after-rebuild')]
        if B == 2:
            labels += [f'{arm}-{suffix}' for arm in ('OFF','ON') for suffix in
                       ('outputs-first-eval','inputs-after-source-write','outputs-after-source-write')]
        else:
            labels += ['OFF-inputs-on-error','ON-inputs-on-error']
        need(set(snaps) == set(labels), 'snapshot coverage')
        specs = expected_sources(B)
        baseline = tensors(screen,snaps['OFF-inputs-before'])
        for arm in ('OFF','ON'):
            path = 'fallback' if arm=='OFF' else 'identity_refill'
            reason = 'disabled' if arm=='OFF' else 'identity_keep_unchanged_capacity'
            need(row['witness'][arm] == dict(path=path,reason=reason,counters=dict(identity_refill=int(arm=='ON'),fallback=int(arm=='OFF'))), 'branch witness')
            shapes = {'K':[B+2,4,1536,256],'V':[B+2,4,1536,256],'conv':[B+2,3,10240],'S':[B+2,48,128,128]}
            need(row['metadata'][arm] == dict(req_ids=[f'old-{i}' for i in range(B)]+['new-0','new-1'],
                 lens=[920+7*i for i in range(B)]+[935,954],shapes=shapes), 'IDs/lens/shapes')
        own = row['independent_sources']; one = 33816576 if B==2 else 52805632
        need(own == dict(OFF_backing_bytes=one,ON_backing_bytes=one,union_backing_bytes=2*one,disjoint=True), 'independent source backing receipt')
        for label, record in snaps.items():
            need(record['path'] == f'cases/{name}/{label}.safetensors' and record['path'] not in paths, 'snapshot naming')
            paths.add(record['path']); saved = tensors(screen,record)
            array_count += len(saved); payload_bytes += sum(len(v[2]) for v in saved.values())
            if 'inputs-' in label:
                need(set(saved) == set(specs), 'complete source tensor set')
                for key, (dtype,shape,seed) in specs.items():
                    need(saved[key][:2] == (dtype,shape), 'source shape/dtype')
                    expected = patterned(dtype,shape,seed)
                    if label.endswith('after-source-write'):
                        first = scalar(-16,dtype); expected = first+expected[len(first):]
                    same_bytes(expected,saved[key][2],name+'/'+label+'/'+key)
            else:
                need(set(saved) == {'K','V','conv','S'}, 'complete output tensor set')
                for field, (dtype,shape,raw) in saved.items():
                    need(shape == shapes[field] and dtype == ('F32' if field=='S' else 'BF16'), 'output shape/dtype')
                    same_bytes(output_bytes(field,baseline),raw,name+'/'+label+'/'+field)
        del baseline
    need(paths == {p.relative_to(screen).as_posix() for p in screen.rglob('*.safetensors')}, 'no unaccounted raw snapshots')
    checks = child['checks']; need(checks == read(screen/'checks.json') and len(checks)==11, 'comparison copies/count')
    expected_labels = ['initial-inputs-OFF-vs-ON','OFF-rebuild-inputs-readonly','ON-rebuild-inputs-readonly','rebuild-outputs-OFF-vs-ON',
                       'OFF-source-write-exact','OFF-output-stable-after-source-write','ON-source-write-exact','ON-output-stable-after-source-write']
    tensor_pairs = compare_bytes = 0
    for i, check in enumerate(checks):
        case = rows[0 if i<8 else 1]
        need(check['index']==i and check['case']==case['id'] and check['label']==expected_labels[i if i<8 else i-8], 'ordered comparison coverage')
        mutation = -16.0 if i in (4,6) else None
        need(check['status']=='PASS' and check['finite'] is True and check['bytes_equal'] is True
             and check['differences']==[] and check['expected_first_scalar_write']==mutation, 'comparison declared contract')
        pair = check['rawpair']
        for rec in pair.values():
            need(rec in case['snapshots'].values(), 'comparison snapshot binding')
        a,b = (tensors(screen,pair[k]) for k in ('reference','candidate'))
        need(set(a)==set(b),'comparison keys')
        for key in a:
            need(a[key][:2]==b[key][:2],'comparison shape/dtype')
            expected = a[key][2]
            if mutation is not None:
                word=scalar(mutation,a[key][0]); expected=word+expected[len(word):]
            same_bytes(expected,b[key][2],f'comparison{i}/{key}')
            tensor_pairs += 1; compare_bytes += len(expected)
    events = [json.loads(line) for line in (screen/'events.jsonl').read_text().splitlines()]
    need(len(events)==11,'comparison event count')
    for event,check in zip(events,checks):
        need({k:v for k,v in event.items() if k not in ('event','elapsed_wall_s')}==check and event['event']=='comparison','comparison event copy')
    need(not (screen/'first-difference.json').exists(),'no numerical first difference')
    return dict(snapshot_paths=len(paths),saved_tensor_records=array_count,saved_tensor_payload_bytes=payload_bytes,
                comparison_groups=11,tensor_pairs=tensor_pairs,compared_payload_bytes=compare_bytes,
                independent_input_patterns=True,independent_B2_rebuild_output_bytes=True,
                first_scalar_write_groups=2,full_matrix_qualified=False)


def physical(raw, pid):
    need(raw['unit']=='byte' and raw['bytes per unit']==1 and raw['errors']==raw['warnings']==[],'physical units/errors')
    need(len(raw['processes'])==1 and raw['processes'][0]['pid']==pid,'physical PID')
    p=raw['processes'][0]; top,cur,peak=p['footprint'],p['auxiliary']['phys_footprint'],p['auxiliary']['phys_footprint_peak']
    need(all(type(v) is int and 0<=v<=2<<30 for v in (top,cur,peak)) and peak>=cur,'physical budget')
    return dict(pid=pid,current_bytes=max(top,cur),peak_bytes=max(top,peak),top_bytes=top,aux_current_bytes=cur,aux_peak_bytes=peak)


def verify(runtime_pins=False):
    raw=ROOT/'raw';owner=raw/'identity-refill-v2-owner';cpu=raw/'identity-refill-v2-cpu-prepare';inp=raw/'identity-refill-v2-inputs'
    lease_dir=raw/'identity-refill-v2-service-lease';screen=owner/'child/screen'
    plan=read(inp/'plan.json');reg=read(inp/'registration.json')
    need(sha(inp/'plan.json')==reg['plan_sha256']==PLAN_SHA and sha(inp/'jobs.json')==reg['jobs_sha256'],'registration SHA')
    need(read(owner/'plan.json')==read(cpu/'plan.json')==plan,'plan copies')
    pins=plan['execution']['pins'];need(len(pins)==reg['pins']==573,'pin count');frozen=0
    for folder,names in ((owner,['pins-before.json','pins-after.json']),(cpu,['pins-before.json'])):
        records=[read(folder/n) for n in names]
        for record in records:
            need(set(record)==set(pins),'pin keys')
            for p,h in pins.items():need(record[p]['sha256']==h,'pin receipt SHA')
        for p,h in pins.items():
            if records[0][p]['source_copied']:
                need(sha(folder/'frozen-source'/p.lstrip('/'))==h,'frozen source SHA');frozen+=1
            if len(records)==2:need(records[0][p]['stat']==records[1][p]['stat'],'pin stat drift')
    runtime_bytes=0
    if runtime_pins:
        for p,h in pins.items():need(sha(p)==h,'runtime SHA');runtime_bytes+=Path(p).stat().st_size
    child=read(screen/'complete.json');complete=read(owner/'complete.json');launch=read(owner/'child/launch.json');release=read(owner/'child/release.json')
    need(child['status']==complete['status']=='FAIL' and child['error']['type']=='ValueError'
         and child['error']['message']=='MLX 1GiB observation budget exceeded','resource abort')
    need(child['pid']==launch['pid']==release['pid']==CHILD and child['ppid']==launch['ppid']==complete['owner_pid']==OWNER,'PID binding')
    need(child['contract']==plan['contract']==read(screen/'plan.json') and child['source_pins_rechecked'] is False,'partial contract')
    need(child['device_run'] is True and child['device']==dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3'),'device identity')
    for p,h in child['sources'].items():need(pins[p]==h,'screen/source pin')
    need(launch['command']==[plan['execution']['python'],'-I','-B',plan['execution']['screen'],'--run','--out',reg['registered_paths']['owner']+'/child/screen']
         and launch['environment']==plan['execution']['environment'],'child launch')
    need(release['returncode']==1 and release['waited_reaped'] is True and release['strict_absence'] is True
         and release['absence_probe']==dict(returncode=1,stdout='',stderr=''),'child reap')
    numbers=numeric(screen,child)
    phases=[json.loads(line) for line in (screen/'phases.jsonl').read_text().splitlines()]
    expected=[('global','initial')]+[('identity-2-to-4',p) for p in ('before_case','after_rebuild_graph','after_outputs_eval','after_source_writes','after_case_release')]
    expected += [('identity-4-to-6','before_case'),('identity-4-to-6','after_rebuild_graph')]
    need([(r['case'],r['phase']) for r in phases]==expected and len(child['physical_receipts'])==8,'partial phase coverage')
    last_elapsed=last_peak=0
    for i,row in enumerate(phases):
        rec=row['physical_receipt'];need(rec==child['physical_receipts'][i] and row['index']==i and row['pid']==CHILD and row['ppid']==OWNER,'phase identity')
        need(sha(screen/rec['path'])==rec['sha256'],'phase physical SHA'); parsed=physical(read(screen/rec['path']),CHILD)
        for key in ('top','aux_current','aux_peak','current','peak'):
            need(row['physical_'+key+'_bytes']==parsed[key+'_bytes'],'phase physical parse')
        need(last_elapsed<=row['elapsed_s']<=child['elapsed_wall_s'] and parsed['aux_peak_bytes']>=last_peak,'phase sequence')
        last_elapsed,last_peak=row['elapsed_s'],parsed['aux_peak_bytes']
        passed=row['peak_bytes']<=1<<30 and row['active_bytes']+row['cache_bytes']<=1<<30
        need(passed==(i<7) and row['status']==('PASS' if passed else 'FAIL'),'resource gate recompute')
        cmd=read(screen/f'physical-{i:03d}.command.json')
        need(cmd['returncode']==0 and cmd['command']==['/usr/bin/footprint','-p',str(CHILD),'--noCategories','-f','bytes','-j',reg['registered_paths']['owner']+f'/child/screen/physical-{i:03d}.raw.json'],'self command')
    need(phases[-1]['peak_bytes']==1131872976 and phases[-1]['active_bytes']==105611296
         and phases[-1]['physical_peak_bytes']==1109951736,'failed phase readings')
    need(complete['physical_samples']==2,'owner physical count')
    for i in range(2):
        row=read(owner/f'footprint-{i:03d}.json')
        need(row['pid']==CHILD and row['owner_pid']==OWNER and row['returncode']==0 and row['status']=='measured'
             and row['physical_pass'] is True and sha(owner/row['raw_file'])==row['raw_sha256'],'owner physical receipt')
        need(row['parsed']==physical(read(owner/row['raw_file']),CHILD),'owner physical parse')
    resources=read(owner/'resources.json');need([r['label'] for r in resources]==['preflight','running','running','postflight'] and complete['samples']==4,'resources')
    for row in resources:
        need(row['pressure_level_value']==2 and row['pressure_level']['returncode']==0 and row['pressure_level']['text'].strip()=='2','pressure')
        vm,swap=row['vm_stat'],row['swap'];need(vm['returncode']==swap['returncode']==0,'resource commands')
        free=int(re.search(r'page size of (\d+) bytes',vm['text'])[1])*int(re.search(r'^Pages free:\s*(\d+)\.',vm['text'],re.M)[1])
        sw=int(re.search(r'^Swapouts:\s*(\d+)\.',vm['text'],re.M)[1]);used=float(re.search(r'used = ([\d.]+)M',swap['text'])[1])
        need(free==row['real_free_bytes'] and free>=50<<30 and sw==row['vm_counters']['Swapouts']==316235 and used==row['swap_used_mb']==445.06,'free/swap')
        need(row['native_process_inventory_ok'] is True and row['process_inventory_ok'] is True and row['authorized_suspended_pids']==[74384],'inventory')
        inventory=row['native_inference_processes']+row['known_inference_processes'];need(all(p['pid'] in (74384,CHILD) for p in inventory),'foreign inference')
        service=[p for p in inventory if p['pid']==74384]
        need(len(service)==1 and service[0]['state']=='T' and service[0]['authorized_suspended'] is True,'service paused')
        power,thermal=row['power_metadata'],row['processinfo_thermal_metadata']
        need(power['source']=='AC Power' and power['raw']['returncode']==0 and 'AC Power' in power['raw']['stdout'],'AC')
        need(thermal['code']==0 and thermal['raw']['returncode']==0 and thermal['raw']['stdout']=='0','thermal')
    lease,job,resume=(read(lease_dir/n) for n in ('lease.json','job-0.json','resume.json'))
    command=[plan['execution']['python'],'-I','-B',plan['execution']['screen'].replace('/screen_v2.py','/owner_v2.py'),
             '--plan',reg['registered_paths']['inputs']+'/plan.json','--plan-sha256',PLAN_SHA,'--out',reg['registered_paths']['owner']]
    need(read(inp/'jobs.json')==lease['jobs']==[job['command']]==[command],'lease/owner/plan binding')
    need(lease['child_pid']==job['pid']==OWNER and job['returncode']==1 and job['waited'] is True,'owner waited')
    need(lease['closed'] is True and lease['service_pid']==74384 and lease['service_identity']=='Fri Oct 16 08:01:58 2026     omlx-server','lease service identity')
    need(resume['identity_match'] is True and resume['state_before']=='T' and resume['state_after']=='S' and lease['paused_at']<resume['time']<=lease['deadline'],'restore receipt')
    need(child['performance_qualified'] is False and child['full_model_qualified'] is False and complete['performance_qualified'] is False,'scope')
    return dict(audit_status='PASS_EVIDENCE_VERIFICATION',experiment_status='RESOURCE_ABORT_INCOMPLETE_NOT_QUALIFIED',
                numerical_qualified=False,performance_qualified=False,full_model_qualified=False,verifier_sha256=sha(__file__),
                plan_sha256=PLAN_SHA,pins=len(pins),archived_source_copies_verified=frozen,
                runtime_pins_rehashed=len(pins) if runtime_pins else 0,runtime_pin_bytes=runtime_bytes,
                completed_cases=1,partial_cases=1,**numbers,physical_phases=8,failed_phase_index=7,
                failed_mlx_peak_bytes=1131872976,mlx_budget_bytes=1<<30,failed_active_bytes=105611296,
                physical_peak_bytes=1109951736,physical_budget_bytes=2<<30,resources=4,
                child_pid=CHILD,owner_pid=OWNER,child_returncode=1,child_reaped=True,owner_waited=True,
                service_identity_restored=True,restoration_scope='Pinned source + saved original-identity T-to-S receipt; no new live probe',
                source_pins=child['sources'],lease_source_sha256=pins[next(p for p in pins if p.endswith('/suspended_service.py'))])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runtime-pins',action='store_true');args=ap.parse_args()
    print(json.dumps(verify(args.runtime_pins),indent=2,allow_nan=False))
