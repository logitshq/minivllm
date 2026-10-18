"""Offline original-byte audit of twelve fixed teacher-forced forward cases.

No MLX/numpy imports. Reuses the frozen bounded safetensors reader and exact
unsigned-word comparator. PASS never qualifies capture/features, accepted
prefixes, rollback, scheduler/ahead, HTTP or performance. Missing/partial runs
remain unqualified; available complete failed outputs can locate a bit mismatch.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
RAW = HERE.parent/'long_decode_isolation/audit_raw.py'
RAW_SHA = 'b3bffc7bd261b9b347b5d299d965273ff2406a92c27576747867aa515be62261'
E024_SHA = '5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
E023_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
CORE = HERE.parent/'long_decode_graph/model-source-pins.json'
CORE_SHA = '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5'
WIDTHS, STARTS = (2, 4, 8), (1, 129)
DTYPES = {'BF16':'mlx.core.bfloat16', 'F32':'mlx.core.float32', 'I32':'mlx.core.int32'}


def need(ok, message):
    if not ok: raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'duplicate JSON key: '+key); result[key] = value
    return result


def parse(data):
    def bad(value): raise ValueError('nonfinite JSON: '+value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=bad)


def read(path): return parse(Path(path).read_bytes())
def same(a, b, label):
    need(json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False), label)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()
def record(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def finite(value): return type(value) in (int, float) and math.isfinite(value) and value >= 0


def raw_reader():
    need(sha(RAW) == RAW_SHA, 'frozen original-byte reader source')
    spec = importlib.util.spec_from_file_location('_exact_bridge_raw_reader', RAW)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def expected_contract():
    return dict(schema='Exact-verifier-bridge-contract-v1', enabled_by_default=False,
        widths=[2,4,8], decode_starts=[1,129], expected_cases=12,
        fixtures='shortest and longest original D512, ties by fixture_id; both starts for each',
        oracle='original runner.decode_model, one actual token/position per call',
        candidate='SpecRunner target expression: snapshot, model(block, prefill, capture=())',
        inputs='fixed IDs generated only by an independent original C1 continuation',
        normal_eos=True, cap=512, early_eos='INCOMPLETE; no fabricated continuation',
        configuration=dict(max_seqs=1,max_len=2048,max_batched_tokens=16384,cache_limit_gib=2,
            compile_decode=True,fuse_decode=True,gates_compile=True,norm_reuse=False),
        arrays='all T full-vocabulary logits and all128 final state arrays, including padded KV backing',
        parameter_check='all original-dtype parameter values; handle IDs recorded, not compared',
        excluded=['drafter/features','accept/rollback','intermediate prefix states','scheduler/ahead',
                  'E024 graph','E028 refill','GDN prework','HTTP','performance'],
        rollback_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False)


def ref_json(ref, expected_sha=None):
    need(set(ref) == {'path','sha256'} and Path(ref['path']).is_absolute(), 'absolute exact JSON ref')
    need(sha(ref['path']) == ref['sha256'] and (expected_sha is None or ref['sha256'] == expected_sha), 'JSON ref SHA')
    return read(ref['path'])


def identities(child, complete, plan_sha):
    plan = read(child/'plan.json')
    need(sha(child/'plan.json') == plan_sha == complete['plan_sha256'], 'actual registered child plan SHA')
    need(set(plan) == {'schema','qualifier_sha256','e024_plan','deadline_s'}
         and plan['schema'] == 'Exact-verifier-bridge-plan-v1', 'exact bridge plan')
    need(finite(plan['deadline_s']) and 0 < plan['deadline_s'] <= 1800, 'bounded deadline')
    need(plan['qualifier_sha256'] == sha(HERE/'qualify.py'), 'plan-pinned actual qualifier source')
    old = ref_json(plan['e024_plan'], E024_SHA)
    inherited = ref_json(old['e023_plan'], E023_SHA)
    same(read(child/'inherited-e024-plan.json'), old, 'retained inherited E024 plan')
    same(read(child/'inherited-e023-plan.json'), inherited, 'retained inherited E023 plan')
    sources = {str(HERE/'qualify.py'):plan['qualifier_sha256'],
        str(HERE.parent/'long_decode_isolation/qualify.py'):'9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e',
        str(HERE.parent/'long_decode_graph/qualify.py'):'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
        str(CORE):CORE_SHA,
        old['e023_qualifier']['path']:'26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'}
    same(complete['sources'], sources, 'complete exact executable source map')
    need(all(sha(path) == digest for path,digest in sources.items()), 'small source pins changed')
    source = ref_json(inherited['source_manifest'])
    same(source['files'], read(CORE)['files'], 'complete fixed model source manifest')
    root = Path(inherited['package_root'])/'tutorial_code/miniengine'
    need(all(sha(root/name) == digest for name,digest in source['files'].items()), 'actual frozen model source files')
    manifest = ref_json(inherited['model_manifest'])
    fixtures = ref_json(inherited['fixtures'])
    need(sha(child/'fixtures.json') == inherited['fixtures']['sha256'], 'retained exact original fixture bytes')
    need(manifest['schema'] == 'E023-model-v1'
         and Path(manifest['model']).resolve() == Path(inherited['model']).resolve()
         and manifest['revision'] == Path(inherited['model']).resolve().name
         and Path(fixtures['model']).resolve() == Path(inherited['model']).resolve(), 'model/fixture identity')
    return fixtures['fixtures'], sources


def selected(fixtures):
    need(len(fixtures) == 8 and len({x['fixture_id'] for x in fixtures}) == 8, 'eight distinct original fixtures')
    for row in fixtures:
        need(row['normal_eos'] is True and type(row['max_output_tokens']) is int and row['max_output_tokens'] == 512
             and 886 <= len(row['prompt_ids']) <= 978
             and all(type(x) is int and 0 <= x < 248320 for x in row['prompt_ids'])
             and row['eos_token_ids'] and all(type(x) is int and 0 <= x < 248320 for x in row['eos_token_ids']), 'fixed D512 fixture')
    ordered = sorted(fixtures, key=lambda x:(len(x['prompt_ids']),x['fixture_id']))
    return [(f,s) for f in (ordered[0],ordered[-1]) for s in STARTS]


def expected_layout(kind, width=1):
    result = {}
    for i in range(64):
        fields = (('K',[1,4,1536,256],'BF16'),('V',[1,4,1536,256],'BF16')) if i % 4 == 3 else (
            ('conv',[1,3,10240],'BF16'),('S',[1,48,128,128],'F32'))
        result.update({f'root/state/{i}/{key}':(shape,dtype) for key,shape,dtype in fields})
    if kind == 'input': result.update({f'root/{key}':([1],'I32') for key in ('tokens','positions','lens')})
    elif kind == 'output': result = {'root/logits':([width,248320],'BF16'), **result}
    else: raise ValueError('fixed snapshot kind')
    return result


def metadata(kind, position, width=1):
    states = []
    for i in range(64):
        fields = ('K','V') if i % 4 == 3 else ('conv','S')
        states.append({key:{'array':f'root/state/{i}/{key}'} for key in fields})
    if kind == 'input': return dict(state=states, **{key:{'array':'root/'+key} for key in ('tokens','positions','lens')})
    return dict(logits={'array':'root/logits'},state=states,offsets=[position+width]*64)


def layout(index, kind, width=1):
    expected = expected_layout(kind, width)
    need(set(index['tensors']) == set(expected), 'all original tensor names required')
    for key,(shape,encoding) in expected.items():
        same(index['tensors'][key]['shape'], shape, 'original shape: '+key)
        need(index['tensors'][key]['encoding'] == encoding, 'original dtype: '+key)


def saved(b, child, desc, path, kind, position, width=1):
    need(desc['path'] == path, 'fixed raw artifact path')
    same(read((child/path).with_suffix('.json')), desc, 'raw descriptor sidecar')
    same(desc['metadata'], metadata(kind,position,width), 'raw metadata/offsets')
    index = b.snapshot_index(child, desc)  # Re-reads every original byte and finite bit pattern.
    layout(index,kind,width)
    return index


def teacher(seed, fixture, start):
    same([seed['fixture_id'],seed['start'],seed['prompt_ids']],
         [fixture['fixture_id'],start,fixture['prompt_ids']], 'fixed reference block identity')
    raw = seed['raw_ids']
    need(len(raw) == start+8 and all(type(t) is int and 0 <= t < 248320 for t in raw), 'complete actual teacher ID history')
    need(not set(raw[:-1]) & set(fixture['eos_token_ids']) and len(raw) < 512, 'normal EOS/cap before teacher block')
    same(seed['tokens'],raw[start-1:start+7], 'candidate input is reference causal slice')
    same(seed['position'],len(fixture['prompt_ids'])+start-1, 'committed anchor position')


def greedy(b, index, width):
    raw = b.tensor_bytes(index, 'root/logits')
    need(len(raw) == width*248320*2, 'all full-vocabulary logits')
    values = []
    for row in range(width):
        words = struct.unpack_from('<248320H', raw, row*248320*2)
        values.append(max(range(248320), key=lambda i:struct.unpack('<f',struct.pack('<I',words[i] << 16))[0]))
    return values  # Only a causal cross-check; equality is all original bytes.


def bookkeeping(rows, position, width):
    tape = [dict(shape=shape,dtype=DTYPES[dtype]) for shape,dtype in (
        ([width,10240],'BF16'),([width,48,128],'BF16'),([width,48,128],'BF16'),
        ([width,48,128],'BF16'),([width,48],'F32'),([width,48],'F32'))]
    same(rows, [dict(kind='AttnCache' if i%4 == 3 else 'GDNCache',offset=position+width,
        record=None if i%4 == 3 else True,tape_arrays=[] if i%4 == 3 else tape) for i in range(64)],
        'actual snapshot/record/offset bookkeeping')


def device_checks(row):
    tag = f"{row['fixture_id']}/s{row['start']}/T{row['width']}"
    inputs, outputs = expected_layout('input'), expected_layout('output',row['width'])
    staging = {name.replace('root/state/','root/'):(shape,dtype)
               for name,(shape,dtype) in inputs.items() if name.startswith('root/state/')}
    checks = []
    for label, arrays in (('same-seed',inputs),('staging-representation',staging),('full-output',outputs),
                          ('reference-seed-unchanged',inputs),('candidate-seed-unchanged',inputs)):
        checks.extend(dict(check=tag+'/'+label,name=name,shape=shape,dtype=DTYPES[dtype],
            reference_finite=True,candidate_finite=True,bit_equal=True) for name,(shape,dtype) in arrays.items())
    same(row['checks'], checks, 'all650 device comparison receipts; no substitution for raw equality')


class BitMismatch(ValueError):
    def __init__(self, point, contrast):
        first = next(row for row in contrast['tensors'] if not row['bit_equal'])
        name = first['name']; parts = name.split('/')
        self.details = dict(case=point,array=name,layer=int(parts[2]) if name.startswith('root/state/') else None,
            **first['first_difference'],differing_tensors=contrast['differing_tensors'],
            differing_words=contrast['differing_words'],differing_bytes=contrast['differing_bytes'])
        super().__init__('original-byte mismatch: '+point+'/'+name)


def compare_pair(b, left, right, point):
    comparison = b.compare_snapshots(left,right)
    if not comparison['bit_equal']: raise BitMismatch(point,comparison)
    return comparison


def case(b, child, row, seed):
    width,position = row['width'],seed['position']
    same([row['fixture_id'],row['start'],row['position'],row['tokens'],row['canonical']],
         [seed['fixture_id'],seed['start'],position,seed['tokens'][:width],seed['canonical']], 'case consumes exact canonical teacher block')
    tag = f"{seed['fixture_id']}/s{seed['start']}/T{width}"
    if not all(key in row for key in ('reference_output','candidate_output')):
        raise FileNotFoundError('both complete outputs not retained: '+tag)
    left = saved(b,child,row['reference_output'],'cases/'+tag+'/reference.safetensors','output',position,width)
    right = saved(b,child,row['candidate_output'],'cases/'+tag+'/candidate.safetensors','output',position,width)
    comparison = compare_pair(b,left,right,tag)
    a,c = greedy(b,left,width),greedy(b,right,width)
    same(a,row['reference_greedy_ids'],'reference argmax from full saved vocabulary')
    same(c,row['candidate_greedy_ids'],'candidate argmax from full saved vocabulary')
    same(a,seed['raw_ids'][seed['start']:seed['start']+width], 'separate original decode reproduces teacher continuation')
    bookkeeping(row['verify_bookkeeping'],position,width)
    device_checks(row)
    need(row['status'] == 'PASS', 'producer case did not complete all invariants')
    return dict(case=tag,tensor_pairs=comparison['tensor_count'],full_vocabulary_rows=width,
                bytes_per_arm=sum(x['bytes'] for x in left['tensors'].values()),bit_equal=True)


def events(child, complete):
    expected = [('memory','before-load')]
    cursor = 0
    for seed in complete['reference_blocks']:
        expected.extend(('reference-token',dict(fixture_id=seed['fixture_id'],start=seed['start'],decode_step=i,
            input_token=seed['raw_ids'][i-1],output_token=seed['raw_ids'][i])) for i in range(1,len(seed['raw_ids'])))
        for _ in WIDTHS:
            expected.append(('case',complete['cases'][cursor])); cursor += 1
    expected.append(('memory','after-release')); previous = -1
    with (child/'events.jsonl').open('rb') as stream:
        for kind, wanted in expected:
            line = stream.readline(); need(line.endswith(b'\n'), 'missing/partial ordered event')
            row = parse(line); elapsed = row.pop('elapsed_wall_s'); actual = row.pop('event')
            need(finite(elapsed) and previous <= elapsed <= complete['elapsed_wall_s'], 'ordered event time')
            previous = elapsed; need(actual == kind, 'actual event order')
            if kind == 'memory':
                need(set(row) == {'phase','active_bytes','peak_bytes','cache_bytes','not_performance'}
                     and row['phase'] == wanted and row['not_performance'] is True
                     and all(type(row[k]) is int and row[k] >= 0 for k in ('active_bytes','peak_bytes','cache_bytes')), 'memory event')
            else: same(row,wanted,'event/terminal case and teacher binding')
        need(not stream.read(1), 'unexpected event after release')


def audit(child, plan_sha256):
    child = Path(child).resolve()
    result = dict(schema='Exact-verifier-bridge-independent-v1',status='INCOMPLETE',forward_qualified=False,
        rollback_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,
        child_directory=str(child),plan_sha256=plan_sha256,cases=[],raw_files=0,
        scope='Only twelve fixed teacher-forced forwards: full logits and all128 final states, exact original bits.',
        limits=['Canonical provenance, separate teacher generation, independent device storage and seed-unchanged checks are source-bound device receipts; their full histories/after-seed payloads are not retained.',
                'Parameter payloads are not stored: only original-dtype before/after receipts are compared.',
                'Model manifest is bound; no model weight/runtime payload rehash or resource/lifecycle qualification here.',
                'No capture/features, intermediate accepted prefix, rollback, scheduler/ahead, HTTP or performance qualification.'])
    try:
        complete = read(child/'complete.json'); start_sha = sha(child/'complete.json')
        need(complete['schema'] == 'Exact-verifier-bridge-result-v1', 'actual bridge result schema')
        result['producer_status'] = complete['status']; result['producer_error'] = complete.get('error')
        need(complete['status'] in ('PASS','FAIL','INCOMPLETE'), 'known producer terminal status')
        if not complete.get('sources'):
            result['first_failure'] = 'producer stopped before source binding'; return result
        same(complete['contract'],expected_contract(),'fixed twelve-case contract and exclusions')
        need(all(complete[k] is False for k in ('rollback_qualified','full_model_qualified','http_qualified','performance_qualified')), 'excluded claims stay false')
        fixtures,sources = identities(child,complete,plan_sha256)
        result['source_pins'] = {**sources,str(RAW):RAW_SHA}
        b = raw_reader(); points = selected(fixtures); blocks = complete['reference_blocks']; rows = complete['cases']
        need(len(blocks) <= 4 and len(rows) <= 12, 'no extra fixed cases or anchors')
        seen = set(); cursor = 0
        for i,seed in enumerate(blocks):
            fixture,start = points[i]; teacher(seed,fixture,start)
            path = f"inputs/{fixture['fixture_id']}-s{start}/canonical.safetensors"
            canonical = saved(b,child,seed['canonical'],path,'input',seed['position']); seen.add(path)
            for name,value in (('tokens',seed['tokens'][0]),('positions',seed['position']),('lens',seed['position']+1)):
                same(struct.unpack('<i',b.tensor_bytes(canonical,'root/'+name))[0],value,'canonical I32 '+name)
            for width in WIDTHS:
                if cursor == len(rows): break
                row = rows[cursor]; same([row['fixture_id'],row['start'],row['width']],
                    [fixture['fixture_id'],start,width], 'fixed case order, no skipped width')
                result['cases'].append(case(b,child,row,seed)); cursor += 1
                seen.update(row[k]['path'] for k in ('reference_output','candidate_output'))
        result['raw_files'] = len(seen)
        if complete['status'] != 'PASS' or len(blocks) != 4 or cursor != 12:
            result.update(status='FAIL' if complete['status'] == 'FAIL' else 'INCOMPLETE',
                first_failure='producer did not complete all twelve cases and postchecks')
            return result
        need(complete['forward_qualified'] is True and complete['source_model_pins_rechecked'] is True
             and complete['device_cleanup_completed'] is True and not complete.get('error')
             and not complete.get('first_device_error') and complete['device_run'] is True, 'complete producer gates')
        same(complete['device'],dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3'),'actual device/runtime receipt')
        need(finite(complete['elapsed_wall_s']), 'terminal elapsed time')
        parameters = []
        for name in ('parameters-before','parameters-after'):
            desc = complete[name]; same(desc,dict(path=name+'.json',sha256=sha(child/(name+'.json')),
                scope='all model parameter original bytes; producer receipt'),'parameter receipt binding')
            values = read(child/desc['path']); b.parameter_receipt(values); parameters.append(b.parameter_values(values))
        same(parameters[0],parameters[1],'unchanged1349 parameter value receipts')
        events(child,complete)
        need({str(p.relative_to(child)) for p in child.rglob('*.safetensors')} == seen and len(seen) == 28,
             'all28 successful raw files, no extra failure artifacts')
        need(not (child/'failure.json').exists(), 'producer failure artifact remains')
        for path in seen:
            need(sha(child/path) == read((child/path).with_suffix('.json'))['sha256'], 'raw changed during audit')
        need(sha(child/'complete.json') == start_sha, 'terminal changed during audit')
        result.update(status='PASS',forward_qualified=True,complete_sha256=start_sha,
            ledger_sha256=sha(child/'events.jsonl'),audit_source=record(__file__),
            output_tensor_pairs=sum(x['tensor_pairs'] for x in result['cases']),
            full_vocabulary_rows_per_arm=sum(x['full_vocabulary_rows'] for x in result['cases']),
            bytes_compared_per_arm=sum(x['bytes_per_arm'] for x in result['cases']),parameter_value_receipts=1349)
    except BitMismatch as exc: result.update(status='FAIL',first_failure=str(exc),first_difference=exc.details)
    except FileNotFoundError as exc: result['first_failure']='missing required evidence: '+str(exc)
    except Exception as exc: result.update(status='FAIL',first_failure=type(exc).__name__+': '+str(exc))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child',type=Path,required=True); p.add_argument('--plan-sha256',required=True)
    p.add_argument('--out',type=Path,required=True); args = p.parse_args(argv)
    need(not args.out.exists(), 'fresh independent output required')
    result = audit(args.child,args.plan_sha256)
    with args.out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__ == '__main__': raise SystemExit(main())
