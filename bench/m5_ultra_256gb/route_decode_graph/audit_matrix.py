"""Offline, stdlib-only audit of the fixed E024 route matrix.

Only ten prescribed snapshots contain full raw bytes. Other point results are
validated source-pinned device comparisons and full-byte hash receipts, not an
independent recomputation. No owner/resource, full-model, HTTP or speed claim.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
QUALIFIER_SHA = '6130d948450c825302d9857450204f8503c0dab449050a1bb59eaa51b77920e3'
RAW_READER = HERE.parent/'long_decode_isolation/audit_raw.py'
RAW_READER_SHA = 'b3bffc7bd261b9b347b5d299d965273ff2406a92c27576747867aa515be62261'
E024_SHA = '5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
E023_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
CANDIDATE_SHA = '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3'
PINS_SHA = '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5'
USED = (257, 1023, 1024, 1280, 1536)
MIXED = (257, 511, 767, 900, 978, 1000, 1022, 1023)
RAW_POINTS = {'matrix-B1-used257': 'canonical', 'matrix-B1-used1023': 'both',
              'matrix-B1-used1024': 'both', 'mixed-used1024': 'both'}
DTYPES = {'BF16': 'mlx.core.bfloat16', 'F32': 'mlx.core.float32', 'I32': 'mlx.core.int32'}
WIDTH = {'BF16': 2, 'F32': 4, 'I32': 4}


def need(ok, message):
    if not ok: raise ValueError(message)
def same(a, b): return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
def read(path): return json.loads(Path(path).read_bytes())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()
def digest(value):
    need(isinstance(value, str) and len(value) == 64 and len(bytes.fromhex(value)) == 32, 'SHA256 encoding')
def record(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def raw_reader():
    need(sha(RAW_READER) == RAW_READER_SHA, 'fixed raw reader source')
    spec = importlib.util.spec_from_file_location('_matrix_raw_reader', RAW_READER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def points():
    result = []
    for used in USED:
        result.extend(dict(name=f'matrix-B{b}-used{used}', kind='matrix', B=b, lens=[used]*b) for b in range(1, 9))
        if used == 1023:
            result.extend(dict(name=f'uniform-used{u}', kind='uniform-chain', B=1, lens=[u]) for u in (1023, 1024, 1025))
        if used in (1023, 1024):
            result.extend(dict(name=f'reuse-used{used}-{i}-B{b}', kind='batch-reuse', B=b, lens=[used]*b) for i, b in enumerate((1, 8, 1)))
    result.extend(dict(name=f'mixed-used{1023+i}', kind='mixed-chain', B=8, lens=[n+i for n in MIXED]) for i in range(3))
    return result


def layout(batch, kind):
    result = {}
    for layer in range(64):
        entries = (('K', [batch, 4, 1536, 256], 'BF16'), ('V', [batch, 4, 1536, 256], 'BF16')) if layer % 4 == 3 else (
            ('conv', [batch, 3, 10240], 'BF16'), ('S', [batch, 48, 128, 128], 'F32'))
        result.update({f'root/state/{layer}/{name}': (shape, dtype) for name, shape, dtype in entries})
    if kind == 'input': result.update({f'root/{name}': ([batch], 'I32') for name in ('tokens', 'positions', 'lens')})
    else:
        need(kind == 'output', 'input/output layout'); result['root/logits'] = ([batch, 248320], 'BF16')
    return result


def metadata(kind):
    state = []
    for layer in range(64):
        names = ('K', 'V') if layer % 4 == 3 else ('conv', 'S')
        state.append({'class': 'BatchKV' if layer % 4 == 3 else 'BatchState',
                      'fields': {name: {'array': f'root/state/{layer}/{name}'} for name in names}})
    result = {'state': state}
    result.update({name: {'array': 'root/'+name} for name in (('tokens', 'positions', 'lens') if kind == 'input' else ('logits',))})
    return result


def hashes(row, batch, kind):
    expected = layout(batch, kind)
    need(set(row) == {'metadata', 'arrays'} and same(row['metadata'], metadata(kind)) and set(row['arrays']) == set(expected), 'complete original state metadata/array keys')
    for name, (shape, dtype) in expected.items():
        item = row['arrays'][name]
        need(set(item) == {'shape', 'dtype', 'bytes', 'sha256', 'finite'} and same(item['shape'], shape)
             and item['dtype'] == DTYPES[dtype] and type(item['bytes']) is int
             and item['bytes'] == math.prod(shape)*WIDTH[dtype] and item['finite'] is True, 'original shape/dtype/bytes/finite: '+name)
        digest(item['sha256'])


def checks(rows, expected):
    need(isinstance(rows, list) and len(rows) == len(expected), 'complete device comparison count')
    wanted = {name: (label, shape, dtype) for name, label, shape, dtype in expected}
    need(len(wanted) == len(expected) and {x['name'] for x in rows} == set(wanted), 'complete unique device comparison names')
    for row in rows:
        label, shape, dtype = wanted[row['name']]
        need(same(row, dict(check=label, name=row['name'], shape=shape, dtype=dtype,
             reference_finite=True, candidate_finite=True, bit_equal=True)), 'original device comparison flag/shape: '+row['name'])


def int_hash(values): return hashlib.sha256(struct.pack('<'+'i'*len(values), *values)).hexdigest()


def verify_point(row, spec, previous):
    need(all(same(row[k], v) for k, v in spec.items()) and row['status'] == 'PASS', 'fixed point order/spec/status')
    b, name = spec['B'], spec['name']
    continuation = previous is not None and spec['kind'] in ('uniform-chain', 'mixed-chain') and previous['kind'] == spec['kind']
    need(row['canonical_provenance'] == ('reference-input-view-after-prior-exact-step' if continuation else 'independent-prefill-rebuild-storage'), 'canonical storage provenance')
    need(row['candidate_bound'] == (1023 if max(spec['lens']) < 1024 else 1536), 'bound route')
    need(len(row['input_buffer_bytes']) == 2 and all(type(v) is int and v > 0 for v in row['input_buffer_bytes']), 'disjoint-input device receipt')
    for section in ('input_before', 'input_after', 'outputs'): need(set(row[section]) == {'A', 'C'}, 'two complete arms')
    for arm in ('A', 'C'):
        for section in ('input_before', 'input_after'): hashes(row[section][arm], b, 'input')
        hashes(row['outputs'][arm], b, 'output')
        need(same(row['input_before'][arm], row['input_after'][arm]), 'input mutated')
        for field, values in (('positions', [n-1 for n in spec['lens']]), ('lens', spec['lens'])):
            need(row['input_before'][arm]['arrays']['root/'+field]['sha256'] == int_hash(values), 'actual context scalar bytes')
    need(same(row['input_before']['A'], row['input_before']['C']) and same(row['outputs']['A'], row['outputs']['C']), 'two-arm full-byte hash/metadata equality')
    checks(row['checks'], [(key, name+'/outputs', shape, DTYPES[dtype]) for key, (shape, dtype) in layout(b, 'output').items()])
    greedy = row['greedy_ids']
    need(isinstance(greedy, list) and len(greedy) == 2 and same(greedy[0], greedy[1]) and len(greedy[0]) == b
         and all(type(n) is int and 0 <= n < 248320 for n in greedy[0]), 'own greedy ID receipt')
    if continuation:
        need(spec['lens'] == [n+1 for n in previous['lens']], 'chain positions advance once')
        for arm, ids in zip(('A', 'C'), previous['greedy_ids']):
            current = row['input_before'][arm]['arrays']; prior = previous['outputs'][arm]['arrays']
            need(current['root/tokens']['sha256'] == int_hash(ids), 'chain must use own prior greedy tokens')
            need(all(same(value, current[key]) for key, value in prior.items() if key.startswith('root/state/')), 'chain must retain own complete prior state')
    wanted = set() if name not in RAW_POINTS else ({'canonical'} if RAW_POINTS[name] == 'canonical' else {'canonical', 'A-output', 'C-output'})
    need(set(row['raw']) == wanted, 'fixed sparse raw coverage')


def graph_call(row, spec, invocation, objects):
    expected = dict(event='graph-call', phase=spec['name'], batch=spec['B'], used=max(spec['lens']),
        positions=[n-1 for n in spec['lens']], lens=spec['lens'], KV_backing_lengths=[[1536, 1536]]*16,
        status='SUBMITTED', invocation=invocation)
    need(all(same(row[k], v) for k, v in expected.items()), 'actual graph invocation/context')
    tokens = row['tokens']; need(len(tokens) == spec['B'] and all(type(n) is int and 0 <= n < 248320 for n in tokens), 'graph token IDs')
    bound = 1023 if max(spec['lens']) < 1024 else 1536
    new = row['actual_compile_objects']
    need(type(row['compiled_function_count']) is int and row['compiled_function_count'] == len(new), 'actual callable family count')
    need(new[:len(objects)] == objects and len(new) == len(objects)+(0 if any(ast.literal_eval(x['key'])[0] == bound for x in objects) else 1), 'compiled callable identity/reuse')
    for item in new:
        need(set(item) == {'key', 'type', 'id'} and item['type'] == 'gc_func' and type(item['id']) is int and item['id'] > 0, 'actual compiled callable receipt')
        key = ast.literal_eval(item['key'])
        need(item['key'] in (repr((1023, ((False, False),)*48)), repr((1536, ((False, False),)*48))), 'exact gates-OFF/norm-OFF callable key')
    need(len({x['key'] for x in new}) == len(new) and len({x['id'] for x in new}) == len(new)
         and any(ast.literal_eval(x['key'])[0] == bound for x in new), 'distinct compiled keys and identities')
    return new


def prefill_checks(row, stage, number, begin, end):
    need(all(same(row[k], v) for k, v in dict(stage=stage, row=number, begin=begin, end=end).items()), 'fixed prefill checkpoint order')
    label = f'prefill-{end}-{number}' if stage == 'matrix' else f'mixed-prefill-{number}'
    expected = [('root', label+'/logits', [1, 248320], DTYPES['BF16'])]
    capacity = ((end+255)//256)*256
    for layer in range(64):
        entries = (('keys', [1, 4, capacity, 256], 'BF16'), ('values', [1, 4, capacity, 256], 'BF16')) if layer % 4 == 3 else (
            ('conv_state', [3, 10240], 'BF16'), ('S', [48, 128, 128], 'F32'))
        expected.extend((f'root/{layer}/{name}', label+'/state', shape, DTYPES[dtype]) for name, shape, dtype in entries)
    checks(row['checks'], expected)


def event_plan():
    result = [('memory', 'before-load-reference'), ('memory', 'before-load-candidate'), ('loaded-weights', None)]
    previous = 0; fixed = points()
    for used in USED:
        result.extend(('prefill', ('matrix', i, previous, used-1)) for i in range(8))
        chosen = [p for p in fixed if (p['kind'] == 'matrix' and p['lens'][0] == used)
                  or (p['kind'] == 'uniform-chain' and used == 1023)
                  or (p['kind'] == 'batch-reuse' and p['lens'][0] == used)]
        for point in chosen:
            result.extend([('memory', 'before-'+point['name']), ('graph-call', point), ('memory', 'after-'+point['name']), ('point', point)])
        previous = used-1
    result.append(('memory', 'before-mixed-prefill'))
    result.extend(('prefill', ('mixed', i, 0, n-1)) for i, n in enumerate(MIXED))
    for point in fixed[-3:]:
        result.extend([('memory', 'before-'+point['name']), ('graph-call', point), ('memory', 'after-'+point['name']), ('point', point)])
    return result


def saved_snapshot(base, child, receipt, row, kind, label):
    need(receipt['path'] == f"raw/{row['name']}/{label}.safetensors", 'fixed raw artifact path')
    need(same(read((child/receipt['path']).with_suffix('.json')), receipt), 'raw descriptor sidecar')
    index = base.snapshot_index(child, receipt)
    source = row['input_before']['A'] if kind == 'input' else row['outputs'][label[0]]
    need(same(receipt['metadata'], source['metadata']), 'raw metadata/ledger binding')
    actual = {name: {k: v for k, v in data.items() if k not in ('offset', 'encoding')} for name, data in index['tensors'].items()}
    need(same(actual, source['arrays']), 'every raw byte hash binds complete ledger')
    return index


def raw_point(base, child, row):
    indices = {label: saved_snapshot(base, child, receipt, row, 'input' if label == 'canonical' else 'output', label)
               for label, receipt in row['raw'].items()}
    if not indices: return dict(snapshots=0, bytes=0, paired_output_tensors=0)
    canonical = indices['canonical']
    for field, values in (('positions', [n-1 for n in row['lens']]), ('lens', row['lens'])):
        need(base.tensor_bytes(canonical, 'root/'+field) == struct.pack('<'+'i'*len(values), *values), 'raw actual context values')
    result = dict(snapshots=len(indices), bytes=sum(sum(t['bytes'] for t in x['tensors'].values()) for x in indices.values()), paired_output_tensors=0)
    if 'A-output' in indices:
        comparison = base.compare_snapshots(indices['A-output'], indices['C-output'])
        result['comparison'] = comparison
        need(comparison['bit_equal'], 'raw original-bit output difference')
        result['paired_output_tensors'] = comparison['tensor_count']
        for arm, ids in zip(('A', 'C'), row['greedy_ids']):
            raw = base.tensor_bytes(indices[arm+'-output'], 'root/logits')
            # BF16 words are converted only for argmax; equality uses original bits.
            greedy = []
            for b in range(row['B']):
                words = struct.iter_unpack('<H', raw[b*248320*2:(b+1)*248320*2])
                values = (struct.unpack('<f', struct.pack('<I', w[0] << 16))[0] for w in words)
                greedy.append(max(enumerate(values), key=lambda pair: pair[1])[0])
            need(same(greedy, ids), 'independent raw-logits greedy IDs')
    return result


def parameters(base, child, complete):
    refs = complete['parameter_receipts']
    need([(r['model'], r['phase']) for r in refs] == [(0, 'before'), (1, 'before'), (0, 'after'), (1, 'after')], 'both independent models parameter boundaries')
    values = []
    for ref in refs:
        need(ref['path'] == f"parameters-{ref['model']}-{ref['phase']}.json", 'parameter receipt path')
        data = base.pinned_json(child, {k: ref[k] for k in ('path', 'sha256')}); base.parameter_receipt(data); values.append(data)
    need(all(same(base.parameter_values(values[0]), base.parameter_values(v)) for v in values[1:]), 'both model parameters exactly unchanged')
    need(all(values[0][k]['handle_id'] != values[1][k]['handle_id'] for k in values[0]), 'initial independent model handle receipt')
    return values[0]


def identities(child, complete, plan_sha256):
    plan = read(child/'plan.json')
    need(sha(child/'plan.json') == plan_sha256 == complete['plan_sha256'], 'bound matrix plan SHA')
    need(set(plan) == {'schema', 'qualifier_sha256', 'candidate', 'e024_plan', 'deadline_s'}
         and plan['schema'] == 'E024-route-matrix-plan-v1' and plan['qualifier_sha256'] == complete['qualifier_sha256'] == QUALIFIER_SHA,
         'exact matrix source/plan schema')
    need(type(plan['deadline_s']) in (int, float) and math.isfinite(plan['deadline_s']) and 0 < plan['deadline_s'] <= 1800, 'fixed maximum deadline')
    need(plan['candidate'] == dict(path=str(HERE/'candidate.py'), sha256=CANDIDATE_SHA), 'fixed route candidate')
    sources = {str(HERE/'qualifier_matrix.py'): QUALIFIER_SHA, str(HERE/'candidate.py'): CANDIDATE_SHA,
        str(HERE/'qualify.py'): 'fe8e1c044e97885ec7c97fddce04a6e34f5b9067be4d982c5f6ad1071c59c801',
        str(HERE.parent/'long_decode_isolation/qualify.py'): '9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e',
        str(HERE.parent/'long_decode_graph/qualify.py'): 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
        str(HERE.parent/'long_decode_graph/candidate.py'): 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
        str(HERE.parent/'pareto_realistic/qualify_native.py'): '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
        str(HERE.parent/'long_decode_graph/model-source-pins.json'): PINS_SHA}
    need(complete['sources'] == sources and all(sha(p) == value for p, value in sources.items()), 'all eight executable/source pins')
    old = plan['e024_plan']; need(set(old) == {'path', 'sha256'} and Path(old['path']).is_absolute() and old['sha256'] == E024_SHA == sha(old['path']), 'original E024 plan')
    oldplan = read(old['path']); need(same(read(child/'inherited-e024-plan.json'), oldplan), 'saved E024 plan')
    inherited = oldplan['e023_plan']; need(inherited['sha256'] == E023_SHA == sha(inherited['path']), 'inherited E023 plan')
    inherited = read(inherited['path']); need(same(read(child/'inherited-e023-plan.json'), inherited), 'saved E023 plan')
    pinned = {}
    for name in ('source_manifest', 'model_manifest', 'fixtures'):
        item = inherited[name]; need(sha(item['path']) == item['sha256'], 'inherited manifest SHA'); pinned[name] = read(item['path'])
    root = Path(inherited['package_root'])/'tutorial_code/miniengine'
    files = pinned['source_manifest']['files']; source_receipt = dict(root=str(root.resolve()), manifest_sha256=PINS_SHA, files=files)
    need(same(complete['model_source_receipt'], source_receipt) and files == read(HERE.parent/'long_decode_graph/model-source-pins.json')['files']
         and set(files) == {str(p.relative_to(root)) for p in root.rglob('*.py')}
         and all(sha(root/p) == value for p, value in files.items()), 'actual core source receipt/tree')
    model = pinned['model_manifest']; need(model['schema'] == 'E023-model-v1' and model['model'] == inherited['model']
         and model['revision'] == Path(inherited['model']).name and pinned['fixtures']['model'] == inherited['model'], 'model identity manifest')
    need(complete['mlx_environment'] == inherited['execution']['environment'], 'runtime environment receipt')
    diagnostic = []
    for f in pinned['fixtures']['fixtures']:
        ids = f['prompt_ids']; repeated = (ids*((1535+len(ids)-1)//len(ids)))[:1535]
        diagnostic.append({**f, 'fixture_id': 'diagnostic-repeat-'+f['fixture_id'], 'prompt_ids': repeated,
                           'source_fixture_id': f['fixture_id'], 'diagnostic_only': True, 'normal_eos': True, 'max_output_tokens': 4})
    need(len(diagnostic) == 8 and same(read(child/'diagnostic-inputs.json'), diagnostic), 'exact repeated diagnostic inputs')
    return plan, inherited


def contract(complete):
    c = complete['contract']
    expected = dict(schema='E024-route-matrix-contract-v1', points=points(), expected_points=52, matrix_points=40,
        uniform_crossing_points=3, mixed_crossing_points=3, batch_reuse_points=6, layers=64, state_arrays=128,
        output_arrays=129, input_arrays=131, KV_backing=1536, prefill_checkpoints=[u-1 for u in USED],
        max_prefill_tokens_per_model=18730, max_decode_output_tokens_per_model=227, raw_points=RAW_POINTS,
        expected_pass_full_snapshots=10, two_callable_families_not_two_machine_graphs=True,
        configuration=dict(max_len=2048, max_seqs=8, max_batched_tokens=16384, cache_limit_gib=2,
            compile_decode=False, fuse_decode=False, gates_compile=False, norm_reuse=False))
    expected.update({k: False for k in ('parameter_replacement_qualified', 'ahead_qualified', 'scheduler_lifecycle_qualified', 'full_model_qualified', 'performance_qualified')})
    need(all(same(c[k], v) for k, v in expected.items()), 'fixed bounded diagnostic contract')


def scan(child, complete, weights, base, summary):
    path = child/'events.jsonl'; before = path.stat(); h = hashlib.sha256(); expected = event_plan()
    index = 0; previous = None; graph = None; objects = []; elapsed = -1.; recorded_points = []; reuse_inputs = {}
    summary.update(point_count=0, device_output_checks=0, prefill_checks=0, actual_graph_calls=0, raw_snapshots=0,
                   raw_tensor_bytes=0, raw_paired_output_tensors=0, raw_points={})
    with path.open('rb') as stream:
        for lineno, line in enumerate(stream, 1):
            need(line.endswith(b'\n') and len(line) < 4 << 20, 'closed bounded ledger row'); h.update(line); row = json.loads(line)
            when = row['elapsed_wall_s']; need(type(when) in (float, int) and math.isfinite(when) and when >= elapsed, 'ledger monotonic elapsed time'); elapsed = when
            if row.get('event') == 'point' and row.get('status') == 'FAIL':
                need(summary['point_count'] < 52 and all(same(row[k], v) for k, v in points()[summary['point_count']].items()), 'failed point must be next fixed point')
                summary['first_failed_point'] = row; summary['ledger_line'] = lineno
                need(complete['status'] != 'PASS', 'failed point cannot have PASS completion')
                # Failed points may end before after-memory; no following event is legal.
                remaining = stream.read(1); need(not remaining, 'events after first failure')
                summary['failed_raw'] = {}
                for label, receipt in row.get('failure_raw', {}).items():
                    need(receipt['path'] == f"failure/{row['name']}/{label}.safetensors", 'failure raw path')
                    need(same(read((child/receipt['path']).with_suffix('.json')), receipt), 'failure raw sidecar')
                    saved = base.snapshot_index(child, receipt)
                    summary['failed_raw'][label] = dict(sha256=saved['sha256'], tensors=len(saved['tensors']))
                if 'A-output' in row.get('failure_raw', {}) and 'C-output' in row['failure_raw']:
                    a = base.snapshot_index(child, row['failure_raw']['A-output'])
                    c = base.snapshot_index(child, row['failure_raw']['C-output'])
                    summary['failed_output_contrast'] = base.compare_snapshots(a, c)
                summary['failed_payload_scope'] = 'Input handles were saved at failure, not historical before bytes; only input_before hash receipts predate execution.'
                need((path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino) == (before.st_size, before.st_mtime_ns, before.st_ino), 'failed ledger changed during audit')
                summary['ledger_sha256'] = h.hexdigest(); return False
            need(index < len(expected), 'extra ledger event'); event, value = expected[index]
            need(row['event'] == event, 'exact device event order')
            if event == 'memory':
                need(row['phase'] == value and row['observation_not_perf'] is True
                     and all(type(row[k]) is int and row[k] >= 0 for k in ('active_bytes', 'cache_bytes', 'peak_bytes')), 'memory phase receipt')
            elif event == 'loaded-weights':
                if weights is None:
                    need(len(row['checks']) == 1349 and len({x['name'] for x in row['checks']}) == 1349, 'partial loaded weight count')
                    need(all(x['check'] == 'independent-loaded-weights' and all(x[k] is True for k in ('reference_finite', 'candidate_finite', 'bit_equal')) for x in row['checks']), 'partial loaded weight flags')
                else:
                    checks(row['checks'], [('root/'+name, 'independent-loaded-weights', p['shape'], p['dtype']) for name, p in weights.items()])
            elif event == 'prefill': prefill_checks(row, *value); summary['prefill_checks'] += 129
            elif event == 'graph-call':
                if complete['status'] != 'PASS' and row['status'] != 'SUBMITTED':
                    need(row['phase'] == value['name'], 'failed graph phase'); summary['failed_graph_call'] = row
                else:
                    summary['actual_graph_calls'] += 1; objects = graph_call(row, value, summary['actual_graph_calls'], objects)
                graph = row
            else:
                verify_point(row, value, previous)
                reuse_key = (row['B'], row['lens'][0])
                if row['kind'] == 'matrix' and reuse_key in ((1, 1023), (8, 1023), (1, 1024), (8, 1024)):
                    reuse_inputs[reuse_key] = row['input_before']
                elif row['kind'] == 'batch-reuse':
                    need(same(row['input_before'], reuse_inputs.get(reuse_key)), 'B1/B8/B1 reuse must use original same-point inputs')
                need(graph is not None and graph['phase'] == row['name'] and row['input_before']['C']['arrays']['root/tokens']['sha256'] == int_hash(graph['tokens']), 'graph invoked on recorded original inputs')
                raw = raw_point(base, child, row)
                if raw['snapshots']: summary['raw_points'][row['name']] = raw
                summary['raw_snapshots'] += raw['snapshots']; summary['raw_tensor_bytes'] += raw['bytes']; summary['raw_paired_output_tensors'] += raw['paired_output_tensors']
                summary['point_count'] += 1; summary['device_output_checks'] += 129; previous = row; graph = None
                recorded_points.append({k: row[k] for k in ('name', 'status', 'raw')})
            index += 1
    need((path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino) == (before.st_size, before.st_mtime_ns, before.st_ino), 'ledger changed during audit'); summary['ledger_sha256'] = h.hexdigest(); summary['last_event_elapsed_s'] = elapsed
    if complete['status'] != 'PASS': return False
    need(index == len(expected) and same(recorded_points, complete['points']), 'complete exact point/event sequence')
    need(complete['actual_graph_calls'] == 52 and same(complete['compiled_objects'], objects) and len(objects) == 2, 'terminal actual compiled calls/families')
    need(summary['raw_snapshots'] == 10 and summary['raw_paired_output_tensors'] == 387, 'full sparse raw evidence completeness')
    return True


def audit(child, plan_sha256):
    child = Path(child).resolve()
    result = dict(schema='E024-route-matrix-audit-v1', status='INCOMPLETE', matrix_qualified=False,
        full_model_qualified=False, performance_qualified=False, owner_resources_audited=False,
        audit_source=record(__file__), raw_reader_source=dict(path=str(RAW_READER), sha256=RAW_READER_SHA),
        scope='52 fixed diagnostic steps only. Ten full raw snapshots and three complete output contrasts are independently byte-read; other states, prefills and parameters retain source-pinned device comparison/full-byte-hash evidence. No parameter replacement, ahead, scheduler, HTTP or performance qualification.')
    try:
        complete_path = child/'complete.json'; complete_sha = sha(complete_path); complete = read(complete_path)
        need(complete['schema'] == 'E024-route-matrix-result-v1', 'matrix result schema')
        result['producer_status'] = complete['status']; result['complete_sha256'] = complete_sha
        if complete['status'] == 'INCOMPLETE': return result
        need(complete['status'] in ('PASS', 'FAIL'), 'known producer status')
        # A rejected or interrupted run remains rejected, even if every saved bit so far agrees.
        if complete['status'] != 'PASS':
            result['status'] = 'FAIL'; result['producer_error'] = complete.get('error'); result['points_saved'] = len(complete.get('points', []))
            need(sha(child/'plan.json') == plan_sha256 == complete['plan_sha256'], 'failed run plan binding')
            base = raw_reader(); scan(child, complete, None, base, result)
            if (child/'failure.json').exists():
                result['producer_first_failure'] = read(child/'failure.json'); result['failure_json_sha256'] = sha(child/'failure.json')
            result['reason'] = 'Producer rejected or stopped before a complete qualified matrix; partial evidence never grants qualification.'
            return result
        need(complete['device_run'] is True and same(complete['device'], dict(name='Apple M5 Ultra', architecture='applegpu_g17d', mlx_version='0.32.3')), 'actual device/runtime')
        need(all(complete[k] is True for k in ('matrix_complete', 'matrix_qualified', 'source_model_pins_rechecked'))
             and complete['full_model_qualified'] is False and complete['performance_qualified'] is False
             and not complete.get('error') and not complete.get('cleanup_error'), 'strict diagnostic completion')
        plan, inherited = identities(child, complete, plan_sha256); contract(complete)
        need(type(complete['elapsed_wall_s']) in (float, int) and 0 < complete['elapsed_wall_s'] <= plan['deadline_s'], 'completed within child deadline')
        base = raw_reader(); weights = parameters(base, child, complete)
        need(scan(child, complete, weights, base, result), 'complete matrix ledger')
        need(result['last_event_elapsed_s'] <= complete['elapsed_wall_s'], 'all events precede terminal completion')
        expected_raw = {f"raw/{p['name']}/{label}.safetensors" for p in complete['points'] for label in p['raw']}
        need({str(p.relative_to(child)) for p in child.rglob('*.safetensors')} == expected_raw, 'exact ten full raw files, no extra/failure payload on PASS')
        need(sha(complete_path) == complete_sha and sha(child/'plan.json') == plan_sha256, 'terminal evidence changed')
        result.update(status='PASS', matrix_qualified=True, parameter_count=len(weights), plan_sha256=plan_sha256,
            model_identity=inherited['model'], model_payload_rehashed=False,
            numerical_evidence='Device bit flags + full original-byte hashes for all 52 points; independent raw byte comparisons at only the three predeclared output points.')
    except FileNotFoundError as error: result.update(status='INCOMPLETE', reason=str(error), matrix_qualified=False)
    except Exception as error: result.update(status='FAIL', error=dict(type=type(error).__name__, message=str(error)), matrix_qualified=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    result = audit(args.child, args.plan_sha256)
    with args.out.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    return {'PASS': 0, 'FAIL': 1, 'INCOMPLETE': 2}[result['status']]


if __name__ == '__main__': raise SystemExit(main())
