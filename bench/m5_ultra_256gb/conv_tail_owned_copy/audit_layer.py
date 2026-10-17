"""Independent E026 real-weight single-layer audit: stdlib only, no execution.

Reads frozen evidence and original-dtype raw bytes; never imports a screen,
owner, MLX, NumPy or model. Qualification is limited to synthetic finite inputs
of one real GDN layer. It is neither full-model nor performance qualification.
"""
import argparse
from array import array
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import tempfile
import traceback


ARMS = ('off', 'on-lazy', 'on-materialize')
LENGTHS = (1, 19, 935, 978)
MODES = ('fresh', 'continuing')
MODEL_REVISION = '5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6'
CONFIG_SHA = '7380cfc6eb77891381ee24e944e6bd0caed7a957707f2512ddd6effc82c7e706'
INDEX_SHA = '8a3f70b51e57843efcb417829dc8be081e414517995234e71fc7cf508ae01f90'
SHARD_SHA = 'a9de796bfb40fbe757ff3e567d83d5d100906d6dd231c8b11c431cd97d3dc288'
HEADER_SHA = '5ee4fe36fedff37be5707fae8d414da5f4fbf11f23c74dd0c3a97d7384f497e7'
SHARD = 'model-00001-of-00003.safetensors'
SHARD_BYTES, HEADER_BYTES, BUNDLE_BYTES = 5343250458, 86450, 65239488
PREFIX = 'language_model.model.layers.0.linear_attn.'
PARAMETERS = {'A_log', 'dt_bias', 'conv1d_weight', 'norm.weight'} | {
    op + '.' + field for op in ('in_proj_a', 'in_proj_b', 'in_proj_qkv', 'in_proj_z', 'out_proj')
    for field in ('weight', 'scales')}
DTYPES = {'BF16': (2, 'H', 7), 'F32': (4, 'I', 23), 'U32': (4, 'I', None), 'U8': (1, 'B', None)}
CANONICAL = {'bfloat16': 'BF16', 'float32': 'F32', 'uint32': 'U32', 'uint8': 'U8'}
ENCODINGS = {'bfloat16': 'uint16-le', 'float32': 'uint32-le', 'uint32': 'uint32-le', 'uint8': 'uint8'}
CHUNK = 1 << 20
CASE_IDS = tuple(f'T{t}-{kind}' for t in LENGTHS for kind in MODES)
CASE_PHASES = ('before_prefill', 'after_hook', 'after_prefill_build', 'after_normal_product_eval',
    'after_common_validation_eval', 'after_raw_artifacts', 'after_drop_observer_refs', 'after_case_release')
SUPPORT_PINS = {
    'layer_weights.py': '8ee0ebd9df9385ec35baf6d0a0ae80bf780cf80a5e255b34d6fe2fed560df77a',
    'integration.py': '906dcfece0ce249d3fc06e968ec552d0ae20b665086e5812c54453fa47780ad1',
    'candidate.py': '4d9658d34734e3c252375c4b76fcb2eccf83555121d7eb51d7f3e81eb2c48ccc'}
CORE_MANIFEST_SHA = '93bf4b26318a1a7820c00eecec5b134f93f1f516435e4b9709e91e8b186a4050'
SCREEN_SHA = '5629475dc5c167af9edb0fa131fe922280d4ccdf1ce16d2a558b057cacbc1549'
OWNER_SHA = 'd7a31ca2f326d036659ea5fdd33a4ba81525b212901c45ad87621f89b2fdb6c9'
ENV = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
       'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}
POLICY = {'real_free_min_bytes': 50 << 30, 'allowed_initial_pressure': [1, 2],
    'pressure_must_never_increase': True, 'swapouts_delta_max': 0, 'swap_used_growth_max_mib': 16,
    'child_physical_max_bytes': 2 << 30, 'mlx_boundary_max_bytes': 1 << 30, 'mlx_cache_limit_bytes': 0,
    'power': 'AC Power', 'thermal_codes': [0, 1],
    'terminal_footprint_policy': 'raw-error+exit0+strict-reap+child-self-peak-proof'}


def same(a, b):
    """Typed JSON equality: bool must not impersonate an integer counter."""
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def nofile(row):
    return {k: v for k, v in row.items() if k != 'file'}


def jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            need(line.endswith('\n') and line.strip(), f'truncated/empty JSONL line {number}: {path}')
            rows.append(json.loads(line))
    return rows


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    need(Path(path).suffix != '.jsonl', 'JSONL must be streamed')
    with Path(path).open() as stream:
        return json.load(stream)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(CHUNK):
            value.update(block)
    return value.hexdigest()


def raw_check(path, dtype, shape):
    """Same-pass SHA/finite check over unsigned words, without float conversion."""
    need(dtype in DTYPES and isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape),
         'unsupported original dtype/shape')
    width, code, exponent_shift = DTYPES[dtype]
    expected_bytes = math.prod(shape) * width
    need(Path(path).stat().st_size == expected_bytes, 'raw tensor byte count differs from complete shape')
    result, count, first_nonfinite = hashlib.sha256(), 0, None
    with Path(path).open('rb') as stream:
        while block := stream.read(CHUNK):
            need(len(block) % width == 0, 'raw element truncated')
            result.update(block)
            words = array(code); words.frombytes(block)
            if sys.byteorder != 'little':
                words.byteswap()
            if exponent_shift is not None:
                for i, word in enumerate(words):
                    if (word >> exponent_shift) & 255 == 255:
                        if first_nonfinite is None:
                            first_nonfinite = {'element': count + i, 'unsigned_bits': word}
            count += len(words)
    return {'bytes': expected_bytes, 'elements': count, 'sha256': result.hexdigest(),
            'finite': first_nonfinite is None, 'first_nonfinite': first_nonfinite}


def compare_raw(reference, candidate, dtype):
    """Record the first exact bit difference; offsets address original dtype."""
    need(dtype in DTYPES, 'unknown comparison dtype')
    width = DTYPES[dtype][0]
    need(Path(reference).stat().st_size == Path(candidate).stat().st_size, 'raw comparison size mismatch')
    position = 0
    with Path(reference).open('rb') as left, Path(candidate).open('rb') as right:
        while a := left.read(CHUNK):
            b = right.read(len(a))
            if a != b:
                first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
                start = first - first % width
                return {'equal': False, 'byte_offset': position + first,
                    'element': (position + first) // width,
                    'reference_unsigned_bits': int.from_bytes(a[start:start+width], 'little'),
                    'candidate_unsigned_bits': int.from_bytes(b[start:start+width], 'little')}
            position += len(a)
    return {'equal': True, 'bytes': position}


def compare_ranges(bundle, bundle_offset, shard, source_offset, size):
    """Selected raw weight ranges only; no repeat scan of unrelated shard data."""
    remaining, result = size, hashlib.sha256()
    with Path(bundle).open('rb') as a, Path(shard).open('rb') as b:
        a.seek(bundle_offset); b.seek(source_offset)
        while remaining:
            n = min(CHUNK, remaining)
            x, y = a.read(n), b.read(n)
            need(len(x) == len(y) == n and x == y, 'bundle does not contain the selected original shard bytes')
            result.update(x); remaining -= n
    return result.hexdigest()


def audit_bundle(path, expected_sha):
    path = Path(path)
    need(sha(path) == expected_sha, 'fixed bundle manifest SHA')
    manifest = read_json(path); source = manifest['source']
    need(manifest['schema'] == 'E026-layer-weights-v1', 'weight schema')
    for name, expected in {'revision': MODEL_REVISION, 'config_sha256': CONFIG_SHA,
        'index_sha256': INDEX_SHA, 'shard': SHARD, 'shard_sha256': SHARD_SHA,
        'header_sha256': HEADER_SHA, 'header_bytes': HEADER_BYTES,
        'fully_hashed_before_extract': True}.items():
        need(source[name] == expected, 'fixed source identity ' + name)
    for name, expected in (('config.json', CONFIG_SHA), ('model.safetensors.index.json', INDEX_SHA),
                           ('shard-header.json', HEADER_SHA)):
        need(sha(path.parent / name) == expected, 'copied metadata SHA ' + name)
    header, index = read_json(path.parent / 'shard-header.json'), read_json(path.parent / 'model.safetensors.index.json')
    selected = sorted(key for key in header if key.startswith(PREFIX))
    need(len(selected) == 14 and {k[len(PREFIX):].replace('conv1d.weight', 'conv1d_weight') for k in selected} == PARAMETERS,
         'fixed 14 original GDN parameter keys')
    need([r['key'] for r in manifest['tensors']] == selected, 'exact lexicographic tensor selection')
    model_dir = Path(source['model_dir'])
    need(model_dir.name == MODEL_REVISION and sha(model_dir / 'config.json') == CONFIG_SHA
         and sha(model_dir / 'model.safetensors.index.json') == INDEX_SHA, 'original source metadata identity')
    original = model_dir / SHARD
    stat = original.stat()
    actual_stat = {key: getattr(stat, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}
    need(actual_stat == source['fd_before'] == source['fd_after'] and stat.st_size == SHARD_BYTES,
         'original source fd/stat identity changed')
    with original.open('rb') as stream:
        need(struct.unpack('<Q', stream.read(8))[0] == HEADER_BYTES
             and hashlib.sha256(stream.read(HEADER_BYTES)).hexdigest() == HEADER_SHA, 'original safetensors header identity')
    preparation = read_json(path.parent / 'preparation.json')
    need(preparation['status'] == 'PASS' and preparation['full_shard_verified'] is True
         and preparation['manifest_sha256'] == expected_sha and preparation['fd_before'] == source['fd_before']
         and preparation['fd_after'] == source['fd_after'], 'CPU full-source preparation receipt')
    bundle = path.parent / 'weights.bin'
    need(manifest['bundle']['filename'] == 'weights.bin' and manifest['bundle']['bytes'] == BUNDLE_BYTES
         and bundle.stat().st_size == BUNDLE_BYTES and sha(bundle) == manifest['bundle']['sha256']
         == preparation['bundle_sha256'], 'bundle full bytes/SHA')
    cursor, parameters = 0, {}
    for row in manifest['tensors']:
        h = header[row['key']]
        name = row['key'][len(PREFIX):].replace('conv1d.weight', 'conv1d_weight')
        need(row['parameter'] == name and row['dtype'] == h['dtype'] and row['shape'] == h['shape']
             and row['source_offsets'] == h['data_offsets'] and index['weight_map'][row['key']] == SHARD,
             'tensor source header/index mapping')
        size = math.prod(h['shape']) * DTYPES[h['dtype']][0]
        need(row['bytes'] == size == row['source_offsets'][1] - row['source_offsets'][0]
             and row['bundle_offsets'] == [cursor, cursor + size], 'tensor offset/shape byte extents')
        need(compare_ranges(bundle, cursor, original, 8 + HEADER_BYTES + row['source_offsets'][0], size)
             == row['sha256'], 'selected real source tensor SHA')
        dtype, shape, tensor_sha = row['dtype'], list(row['shape']), row['sha256']
        if name == 'conv1d_weight':
            need(shape == [10240, 4, 1] and dtype == 'BF16', 'fixed conv load reshape')
            shape = [10240, 4]
        if name in ('A_log', 'dt_bias'):
            need(dtype == 'BF16' and shape == [48], 'fixed gate scalar dtype/shape')
            with bundle.open('rb') as stream:
                stream.seek(cursor); raw = stream.read(size)
            words = struct.unpack('<48H', raw)
            need(all((w >> 7) & 255 != 255 for w in words), 'gate source nonfinite')
            converted = struct.pack('<48I', *(word << 16 for word in words))
            dtype, size, tensor_sha = 'F32', len(converted), hashlib.sha256(converted).hexdigest()
        parameters[name] = {'dtype': next(k for k, v in CANONICAL.items() if v == dtype),
                            'shape': shape, 'bytes': size, 'sha256': tensor_sha}
        cursor = row['bundle_offsets'][1]
    need(cursor == BUNDLE_BYTES, 'bundle trailing bytes')
    return parameters, {'manifest_sha256': expected_sha, 'bundle_sha256': manifest['bundle']['sha256'],
        'bundle_bytes': BUNDLE_BYTES, 'tensor_count': 14, 'selected_source_bytes_independently_compared': BUNDLE_BYTES,
        'whole_5GiB_shard_rehashed_by_auditor': False,
        'source_scope': 'Preparation full-source SHA receipt plus independent current header/stat/metadata and every selected raw tensor byte'}


def audit_descriptor(directory, descriptor, expected, *, with_file=True):
    need(set(descriptor) == {'sha256', 'bytes', 'dtype', 'shape', 'encoding', 'finite'} | ({'file'} if with_file else set()),
         'raw descriptor fields differ')
    for key in ('dtype', 'shape', 'bytes'):
        need(same(descriptor[key], expected[key]), 'raw descriptor expected ' + key)
    dtype = descriptor['dtype']
    need(descriptor['encoding'] == ENCODINGS[dtype], 'original-dtype unsigned encoding')
    need(descriptor['finite'] is (None if dtype.startswith('uint') else True), 'finite descriptor semantics')
    if 'sha256' in expected:
        need(descriptor['sha256'] == expected['sha256'], 'expected parameter SHA')
    if not with_file:
        return None
    need(not Path(descriptor['file']).is_absolute() and '..' not in Path(descriptor['file']).parts,
         'raw artifact path must be relative')
    file = (directory / descriptor['file']).resolve()
    need(file.is_relative_to(directory.resolve()), 'raw artifact path escaped case directory')
    checked = raw_check(file, CANONICAL[dtype], descriptor['shape'])
    need(checked['sha256'] == descriptor['sha256'] and checked['bytes'] == descriptor['bytes'], 'raw descriptor SHA/bytes')
    need(checked['finite'], 'nonfinite raw payload: ' + json.dumps(checked['first_nonfinite']))
    return file


def expected_seed(kind, count):
    """Yield exact little-endian finite payload bytes in small blocks."""
    need(kind in ('x', 'conv', 'S'), 'unknown synthetic input')
    for base in range(0, count, 8192):
        indexes = range(base, min(count, base + 8192))
        if kind == 'S':
            yield struct.pack('<' + 'I' * len(indexes), *(0x37800000 | ((13*i+37) & 0x7fffff)
                         | (((i+37) & 1) << 31) for i in indexes))
        else:
            exponent, salt = (0x3c00, 11) if kind == 'x' else (0x3b00, 23)
            yield struct.pack('<' + 'H' * len(indexes), *(exponent | ((17*i+salt) & 127)
                         | (((i+salt) & 1) << 15) for i in indexes))


def seed_sha(kind, shape):
    h = hashlib.sha256()
    for block in expected_seed(kind, math.prod(shape)):
        h.update(block)
    return h.hexdigest()


def physical_values(raw, pid):
    need(raw['unit'] == 'byte' and type(raw['bytes per unit']) is int and raw['bytes per unit'] == 1,
         'physical footprint units are not bytes')
    need(raw['errors'] == [] and raw['warnings'] == [], 'physical footprint errors/warnings')
    need(len(raw['processes']) == 1 and raw['processes'][0]['pid'] == pid, 'physical footprint PID ambiguity')
    p = raw['processes'][0]
    current, peak = p['auxiliary']['phys_footprint'], p['auxiliary']['phys_footprint_peak']
    need(type(current) is int and type(peak) is int and 0 <= current == p['footprint'] <= peak <= 2 << 30,
         'physical footprint current/peak violates diagnostic bound')
    return {'pid': pid, 'current_bytes': current, 'peak_bytes': peak}


def owner_footprint(owner, record, pid, owner_pid):
    need(record['pid'] == pid and record['owner_pid'] == owner_pid, 'physical probe owner chain')
    raw_path = owner / record['raw_file']
    need(raw_path.resolve().is_relative_to(owner.resolve()) and record['command'] ==
         ['/usr/bin/footprint', '-p', str(pid), '--noCategories', '-f', 'bytes', '-j', str(raw_path)],
         'actual footprint command')
    if record['status'] == 'terminal-exit0-race-not-measured':
        need(record['physical_pass'] is None and record['child_poll_after_probe'] == 0
             and (record['returncode'] != 0 or not record['raw_exists']), 'invalid terminal-exit race claim')
        if record['raw_exists']:
            need(sha(raw_path) == record['raw_sha256'], 'terminal race raw bytes not preserved')
        return None
    need(record['status'] == 'measured' and record['physical_pass'] is True and record['returncode'] == 0
         and record['raw_exists'] is True and sha(raw_path) == record['raw_sha256'], 'physical measurement receipt')
    parsed = physical_values(read_json(raw_path), pid)
    need(parsed == record['parsed'], 'raw physical parser mismatch')
    return parsed


def audit_resources(owner, owner_pid, children, complete):
    aggregate = read_json(owner / 'resources.json')
    paths = sorted(owner.glob('resource-[0-9][0-9][0-9].json'))
    need(len(paths) == len(aggregate) == complete['samples'], 'resource sample coverage')
    need([p.name for p in paths] == [f'resource-{i:03d}.json' for i in range(len(paths))], 'resource sequence')
    rows = [read_json(p) for p in paths]
    need(rows == aggregate, 'resource aggregate differs from original probes')
    labels = ['preflight'] + [label for arm in ARMS for label in (arm + '/before', arm + '/after')] + ['postflight']
    need([r['label'] for r in rows if not r['label'].endswith('/running')] == labels, 'full arm boundary probes')
    active_arm, min_pressure, physical, races, free_sizes = None, 2, [], 0, []
    running = {arm: 0 for arm in ARMS}
    first = rows[0]
    for i, row in enumerate(rows):
        pressure = int(row['pressure_level']['text'].strip())
        need(row['pressure_level']['returncode'] == 0 and pressure == row['pressure_level_value']
             and pressure in (1, 2) and pressure <= min_pressure, 'pressure unknown/critical/increased')
        min_pressure = pressure
        vm = row['vm_stat']['text']
        page = int(re.search(r'page size of (\d+) bytes', vm)[1])
        free = int(re.search(r'^Pages free:\s*(\d+)\.', vm, re.M)[1]) * page
        need(row['vm_stat']['returncode'] == 0 and free == row['real_free_bytes'] and free >= 50 << 30,
             'physical free pages below threshold')
        free_sizes.append(free)
        for key, value in row['vm_counters'].items():
            need(int(re.search('^' + key + r':\s*(\d+)', vm, re.M)[1]) == value, 'raw VM counter mismatch')
        need(row['vm_counters']['Swapouts'] == first['vm_counters']['Swapouts'], 'new/reset Swapouts')
        used = float(re.search(r'used = ([\d.]+)M', row['swap']['text'])[1])
        need(row['swap']['returncode'] == 0 and 0 <= used == row['swap_used_mb'] <= first['swap_used_mb'] + 16,
             'swap growth/unknown')
        power, thermal = row['power_metadata'], row['processinfo_thermal_metadata']
        need(power['status'] == thermal['status'] == 'known' and power['source'] == 'AC Power'
             and power['raw']['returncode'] == thermal['raw']['returncode'] == 0
             and 'AC Power' in power['raw']['stdout'] and thermal['code'] in (0, 1)
             and thermal['raw']['stdout'].strip() == str(thermal['code']), 'power/thermal raw evidence')
        need(row['native_process_inventory_ok'] is True and row['process_inventory_ok'] is True
             and row['authorized_suspended_pids'] == [74384], 'inventory unknown')
        service = [p for p in row['native_inference_processes'] if p['pid'] == 74384]
        need(len(service) == 1 and service[0]['authorized_suspended'] is True and 'T' in service[0]['state'],
             'authorized external service not stopped')
        label = row['label']; arm = label.split('/')[0] if '/' in label else None
        pid = children[arm] if label.endswith('/running') else None
        if label.endswith('/before'):
            need(active_arm is None, 'overlapping arm windows'); active_arm = arm
        elif label.endswith('/running'):
            need(active_arm == arm, 'running outside arm window'); running[arm] += 1
            need(row['owned_model_pid'] == pid and row['physical_receipt'] == f'footprint-{i:03d}.json', 'physical target/sample binding')
            receipt = read_json(owner / row['physical_receipt'])
            measured = owner_footprint(owner, receipt, pid, owner_pid)
            if measured is None:
                need(row['terminal_exit0_observed'] is True, 'terminal-race exit not observed')
                races += 1
            else:
                physical.append(measured)
            if not row['terminal_exit0_observed']:
                exact = row['exact_owned_pid_metadata']
                need(exact['status'] == 'known' and exact['pid'] == pid and exact['ppid'] == owner_pid,
                     'direct live child metadata missing')
        elif label.endswith('/after'):
            need(active_arm == arm, 'after outside arm window'); active_arm = None
        for p in row['native_inference_processes'] + row['known_inference_processes']:
            need(p['pid'] in (74384, pid), 'foreign inference process')
            if p['pid'] == pid:
                need(p['ppid'] == owner_pid and p['owned'] is True, 'foreign child ownership')
        if i:
            need(row['monotonic_s'] > rows[i-1]['monotonic_s'], 'nonmonotonic resource probes')
    need(all(running.values()) and complete['physical_samples'] == sum(running.values()) == len(physical) + races,
         'physical/arm probe coverage')
    return {'samples': len(rows), 'physical_measured': len(physical), 'terminal_races_not_measurements': races,
        'minimum_free_bytes': min(free_sizes), 'maximum_physical_peak_bytes': max((p['peak_bytes'] for p in physical), default=None),
        'initial_pressure': first['pressure_level_value'], 'final_pressure': rows[-1]['pressure_level_value'],
        'swapouts_growth': 0, 'max_swap_growth_mib': max(r['swap_used_mb'] for r in rows) - first['swap_used_mb'],
        'scope': 'Single-layer numerical policy only; no hard allocation bound or pressure-1 performance qualification'}


def expected_tensor(dtype, shape, digest=None):
    result = {'dtype': dtype, 'shape': shape, 'bytes': math.prod(shape) * DTYPES[CANONICAL[dtype]][0]}
    if digest is not None:
        result['sha256'] = digest
    return result


def audit_case(out, row, arm, length, kind, parameters, witness, gdn_sha, progress):
    case = f'T{length}-{kind}'; offset = 7 if kind == 'continuing' else 0
    need(same(row, read_json(out / 'cases' / case / 'case.json')), 'case receipt differs from complete')
    need(same([row['case'], row['status'], row['length'], row['kind'], row['layer_index']],
              [case, 'PASS', length, kind, 0]), 'case identity/status')
    need(same([row['offset_before'], row['offset_after']], [offset, offset + length]), 'logical cache offset')
    need(set(row['input']) == set(row['input_after']) == {'x', 'conv_seed', 'S_seed'}, 'all initial inputs required')
    paths = {}
    for name, seed_kind, dtype, shape in (('x', 'x', 'bfloat16', [length, 5120]),
        ('conv_seed', 'conv', 'bfloat16', [3, 10240]), ('S_seed', 'S', 'float32', [48, 128, 128])):
        item = row['input'][name]
        if kind == 'fresh' and name != 'x':
            need(item is None and row['input_after'][name] is None, 'fresh state must be absent')
            continue
        expected = expected_tensor(dtype, shape, seed_sha(seed_kind, shape))
        paths['input/' + name] = audit_descriptor(out, item, expected)
        audit_descriptor(out, row['input_after'][name], expected, with_file=False)
        need(same(nofile(item), row['input_after'][name]), 'initial input/seed changed during layer')
    shapes = {'output': ('bfloat16', [length, 5120]), 'conv_y': ('bfloat16', [length, 10240]),
        'conv_state': ('bfloat16', [3, 10240]), 'S': ('float32', [48, 128, 128]),
        'original_conv_tail': ('bfloat16', [3, 10240])}
    need(set(row['arrays']) == set(shapes), 'all five complete arrays required')
    for name, (dtype, shape) in shapes.items():
        paths['arrays/' + name] = audit_descriptor(out, row['arrays'][name], expected_tensor(dtype, shape))
    difference = compare_raw(paths['arrays/original_conv_tail'], paths['arrays/conv_state'], 'BF16')
    if not difference['equal']:
        progress['first_difference'] = dict(arm=arm, case=case, field='original tail vs retained tail', **difference)
    need(difference['equal'], 'candidate tail changed actual original convolution bits')
    for field in ('parameters_before', 'parameters_after'):
        need(set(row[field]) == set(parameters), 'case parameter set')
        for name, expected in parameters.items():
            audit_descriptor(out, row[field][name], expected, with_file=False)
    required = dict(sequence=1, layer=0, route='single-layer-eager', input_shape=[length, 10240],
        tail_shape=[3, 10240], input_dtype='mlx.core.bfloat16', tail_dtype='mlx.core.bfloat16',
        offset=offset, state_present=kind == 'continuing', eligible=True, enabled=arm != 'off',
        materialize=arm == 'on-materialize', replaced=arm != 'off', y_identity_preserved=True)
    need(same(row['hook_witness'], required) and same(witness, dict(case=case, **required)), 'actual hook route/identity witness')
    need(same(row['profiler'], dict(count=1, restored=True, original_source_sha256=gdn_sha,
        observer='exact-original-code-return')), 'actual original convolution observer/source/restoration')
    progress['cases_validated'] = progress.get('cases_validated', 0) + 1
    return paths


def audit_phases(out, result, pid, owner_pid, arm_window=None):
    rows = jsonl(out / 'phases.jsonl')
    expected = [('global', 'after_weights_load'), ('global', 'after_parameters_before')]
    expected += [(case, phase) for case in CASE_IDS for phase in CASE_PHASES]
    expected += [('global', 'after_parameters_after'), ('global', 'after_layer_release')]
    need(len(rows) == len(result['physical_receipts']) == len(expected) == 68, 'exact 68 memory/physical phases')
    if arm_window is not None:
        need(arm_window[0] <= rows[0]['monotonic_s'] <= rows[-1]['monotonic_s'] <= arm_window[1],
             'phase observations outside actual child spawn/deadline')
    previous_time = previous_elapsed = previous_peak = previous_physical = -1
    chart = []
    for index, (row, pair) in enumerate(zip(rows, expected)):
        need(same([row['index'], row['case'], row['phase'], row['pid'], row['ppid']],
                  [index, *pair, pid, owner_pid]) and row.get('status', 'PASS') == 'PASS'
             and row['groups_are_evaluated'] is True, 'phase order/identity/status')
        now, elapsed = row['monotonic_s'], row['elapsed_s']
        need(type(now) in (int, float) and math.isfinite(now) and now >= previous_time
             and type(elapsed) in (int, float) and math.isfinite(elapsed)
             and previous_elapsed <= elapsed <= result['elapsed_wall_s'], 'phase monotonic times')
        previous_time, previous_elapsed = now, elapsed
        for key in ('active_bytes', 'peak_bytes', 'cache_bytes', 'unique_buffer_bytes'):
            need(type(row[key]) is int and row[key] >= 0, 'MLX observation integer bounds')
        need(row['active_bytes'] + row['cache_bytes'] <= 1 << 30
             and max(previous_peak, row['active_bytes']) <= row['peak_bytes'] <= 1 << 30
             and row['unique_buffer_bytes'] <= row['active_bytes'], 'MLX observed boundary/peak/union')
        previous_peak = row['peak_bytes']
        groups = row['group_buffer_bytes']
        phase = row['phase']
        group_names = ({'persistent_norm_eps'} if phase == 'after_layer_release' else
            {'weights', 'output', 'state'} if phase == 'after_drop_observer_refs' else
            {'weights', 'inputs', 'products'} if phase in ('after_common_validation_eval', 'after_raw_artifacts') else
            {'weights', 'inputs', 'output'} if phase == 'after_normal_product_eval' else
            {'weights', 'inputs'} if phase in ('before_prefill', 'after_hook', 'after_prefill_build') else {'weights'})
        need(set(groups) == group_names and all(type(v) is int and 0 <= v <= row['unique_buffer_bytes'] for v in groups.values())
             and row['unique_buffer_bytes'] <= sum(groups.values()), 'evaluated group/union evidence')
        raw = out / f'physical-{index:03d}.raw.json'
        command = read_json(out / f'physical-{index:03d}.command.json')
        need(command['command'] == ['/usr/bin/footprint', '-p', str(pid), '--noCategories', '-f', 'bytes', '-j', str(raw)]
             and command['returncode'] == 0 and isinstance(command['stdout'], str) and isinstance(command['stderr'], str),
             'self physical probe command/exit')
        receipt = dict(path=raw.name, sha256=sha(raw), pid=pid, case=pair[0], phase=pair[1])
        need(same(row['physical_receipt'], receipt) and same(result['physical_receipts'][index], receipt), 'self physical raw identity')
        physical = physical_values(read_json(raw), pid)
        need(same([row['physical_current_bytes'], row['physical_peak_bytes']], [physical['current_bytes'], physical['peak_bytes']])
             and physical['peak_bytes'] >= previous_physical, 'self physical historical peak')
        previous_physical = physical['peak_bytes']
        chart.append({key: row[key] for key in ('index', 'case', 'phase', 'active_bytes', 'peak_bytes', 'cache_bytes',
            'unique_buffer_bytes', 'group_buffer_bytes', 'physical_current_bytes', 'physical_peak_bytes')})
    release = result['release']
    need(release['layer_weakref_dead'] is True and type(result['initial_active_bytes']) is int
         and 0 <= result['initial_active_bytes'] <= 1 << 30
         and same(release['initial_active_bytes'], result['initial_active_bytes'])
         and same(release['final_active_bytes'], rows[-1]['active_bytes'])
         and same(release['active_delta_bytes'], rows[-1]['active_bytes'] - result['initial_active_bytes'])
         and same(release['known_persistent_buffer_bytes'], rows[-1]['unique_buffer_bytes']), 'release weakref/active/global-root evidence')
    return chart


def audit_child(out, arm, pid, owner_pid, plan, parameters, progress, arm_window):
    result = read_json(out / 'complete.json')
    need(same([result['status'], result['arm'], result['pid'], result['ppid'], result['device_run']],
              ['PASS', arm, pid, owner_pid, True]), 'child completed identity')
    need(result['source_pins_rechecked'] is True and result['bundle_rechecked'] is True
         and result['performance_qualified'] is False and result['cross_arm_equality'] == 'PENDING_INDEPENDENT_RAW_AUDIT',
         'child final recheck/limited claim')
    need(type(result['elapsed_wall_s']) in (int, float) and 0 < result['elapsed_wall_s'] <= 120, 'child observed deadline')
    required = dict(schema='E026-real-weight-layer-v1', arm=arm, layer_index=0, lengths=list(LENGTHS),
        cache_kinds=list(MODES), cases=8, case_ids=list(CASE_IDS), cache_limit_bytes=0,
        mlx_observation_budget_bytes=1 << 30, physical_observation_budget_bytes=2 << 30,
        child_alarm_seconds=120, phase_count=68, case_phases=list(CASE_PHASES), continuing_offset=7,
        continuing_prefix_forward=False)
    need(same({k: result['contract'][k] for k in required}, required), 'fixed child contract')
    pins = plan['execution']['pins']; package = Path(plan['package_root'])
    manifest_path = package.parent / 'native-source.json'
    need(pins[str(manifest_path)] == CORE_MANIFEST_SHA == sha(manifest_path), 'fixed frozen core source manifest')
    source_manifest = read_json(manifest_path)
    expected_sources = {str((package / 'tutorial_code/miniengine' / k).resolve()): v for k, v in source_manifest['files'].items()}
    screen = Path(plan['execution']['screen']['path'])
    expected_sources.update({str(screen.parent / k): v for k, v in SUPPORT_PINS.items()})
    expected_sources[str(screen)] = plan['execution']['screen']['sha256']
    expected_sources[str(manifest_path)] = CORE_MANIFEST_SHA
    need(same(result['sources'], expected_sources) and all(pins.get(p) == h for p, h in expected_sources.items()),
         'loaded core/support source set and SHA')
    for field in ('mlx_file', 'numpy_file'):
        need(str(Path(result['runtime'][field]).resolve()) in {str(Path(p).resolve()) for p in pins}, 'loaded runtime outside source pins')
    need(result['runtime']['device']['device_name'] == 'Apple M5 Ultra'
         and result['runtime']['metal']['architecture'] == 'applegpu_g17d', 'actual backend identity')
    need(result['bundle_manifest_sha256'] == plan['bundle']['sha256']
         and result['bundle_sha256'] == read_json(plan['bundle']['path'])['bundle']['sha256'], 'actual weight bundle identity')
    paths = {}
    for field in ('parameters_before', 'parameters_after'):
        need(set(result[field]) == set(parameters), '14 actual loaded parameters')
        for name, expected in parameters.items():
            paths[field + '/' + name] = audit_descriptor(out, result[field][name], expected)
    need([row['case'] for row in result['cases']] == list(CASE_IDS), 'eight exact ordered cases')
    witnesses = jsonl(out / 'witness.jsonl'); need(len(witnesses) == 8, 'eight actual hook witnesses')
    gdn_sha = expected_sources[str(package / 'tutorial_code/miniengine/model/gdn.py')]
    for i, row in enumerate(result['cases']):
        progress['current_case'] = dict(arm=arm, case=CASE_IDS[i])
        checked = audit_case(out, row, arm, LENGTHS[i // 2], MODES[i % 2], parameters, witnesses[i], gdn_sha, progress)
        paths.update({CASE_IDS[i] + '/' + k: v for k, v in checked.items()})
    chart = audit_phases(out, result, pid, owner_pid, arm_window)
    progress.setdefault('arms_validated', []).append(arm)
    return {'arm': arm, 'pid': pid, 'cases': 8, 'phases': 68, 'arrays_raw_checked': len(paths),
        'raw_bytes_checked': sum(p.stat().st_size for p in paths.values()),
        'complete_sha256': sha(out / 'complete.json'), 'release': result['release'], 'memory_chart_rows': chart}, paths


def cross_compare(left, right, arm, parameters, progress):
    need(set(left) == set(right), 'cross-arm complete parameter/input/product sets')
    total = 0
    for field in left:
        # Width is known from the independent field contract, not producer claims.
        if field.startswith(('parameters_before/', 'parameters_after/')):
            dtype = CANONICAL[parameters[field.split('/', 1)[1]]['dtype']]
        else:
            dtype = 'F32' if field.endswith(('/S', '/S_seed')) else 'BF16'
        comparison = compare_raw(left[field], right[field], dtype)
        if not comparison['equal']:
            progress['first_difference'] = dict(arm=arm, field=field, reference=str(left[field]), candidate=str(right[field]), **comparison)
        need(comparison['equal'], 'cross-arm raw bytes differ: ' + field)
        total += comparison['bytes']
    return {'arm': arm, 'arrays': len(left), 'bytes': total, 'equal': True}


def verify_plan_pins(owner, plan):
    pins = plan['execution']['pins']
    before, after = read_json(owner / 'verified-pins.json'), read_json(owner / 'pins-after.json')
    need(set(before) == set(after) == set(pins), 'complete before/after source pin coverage')
    for path, expected in pins.items():
        need(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}', expected), 'absolute SHA source pin')
        copied = Path(path).suffix in ('.py', '.json', '.md', '.h', '.cpp', '.txt')
        a, b = before[path], after[path]
        need(a['sha256'] == b['sha256'] == expected and a['source_copied'] is copied and b['source_copied'] is False
             and same(a['stat'], b['stat']), 'before/after source stat/SHA changed: ' + path)
        evidence = owner / 'frozen-source' / path.lstrip('/') if copied else Path(path)
        need(evidence.stat().st_size == a['stat']['size'] and sha(evidence) == expected, 'independent source hash: ' + path)
    screen = plan['execution']['screen']
    need(pins[screen['path']] == screen['sha256'] == SCREEN_SHA
         and pins[str(Path(screen['path']).parent / 'layer_owner.py')] == OWNER_SHA, 'frozen screen/owner source binding')
    bundle = Path(plan['bundle']['path'])
    need(pins[str(bundle)] == plan['bundle']['sha256'] and pins[str(bundle.parent / 'weights.bin')]
         == read_json(bundle)['bundle']['sha256'], 'bundle identity outside execution pins')
    return len(pins)


def audit_lease_command(command, owner, plan, plan_sha256):
    """Bind the external service lease to this exact owner execution."""
    owner_source = str(Path(plan['execution']['screen']['path']).parent / 'layer_owner.py')
    need(type(command) is list and len(command) == 12 and command[1:4] == ['-I', '-B', owner_source],
         'outer lease did not execute the fixed isolated layer owner')
    need(str(Path(command[0]).resolve()) in {str(Path(p).resolve()) for p in plan['execution']['pins']},
         'outer owner Python outside execution pins')
    flags = dict(zip(command[4::2], command[5::2]))
    need(set(flags) == {'--plan', '--plan-sha256', '--out', '--suspended-pid'}
         and flags['--plan-sha256'] == plan_sha256 and flags['--out'] == str(owner)
         and flags['--suspended-pid'] == '74384', 'outer lease owner plan/output/service argv binding')
    original_plan = Path(flags['--plan'])
    need(original_plan.is_absolute() and sha(original_plan) == plan_sha256 and same(read_json(original_plan), plan),
         'external owner actual plan path/SHA differs')


def audit(owner, lease, plan_sha256, progress):
    need(re.fullmatch('[0-9a-f]{64}', plan_sha256), 'explicit frozen plan SHA required')
    required = [owner / 'complete.json', owner / 'resources.json', owner / 'pins-after.json',
                lease / 'lease.json', lease / 'resume.json', lease / 'job-0.json']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return dict(status='INCOMPLETE', qualified=False, missing_terminal_evidence=missing)
    complete, lease_data = read_json(required[0]), read_json(lease / 'lease.json')
    progress['recorded_owner_status'] = complete['status']
    progress['recorded_arm_statuses'] = [{k: r.get(k) for k in ('arm', 'pid', 'status', 'cases')} for r in complete['arms']]
    if not lease_data['closed']:
        return dict(status='INCOMPLETE', qualified=False, reason='external service lease not closed')
    need(complete['schema'] == 'E026-layer-owner-result-v1' and complete['status'] == 'PASS'
         and complete['errors'] == [] and 0 < complete['elapsed_wall_s'] <= 480 and complete['pins_rechecked'] is True,
         'owner did not complete numerical resource/lifecycle contract')
    owner_pid = complete['owner_pid']; plan = read_json(owner / 'plan.json')
    need(sha(owner / 'plan.json') == plan_sha256 and plan['schema'] == 'E026-layer-plan-v1'
         and same([plan[k] for k in ('arms', 'arm_seconds', 'work_seconds', 'cleanup_seconds', 'total_seconds')],
                  [list(ARMS), 120, 450, 30, 480]) and same(plan['resource_policy'], POLICY)
         and same(plan['execution']['environment'], ENV), 'fixed plan identity/budgets/resource policy/environment')
    attempt = read_json(owner / 'attempt.json')
    need(attempt['owner_pid'] == owner_pid and attempt['plan_sha256'] == plan_sha256
         and same([attempt[k] for k in ('arm_seconds', 'work_seconds', 'cleanup_seconds', 'total_seconds')], [120, 450, 30, 480]),
         'owner launch identity/budgets')
    progress['execution_pins_verified'] = verify_plan_pins(owner, plan)
    parameters, bundle = audit_bundle(Path(plan['bundle']['path']), plan['bundle']['sha256'])
    progress['bundle'] = bundle
    children, results, paths, releases = {}, [], {}, []
    for arm in ARMS:
        directory = owner / arm; child = read_json(directory / 'child.json'); release = read_json(directory / 'release.json')
        pid = child['pid']; children[arm] = pid
        command = [plan['execution']['python'], '-I', '-B', plan['execution']['screen']['path'], '--run', '--arm', arm,
            '--bundle-manifest', plan['bundle']['path'], '--bundle-sha256', plan['bundle']['sha256'],
            '--package-root', plan['package_root'], '--out', str(directory / 'screen')]
        need(child['arm'] == arm and child['ppid'] == owner_pid and child['command'] == command
             and 0 < child['arm_deadline_monotonic'] - child['spawned_monotonic'] <= 120, 'direct child command/budget')
        need(release['pid'] == pid and release['returncode'] == 0 and release['waited_reaped'] is True
             and release['strict_absence'] is True and release['cleanup_notes'] == []
             and release['absence_probe']['returncode'] == 1 and not release['absence_probe']['stdout'].strip(), 'direct child zero exit/reap/strict absence')
        checked, paths[arm] = audit_child(directory / 'screen', arm, pid, owner_pid, plan, parameters, progress,
            (child['spawned_monotonic'], child['arm_deadline_monotonic']))
        results.append(checked); releases.append(release)
    need(len(set(children.values())) == 3, 'three fresh processes required')
    need([r['arm'] for r in complete['arms']] == list(ARMS), 'owner full ordered arm summary')
    for summary, checked in zip(complete['arms'], results):
        need(summary['status'] == 'PASS' and summary['pid'] == checked['pid'] and summary['cases'] == 8
             and summary['physical_phases'] == 68 and summary['complete_sha256'] == checked['complete_sha256'], 'owner raw result summary')
    comparisons = [cross_compare(paths['off'], paths[arm], arm, parameters, progress) for arm in ARMS[1:]]
    resources = audit_resources(owner, owner_pid, children, complete)
    job, resume = read_json(lease / 'job-0.json'), read_json(lease / 'resume.json')
    need(lease_data['child_pid'] == job['pid'] == owner_pid and len(lease_data['jobs']) == 1
         and job['command'] == lease_data['jobs'][0] and job['returncode'] == 0 and job['waited'] is True, 'external lease owner exit/wait')
    audit_lease_command(job['command'], owner, plan, plan_sha256)
    need(lease_data['service_pid'] == 74384 and lease_data['paused_at'] < resume['time'] <= lease_data['deadline']
         and resume['identity_match'] is True and 'T' in resume['state_before']
         and resume['state_after'] and 'T' not in resume['state_after'], 'external service same identity restored')
    return dict(status='PASS', qualified=True, qualification_scope='Single real-weight GDN layer, synthetic finite input, numerical/state/backing diagnostic only',
        performance_qualified=False, full_model_qualified=False, plan_sha256=plan_sha256, owner_pid=owner_pid,
        execution_pins_verified=progress['execution_pins_verified'], bundle=bundle, arms=results,
        cross_arm=comparisons, resources=resources, releases=releases, service_restore=resume,
        chartdata=dict(kind='memory_only_not_pareto', units='bytes', rows=[dict(arm=r['arm'], **row) for r in results for row in r['memory_chart_rows']]),
        limits='Continuing state is synthetic offset 7, not prefix execution. Normal product and common-validation phases remain distinct; observer roots affect storage. No full-model/D/HTTP/Pareto/performance/adoption or E023-pressure-cause claim. Observed peaks are not hard allocator bounds.')


def self_test():
    checks = 0
    def rejected(function):
        nonlocal checks
        try:
            function()
        except (ValueError, KeyError, TypeError, OSError):
            checks += 1
        else:
            raise AssertionError('corrupted evidence accepted')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for dtype, finite, nonfinite in (
            ('BF16', [0, 0x8000, 0x3f80, 0xbf80], [0x7f80, 0x7fc1]),
            ('F32', [0, 0x80000000, 0x3f800000, 0xbf800000], [0x7f800000, 0x7fc00001]),
        ):
            code = DTYPES[dtype][1]
            a, b = root / 'left.bin', root / 'right.bin'
            payload = struct.pack('<' + code * len(finite), *finite)
            a.write_bytes(payload); b.write_bytes(payload)
            need(raw_check(a, dtype, [4])['finite'], 'finite signed zeros rejected')
            need(compare_raw(a, b, dtype)['equal'], 'identical raw bits differ')
            changed = bytearray(payload); changed[-1] ^= 1; b.write_bytes(changed)
            diff = compare_raw(a, b, dtype)
            need(not diff['equal'] and diff['element'] == 3, 'first difference location wrong')
            a.write_bytes(struct.pack('<' + code * 2, *nonfinite))
            need(raw_check(a, dtype, [2])['first_nonfinite']['element'] == 0, 'nonfinite payload accepted')
        a, b = root / 'bundle', root / 'shard'
        a.write_bytes(b'xxabcdyy'); b.write_bytes(b'prefixabcdsuffix')
        need(compare_ranges(a, 2, b, 6, 4) == hashlib.sha256(b'abcd').hexdigest(), 'selected weight offsets')
        rejected(lambda: compare_ranges(a, 1, b, 6, 4))
        need(not same(True, 1), 'typed JSON equality')
        path = root / 'truncated.jsonl'; path.write_text('{"x":1}')
        rejected(lambda: jsonl(path))
        path.write_text('{"x":1}\n\n'); rejected(lambda: jsonl(path))
        def dump(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) + '\n')
        def record(name, dtype, shape, payload):
            target = root / (name + '.bin'); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
            return dict(file=str(target.relative_to(root)), sha256=sha(target), **expected_tensor(dtype, shape),
                encoding=ENCODINGS[dtype], finite=None if dtype.startswith('uint') else True)
        params = {name: expected_tensor('uint8', [1], hashlib.sha256(b'\x01').hexdigest()) for name in PARAMETERS}
        paramrows = {name: dict(**value, encoding='uint8', finite=None) for name, value in params.items()}
        x = record('cases/T1-fresh/input-x', 'bfloat16', [1, 5120], b''.join(expected_seed('x', 5120)))
        row = dict(case='T1-fresh', status='PASS', length=1, kind='fresh', layer_index=0, offset_before=0, offset_after=1,
            input=dict(x=x, conv_seed=None, S_seed=None), input_after=dict(x=nofile(x), conv_seed=None, S_seed=None),
            parameters_before=paramrows, parameters_after=copy.deepcopy(paramrows))
        hook = dict(sequence=1, layer=0, route='single-layer-eager', input_shape=[1, 10240], tail_shape=[3, 10240],
            input_dtype='mlx.core.bfloat16', tail_dtype='mlx.core.bfloat16', offset=0, state_present=False,
            eligible=True, enabled=False, materialize=False, replaced=False, y_identity_preserved=True)
        row.update(hook_witness=hook, profiler=dict(count=1, restored=True, original_source_sha256='gdn', observer='exact-original-code-return'))
        row['arrays'] = {name: record('cases/T1-fresh/' + name, dtype, shape,
            b'\0' * expected_tensor(dtype, shape)['bytes']) for name, dtype, shape in (
            ('output', 'bfloat16', [1, 5120]), ('conv_y', 'bfloat16', [1, 10240]),
            ('conv_state', 'bfloat16', [3, 10240]), ('S', 'float32', [48, 128, 128]),
            ('original_conv_tail', 'bfloat16', [3, 10240]))}
        def case_check(value):
            dump(root / 'cases/T1-fresh/case.json', value)
            return audit_case(root, value, 'off', 1, 'fresh', params, dict(case='T1-fresh', **value['hook_witness']), 'gdn', {})
        case_check(row)
        for change in ('shape', 'seed', 'hook', 'offset', 'parameter', 'nonfinite', 'tail', 'input_after'):
            altered = copy.deepcopy(row)
            if change == 'shape': altered['arrays']['S']['shape'] = [1, 48, 128, 128]
            elif change == 'seed':
                payload = bytearray(b''.join(expected_seed('x', 5120))); payload[0] ^= 1
                altered['input']['x'] = record('changed-seed', 'bfloat16', [1, 5120], payload)
                altered['input_after']['x'] = nofile(altered['input']['x'])
            elif change == 'hook': altered['hook_witness']['enabled'] = True
            elif change == 'offset': altered['offset_after'] = 2
            elif change == 'parameter': altered['parameters_after']['A_log']['sha256'] = '0' * 64
            elif change == 'input_after': altered['input_after']['conv_seed'] = nofile(x)
            elif change == 'nonfinite':
                altered['arrays']['output'] = record('badfinite', 'bfloat16', [1, 5120], b'\xc1\x7f' + b'\0' * 10238)
            else: altered['arrays']['conv_state'] = record('changed-tail', 'bfloat16', [3, 10240], b'\x01\x00' + b'\0' * 61438)
            rejected(lambda value=altered: case_check(value))
        # Full phase fixture is metadata only. These are not device observations.
        pairs = [('global', 'after_weights_load'), ('global', 'after_parameters_before')]
        pairs += [(case, phase) for case in CASE_IDS for phase in CASE_PHASES]
        pairs += [('global', 'after_parameters_after'), ('global', 'after_layer_release')]
        phases, receipts = [], []
        for i, (case, phase) in enumerate(pairs):
            raw = root / f'physical-{i:03d}.raw.json'
            dump(raw, {'unit': 'byte', 'bytes per unit': 1, 'errors': [], 'warnings': [], 'processes':
                [{'pid': 23, 'footprint': 200, 'auxiliary': {'phys_footprint': 200, 'phys_footprint_peak': 300}}]})
            dump(root / f'physical-{i:03d}.command.json', dict(command=['/usr/bin/footprint', '-p', '23', '--noCategories', '-f', 'bytes', '-j', str(raw)], returncode=0, stdout='', stderr=''))
            names = ({'persistent_norm_eps'} if phase == 'after_layer_release' else {'weights', 'output', 'state'} if phase == 'after_drop_observer_refs'
                else {'weights', 'inputs', 'products'} if phase in ('after_common_validation_eval', 'after_raw_artifacts')
                else {'weights', 'inputs', 'output'} if phase == 'after_normal_product_eval'
                else {'weights', 'inputs'} if phase in ('before_prefill', 'after_hook', 'after_prefill_build') else {'weights'})
            receipt = dict(path=raw.name, sha256=sha(raw), pid=23, case=case, phase=phase); receipts.append(receipt)
            phases.append(dict(index=i, case=case, phase=phase, pid=23, ppid=22, monotonic_s=100+i/100,
                elapsed_s=i/100, active_bytes=100, peak_bytes=100, cache_bytes=0, groups_are_evaluated=True,
                unique_buffer_bytes=100, group_buffer_bytes={k: 100 for k in names}, physical_receipt=receipt,
                physical_current_bytes=200, physical_peak_bytes=300))
        complete = dict(elapsed_wall_s=1, physical_receipts=receipts, initial_active_bytes=0,
            release=dict(layer_weakref_dead=True, initial_active_bytes=0, final_active_bytes=100,
                         active_delta_bytes=100, known_persistent_buffer_bytes=100))
        def phase_check(values, final=complete):
            (root / 'phases.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in values))
            return audit_phases(root, final, 23, 22)
        need(len(phase_check(phases)) == 68, 'synthetic phase fixture invalid')
        for change in ('missing', 'order', 'peak', 'group', 'pid', 'time', 'release'):
            changed, final = copy.deepcopy(phases), copy.deepcopy(complete)
            if change == 'missing': changed.pop()
            elif change == 'order': changed[2], changed[3] = changed[3], changed[2]
            elif change == 'peak': changed[-1]['peak_bytes'] = (1 << 30) + 1
            elif change == 'group': changed[3]['group_buffer_bytes']['pending_unevaluated'] = 1
            elif change == 'pid': changed[-1]['physical_receipt']['pid'] = 99
            elif change == 'time': changed[-1]['elapsed_s'] = float('nan')
            else: final['release']['layer_weakref_dead'] = False
            rejected(lambda values=changed, value=final: phase_check(values, value))
        phase_check(phases)
        rejected(lambda: audit_phases(root, complete, 23, 22, (101, 102)))
        rejected(lambda: audit_phases(root, complete, 23, 22, (99, 100.1)))
        physical = read_json(root / 'physical-000.raw.json')
        for change in ('unit', 'warnings', 'pid', 'budget'):
            bad = copy.deepcopy(physical)
            if change == 'unit': bad['bytes per unit'] = 1024
            elif change == 'warnings': bad['warnings'] = ['partial']
            elif change == 'pid': bad['processes'][0]['pid'] = 24
            else: bad['processes'][0]['auxiliary']['phys_footprint_peak'] = (2 << 30) + 1
            rejected(lambda value=bad: physical_values(value, 23))
        # Raw cross-arm differences preserve original BF16 element and bits.
        a, b = root / 'lhs', root / 'rhs'; a.write_bytes(b'\0\0\0\0'); b.write_bytes(b'\0\0\1\0')
        partial = {}
        rejected(lambda: cross_compare({'T1-fresh/arrays/output': a}, {'T1-fresh/arrays/output': b}, 'on-lazy', {}, partial))
        need(partial['first_difference']['element'] == 1 and partial['first_difference']['candidate_unsigned_bits'] == 1, 'cross-arm original dtype first difference')
        plan = {'execution': {'screen': {'path': str(root / 'layer_screen.py')}, 'pins': {str(root / 'python'): 'synthetic'}}}
        dump(root / 'plan.json', plan); plan_sha = sha(root / 'plan.json')
        command = [str(root / 'python'), '-I', '-B', str(root / 'layer_owner.py'), '--plan', str(root / 'plan.json'),
            '--plan-sha256', plan_sha, '--out', str(root / 'owner'), '--suspended-pid', '74384']
        audit_lease_command(command, root / 'owner', plan, plan_sha)
        for index in (0, 3, 5, 7, 9, 11):
            changed = command[:]; changed[index] += '-wrong'
            rejected(lambda value=changed: audit_lease_command(value, root / 'owner', plan, plan_sha))
    print(f'PASS stdlib mechanisms and {checks} corruption rejections; synthetic fixtures only, no device run')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--bundle-manifest', type=Path)
    parser.add_argument('--bundle-sha256')
    parser.add_argument('--owner', type=Path)
    parser.add_argument('--lease', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    full = all((args.owner, args.lease, args.plan_sha256))
    bundle_only = bool(args.bundle_manifest and args.bundle_sha256)
    if not args.out or full == bundle_only or (full and (args.bundle_manifest or args.bundle_sha256)) or (bundle_only and (args.owner or args.lease or args.plan_sha256)):
        parser.error('--out plus either --owner --lease --plan-sha256 or --bundle-manifest --bundle-sha256 required')
    need(not args.out.exists(), 'refusing to overwrite prior audit evidence')
    progress = {}
    try:
        if bundle_only:
            parameters, checked = audit_bundle(args.bundle_manifest.resolve(), args.bundle_sha256)
            result = dict(status='PASS', qualified=False, bundle_qualified=True, bundle=checked, expected_loaded_parameters=parameters,
                scope='CPU raw weight bundle/source identity only; no layer execution or performance qualification')
        else:
            result = audit(args.owner.resolve(), args.lease.resolve(), args.plan_sha256, progress)
    except Exception as error:
        result = dict(status='FAIL', qualified=False, error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), partial=progress)
    result.update(audit_source_sha256=sha(__file__), performance_qualified=False, full_model_qualified=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'qualified', 'performance_qualified')}))
    return int(result['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
