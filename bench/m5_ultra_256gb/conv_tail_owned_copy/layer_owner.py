"""E026-layer-v2: three fresh, bounded real-weight single-GDN diagnostics.

Stdlib import/help only. No service control: the outer suspended_service lease
alone stops/restores 74384. This exception permits existing pressure 2 only for
the independently registered layer numerical diagnostic, never a full model or
performance run. Boundary/process measurements are not OS hard memory caps.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
GIB = 1 << 30
ARMS = ['off', 'on-lazy', 'on-materialize']
LENGTHS = [1, 19, 935, 978]
CASE_PHASES = ['before_prefill', 'after_hook', 'after_prefill_build', 'after_normal_product_eval',
               'after_common_validation_eval', 'after_raw_artifacts', 'after_drop_observer_refs', 'after_case_release']
CASES = [f'T{t}-{kind}' for t in LENGTHS for kind in ('fresh', 'continuing')]
ENV = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
       'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}
POLICY = {'real_free_min_bytes': 50 * GIB, 'allowed_initial_pressure': [1, 2],
          'pressure_must_never_increase': True, 'swapouts_delta_max': 0,
          'swap_used_growth_max_mib': 16, 'child_physical_max_bytes': 2 * GIB,
          'mlx_boundary_max_bytes': GIB, 'mlx_cache_limit_bytes': 0,
          'power': 'AC Power', 'thermal_codes': [0, 1],
          'terminal_footprint_policy': 'raw-error+exit0+strict-reap+child-self-peak-proof',
          'physical_reading_semantics': 'separate-int-bounds;current=max(top,aux-current);peak=max(top,aux-peak);aux-peak>=aux-current'}
REASON = ('Existing pressure 2 is permitted only for this isolated, real-weight '
          'single-GDN numerical diagnostic with 1GiB MLX boundaries/cache0 and '
          '2GiB observed process current/peak. Never full-model/performance admission; '
          'neither observation is an OS hard memory cap.')


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def write(out, name, value):
    path = Path(out) / name
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def sha(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(8 << 20):
            check(); digest.update(data)
    check()
    return digest.hexdigest()


def stat(path):
    s = Path(path).stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'dev': s.st_dev, 'ino': s.st_ino}


class Budget:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.start = clock()
        self.work_end = self.start + 450
        self.total_end = self.start + 480
        self.arm_end = None

    def check(self):
        if self.clock() >= min(self.work_end, self.arm_end or self.work_end):
            raise TimeoutError('E026-layer work/arm deadline')

    def arm(self):
        self.check()
        self.arm_end = min(self.work_end, self.clock() + 120)
        self.alarm()

    def alarm(self):
        self.check()
        signal.setitimer(signal.ITIMER_REAL, max(.001, min(self.work_end, self.arm_end or self.work_end) - self.clock()))

    def cleanup_alarm(self):
        remaining = self.total_end - self.clock()
        need(remaining > 0, 'total cleanup deadline exhausted')
        signal.setitimer(signal.ITIMER_REAL, remaining)

    def cleanup_check(self):
        if self.clock() >= self.total_end:
            raise TimeoutError('E026-layer cleanup deadline')


def verify_pins(pins, out, name, check, freeze=False):
    verified = {}
    for file, expected in pins.items():
        path = Path(file)
        need(path.is_absolute() and '..' not in path.parts, 'absolute pin path required')
        need(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected), 'invalid pin digest')
        before = stat(path)
        need(sha(path, check) == expected and stat(path) == before, 'pin mismatch/change: ' + file)
        copied = freeze and path.suffix in ('.py', '.json', '.md', '.h', '.cpp', '.txt')
        if copied:
            target = out / 'frozen-source' / file.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open('rb') as src, target.open('wb') as dst:
                while data := src.read(8 << 20):
                    check(); dst.write(data)
            need(sha(target, check) == expected, 'source changed while copying')
        verified[file] = {'sha256': expected, 'stat': before, 'source_copied': copied}
    write(out, name, verified)
    return verified


def prepare(args, check):
    need(args.plan.is_absolute() and sha(args.plan, check) == args.plan_sha256, 'plan SHA mismatch')
    plan = json.loads(args.plan.read_text())
    expected = {'schema', 'arms', 'arm_seconds', 'work_seconds', 'cleanup_seconds', 'total_seconds',
                'execution', 'package_root', 'bundle', 'resource_policy'}
    need(set(plan) == expected and plan['schema'] == 'E026-layer-plan-v2', 'independent layer plan required')
    need(equal({k: plan[k] for k in ('arms', 'arm_seconds', 'work_seconds', 'cleanup_seconds', 'total_seconds')},
               {'arms': ARMS, 'arm_seconds': 120, 'work_seconds': 450, 'cleanup_seconds': 30, 'total_seconds': 480})
         and equal(plan['resource_policy'], POLICY), 'unregistered budgets/policy')
    ex = plan['execution']
    need(set(ex) == {'python', 'screen', 'pins', 'environment'} and equal(ex['environment'], ENV), 'fixed execution environment required')
    need(set(ex['screen']) == {'path', 'sha256'} and Path(ex['screen']['path']) == HERE / 'layer_screen.py', 'adjacent layer screen required')
    python = Path(ex['python'])
    need(python.is_absolute() and os.access(python, os.X_OK), 'absolute executable Python required')
    package = Path(plan['package_root'])
    core = package / 'tutorial_code/miniengine'
    need(package.is_absolute() and core.is_dir(), 'frozen package root missing')
    need(set(plan['bundle']) == {'path', 'sha256'}, 'bundle manifest path/SHA required')
    manifest_path = Path(plan['bundle']['path'])
    need(manifest_path.is_absolute() and sha(manifest_path, check) == plan['bundle']['sha256'], 'bundle manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    need(manifest['bundle']['filename'] == 'weights.bin', 'fixed weight filename required')
    payload = manifest_path.parent / 'weights.bin'
    pins = ex['pins']
    required = {Path(__file__).resolve(), HERE / 'layer_screen.py', HERE / 'layer_weights.py',
                HERE / 'candidate.py', HERE / 'integration.py', HERE / 'layer-preregistration.md',
                HERE / 'source-pins.json', HERE / 'test_layer_owner_cpu.py',
                HERE / 'test_layer_cpu.py', python.resolve(), manifest_path, payload, args.plan}
    required |= {manifest_path.parent / n for n in
                 ('config.json', 'model.safetensors.index.json', 'shard-header.json', 'preparation.json')}
    required |= {BASE / n for n in ('resources_v2.py', 'resources.py', 'resource_base.py', 'suspended_service.py')}
    required |= set(core.rglob('*.py'))
    required.add(package.parent / 'native-source.json')
    # The plan hashes itself separately, not through a recursive self digest.
    required.remove(args.plan)
    need(required <= {Path(p) for p in pins}, 'pins omit source/runtime/bundle files')
    need(pins[str(manifest_path)] == plan['bundle']['sha256'] and pins[str(payload)] == manifest['bundle']['sha256'], 'bundle payload/manifest pins conflict')
    need(pins[str(HERE / 'layer_screen.py')] == ex['screen']['sha256'], 'screen pins conflict')
    for fragment in ('/mlx/core.', '/mlx/lib/libmlx.dylib', '/mlx/lib/mlx.metallib', '/numpy/__init__.py', '/numpy/_core/_multiarray_umath.'):
        need(any(fragment in p for p in pins), 'missing runtime pin: ' + fragment)
    source_evidence = json.loads((HERE / 'source-pins.json').read_text())
    for row in source_evidence['sources'].values():
        path = (HERE / row['copy']).resolve()
        need(path.is_relative_to(HERE) and pins.get(str(path)) == row['sha256'], 'official source evidence omitted')
    write(args.out, 'plan.json', plan)
    verify_pins(pins, args.out, 'verified-pins.json', check, freeze=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('MLX_')}
    return plan, {**env, **ENV}


def parse_footprint(value, pid):
    need(value.get('unit') == 'byte' and type(value.get('bytes per unit')) is int and value['bytes per unit'] == 1, 'footprint byte units required')
    need(value.get('errors') == [] and value.get('warnings') == [], 'footprint errors/warnings')
    processes = value.get('processes')
    need(type(processes) is list and len(processes) == 1 and type(processes[0].get('pid')) is int and processes[0]['pid'] == pid, 'footprint exact child PID required')
    p = processes[0]
    top, current, peak = p.get('footprint'), p['auxiliary']['phys_footprint'], p['auxiliary']['phys_footprint_peak']
    need(all(type(v) is int and 0 <= v <= 2 * GIB for v in (top, current, peak))
         and peak >= current, 'child physical current/peak exceeds 2GiB or is unknown')
    # The actual OS output contained unequal top-level and auxiliary readings.
    # Their difference is unexplained. Preserve and bound all three separately.
    return {'pid': pid, 'current_bytes': max(top, current), 'peak_bytes': max(top, peak),
            'top_bytes': top, 'aux_current_bytes': current, 'aux_peak_bytes': peak}


def footprint(child, out, index):
    raw_path = out / f'footprint-{index:03d}.raw.json'
    command = ['/usr/bin/footprint', '-p', str(child.pid), '--noCategories', '-f', 'bytes', '-j', str(raw_path)]
    began = time.monotonic()
    row = {'command': command, 'pid': child.pid, 'owner_pid': os.getpid(), 'raw_file': raw_path.name}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        row.update(returncode=None, stdout=exc.stdout.decode(errors='replace') if isinstance(exc.stdout, bytes) else exc.stdout or '',
                   stderr=exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else exc.stderr or '', error_type='TimeoutExpired')
    except OSError as exc:
        row.update(returncode=None, stdout='', stderr=str(exc), error_type=type(exc).__name__)
    row.update(elapsed_wall_s=time.monotonic() - began, raw_exists=raw_path.exists())
    row['probe_slow'] = row['elapsed_wall_s'] > 1
    if raw_path.exists():
        row['raw_sha256'] = sha(raw_path)
    try:
        if row['returncode'] != 0 or not raw_path.exists():
            terminal = child.poll()
            row['child_poll_after_probe'] = terminal
            need(terminal == 0, 'physical probe failed while child live or failed')
            row.update(status='terminal-exit0-race-not-measured', physical_pass=None,
                       requires='strict reap plus every-case/final child self current/peak proof')
        else:
            # A successful but malformed/over-budget measurement is never an exit race.
            row['parsed'] = parse_footprint(json.loads(raw_path.read_text()), child.pid)
            row.update(status='measured', physical_pass=True)
        return row
    finally:
        write(out, f'footprint-{index:03d}.json', row)


def guard(samples, owned_pid=None, terminal_exit0=False):
    last, first = samples[-1], samples[0]
    pressures = [r.get('pressure_level_value') for r in samples]
    need(all(type(v) is int and v in (1, 2) for v in pressures) and pressures[-1] <= min(pressures), 'pressure worsened/unknown or improved then rebounded')
    for k in ('vm_stat', 'swap', 'pressure_level'):
        need(last.get(k, {}).get('returncode') == 0, 'resource command failed: ' + k)
    need(last['pressure_level']['text'].strip() == str(pressures[-1]), 'raw pressure mismatch')
    page = re.search(r'page size of (\d+) bytes', last['vm_stat']['text'])
    free = re.search(r'^Pages free:\s*(\d+)\.', last['vm_stat']['text'], re.M)
    swapout = re.search(r'^Swapouts:\s*(\d+)\.', last['vm_stat']['text'], re.M)
    need(page and free and swapout, 'real free/Swapouts fields missing')
    last['real_free_bytes'] = int(page[1]) * int(free[1])
    need(last['real_free_bytes'] >= 50 * GIB, 'real free pages below 50GiB')
    value, initial = last.get('vm_counters', {}).get('Swapouts'), first.get('vm_counters', {}).get('Swapouts')
    need(type(value) is int and type(initial) is int and value == initial == int(swapout[1]), 'new/reset/unknown Swapouts')
    raw_swap = re.search(r'used = ([\d.]+)M', last['swap']['text'])
    used, base = last.get('swap_used_mb'), first.get('swap_used_mb')
    need(raw_swap and type(used) in (int, float) and type(base) in (int, float) and math.isfinite(used)
         and math.isfinite(base) and 0 <= used == float(raw_swap[1]) <= base + 16, 'swap growth >16MiB or raw mismatch')
    power, thermal = last.get('power_metadata', {}), last.get('processinfo_thermal_metadata', {})
    need(power.get('status') == 'known' and power.get('source') == 'AC Power'
         and power.get('raw', {}).get('returncode') == 0 and "AC Power" in power['raw']['stdout'], 'AC power unavailable')
    need(thermal.get('status') == 'known' and type(thermal.get('code')) is int and thermal['code'] in (0, 1)
         and thermal.get('raw', {}).get('returncode') == 0 and thermal['raw']['stdout'] == str(thermal['code']), 'thermal gate failed')
    need(last.get('native_process_inventory_ok') is True and last.get('process_inventory_ok') is True
         and last.get('authorized_suspended_pids') == [74384], 'process inventory/suspension contract unavailable')
    native = last.get('native_inference_processes', [])
    service = [p for p in native if p['pid'] == 74384]
    need(len(service) == 1 and 'T' in service[0]['state'] and service[0].get('authorized_suspended') is True, 'lease-authorized 74384 not observed T')
    for p in native + last.get('known_inference_processes', []):
        need(p['pid'] in (74384, owned_pid), 'foreign inference process')
        if p['pid'] == owned_pid:
            need(p.get('ppid') == os.getpid() and p.get('owned') is True, 'child not directly owned')
    if owned_pid is not None and not terminal_exit0:
        matching = [p for p in native if p['pid'] == owned_pid]
        need(len(matching) == 1, 'live child missing from inventory')
        exact = last.get('exact_owned_pid_metadata', {})
        need(exact.get('status') == 'known' and exact.get('pid') == owned_pid and exact.get('ppid') == os.getpid(), 'exact live child identity missing')


def reap(child, out):
    notes, failure = [], None
    try:
        if child.poll() is None:
            try:
                child.terminate(); child.wait(timeout=3)
            except Exception as exc:
                notes.append(type(exc).__name__)
            if child.poll() is None:
                child.kill()
        child.wait(timeout=3)
        raw = subprocess.run(['/bin/ps', '-p', str(child.pid), '-o', 'pid='], capture_output=True, text=True, timeout=2)
        row = {'pid': child.pid, 'returncode': child.returncode, 'waited_reaped': True,
               'strict_absence': raw.returncode == 1 and not raw.stdout.strip(), 'cleanup_notes': notes,
               'absence_probe': {'returncode': raw.returncode, 'stdout': raw.stdout, 'stderr': raw.stderr}}
        need(row['strict_absence'], 'strict child PID absence not established')
    except BaseException as exc:
        failure = exc
        row = {'pid': child.pid, 'returncode': child.returncode, 'waited_reaped': False,
               'strict_absence': False, 'cleanup_notes': notes, 'error_type': type(exc).__name__}
    write(out, 'release.json', row)
    if failure is not None:
        raise failure


def nofile(value):
    return {k: v for k, v in value.items() if k != 'file'}


def raw_record(out, row, dtype, shape, check=lambda: None):
    """Validate full-size actual raw bytes, not just a child-provided digest."""
    need(set(row) == {'file', 'sha256', 'bytes', 'dtype', 'shape', 'encoding', 'finite'}, 'raw descriptor fields')
    width, encoding, finite = {'bfloat16': (2, 'uint16-le', True), 'float32': (4, 'uint32-le', True),
                              'uint32': (4, 'uint32-le', None), 'uint8': (1, 'uint8', None)}[dtype]
    need(row['dtype'] == dtype and equal(row['shape'], shape)
         and all(type(v) is int and v > 0 for v in shape), 'raw shape/dtype contract')
    need(row['encoding'] == encoding and row['finite'] is finite and type(row['bytes']) is int
         and row['bytes'] == math.prod(shape) * width, 'raw byte/encoding/finite contract')
    relative = Path(row['file'])
    need(not relative.is_absolute() and '..' not in relative.parts, 'raw path must be contained')
    path = (out / relative).resolve()
    need(path.is_relative_to(out.resolve()) and path.stat().st_size == row['bytes'], 'raw file containment/size')
    before = stat(path)
    need(sha(path, check) == row['sha256'] and stat(path) == before, 'raw SHA/change: ' + str(relative))
    return path


def jsonl(path):
    raw = Path(path).read_text()
    need(raw.endswith('\n') and all(line for line in raw.splitlines()), 'partial/empty JSONL line')
    return [json.loads(line) for line in raw.splitlines()]


def validate_child(out, arm, pid, plan, check=lambda: None):
    """Validate a complete local arm; cross-arm bytes are a separate hard gate."""
    result = json.loads((out / 'complete.json').read_text())
    need(result.get('status') == 'PASS' and result.get('device_run') is True
         and result.get('arm') == arm and type(result.get('pid')) is int and result['pid'] == pid
         and result.get('ppid') == os.getpid(), 'incomplete/wrong child identity')
    need(result.get('source_pins_rechecked') is True and result.get('bundle_rechecked') is True
         and result.get('performance_qualified') is False
         and result.get('cross_arm_equality') == 'PENDING_INDEPENDENT_RAW_AUDIT', 'child recheck/scope missing')
    elapsed = result.get('elapsed_wall_s')
    need(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 < elapsed <= 120, 'child elapsed budget')
    contract = result['contract']
    required_contract = dict(schema='E026-real-weight-layer-v2', arm=arm, layer_index=0,
        lengths=LENGTHS, cache_kinds=['fresh', 'continuing'], cases=8, case_ids=CASES,
        cache_limit_bytes=0, mlx_observation_budget_bytes=GIB, physical_observation_budget_bytes=2*GIB,
        child_alarm_seconds=120, phase_count=68, case_phases=CASE_PHASES,
        continuing_offset=7, continuing_prefix_forward=False)
    need(equal({k: contract.get(k) for k in required_contract}, required_contract), 'child contract differs')
    pins = plan['execution']['pins']
    package = Path(plan['package_root'])
    manifest_path = package.parent / 'native-source.json'
    manifest = json.loads(manifest_path.read_text())
    required_sources = {str((package / 'tutorial_code/miniengine' / p).resolve()) for p in manifest['files']}
    required_sources |= {str(manifest_path), *(str(HERE / n) for n in ('layer_weights.py', 'integration.py', 'candidate.py', 'layer_screen.py'))}
    need(set(result['sources']) == required_sources and all(pins.get(p) == h for p, h in result['sources'].items()), 'loaded sources differ from execution pins')
    for key in ('mlx_file', 'numpy_file'):
        path = str(Path(result['runtime'][key]).resolve())
        need(path in pins and sha(path, check) == pins[path], 'runtime outside frozen pins')
    bundle = json.loads(Path(plan['bundle']['path']).read_text())
    need(result['bundle_manifest_sha256'] == plan['bundle']['sha256']
         and result['bundle_sha256'] == bundle['bundle']['sha256'], 'child bundle identity')
    expected_parameters = {}
    for row in bundle['tensors']:
        name = row['parameter']; dtype = {'BF16': 'bfloat16', 'U32': 'uint32', 'U8': 'uint8'}[row['dtype']]
        shape = row['shape']
        if name in ('A_log', 'dt_bias'): dtype = 'float32'
        if name == 'conv1d_weight': shape = [shape[0], math.prod(shape[1:])]
        expected_parameters[name] = (dtype, shape)
    need(len(expected_parameters) == 14, 'exact 14 parameters required')
    for key in ('parameters_before', 'parameters_after'):
        need(set(result[key]) == set(expected_parameters), 'parameter set differs')
        for name, (dtype, shape) in expected_parameters.items(): raw_record(out, result[key][name], dtype, shape, check)
    baseline = {k: nofile(v) for k, v in result['parameters_before'].items()}
    need(equal(baseline, {k: nofile(v) for k, v in result['parameters_after'].items()}), 'parameters changed')
    rows = result['cases']
    need(type(rows) is list and [r.get('case') for r in rows] == CASES, 'incomplete/duplicate/reordered case matrix')
    witnesses = jsonl(out / 'witness.jsonl')
    need(len(witnesses) == 8, 'hook witness coverage')
    for index, row in enumerate(rows):
        check(); length, kind = LENGTHS[index // 2], ('fresh', 'continuing')[index % 2]
        case = CASES[index]; offset = 7 if kind == 'continuing' else 0
        need(equal(row, json.loads((out / 'cases' / case / 'case.json').read_text())), 'case file differs from complete')
        need(row['status'] == 'PASS' and equal([row['length'], row['kind'], row['layer_index']], [length, kind, 0]), 'case contract')
        need(equal([row['offset_before'], row['offset_after']], [offset, offset + length]), 'cache offset')
        need(set(row['input']) == {'x', 'conv_seed', 'S_seed'} and set(row['input_after']) == set(row['input']), 'input evidence incomplete')
        for name, dtype, shape in (('x', 'bfloat16', [length, 5120]), ('conv_seed', 'bfloat16', [3, 10240]), ('S_seed', 'float32', [48, 128, 128])):
            if kind == 'fresh' and name != 'x':
                need(row['input'][name] is None and row['input_after'][name] is None, 'fresh seed must be absent')
            else:
                raw_record(out, row['input'][name], dtype, shape, check)
                need(equal(nofile(row['input'][name]), row['input_after'][name]), 'input seed changed')
        shapes = {'output': ('bfloat16', [length, 5120]), 'conv_y': ('bfloat16', [length, 10240]),
                  'conv_state': ('bfloat16', [3, 10240]), 'S': ('float32', [48, 128, 128]),
                  'original_conv_tail': ('bfloat16', [3, 10240])}
        need(set(row['arrays']) == set(shapes), 'product evidence incomplete')
        for name, (dtype, shape) in shapes.items(): raw_record(out, row['arrays'][name], dtype, shape, check)
        need(equal(nofile(row['arrays']['conv_state']), nofile(row['arrays']['original_conv_tail'])), 'owned tail differs from original tail')
        need(equal(row['parameters_before'], baseline) and equal(row['parameters_after'], baseline), 'per-case parameters changed')
        hook = row['hook_witness']
        required_hook = dict(sequence=1, layer=0, route='single-layer-eager', input_shape=[length, 10240],
            tail_shape=[3, 10240], input_dtype='mlx.core.bfloat16', tail_dtype='mlx.core.bfloat16',
            offset=offset, state_present=kind == 'continuing', eligible=True, enabled=arm != 'off',
            materialize=arm == 'on-materialize', replaced=arm != 'off', y_identity_preserved=True)
        need(equal(hook, required_hook) and equal(witnesses[index], {'case': case, **required_hook}), 'actual eligible hook route differs')
        need(equal(row['profiler'], dict(count=1, restored=True, original_source_sha256=pins[str(package / 'tutorial_code/miniengine/model/gdn.py')], observer='exact-original-code-return')), 'original convolution observation missing')
    phases = jsonl(out / 'phases.jsonl')
    expected = [('global', 'after_weights_load'), ('global', 'after_parameters_before')]
    expected += [(case, phase) for case in CASES for phase in CASE_PHASES]
    expected += [('global', 'after_parameters_after'), ('global', 'after_layer_release')]
    receipts = result['physical_receipts']
    need(len(phases) == len(receipts) == 68, 'incomplete physical/MLX phase coverage')
    previous_time, previous_peak, previous_physical_peak = -1, -1, -1
    for index, (phase, pair, receipt) in enumerate(zip(phases, expected, receipts)):
        check()
        need(equal([phase['index'], phase['case'], phase['phase'], phase['pid'], phase['ppid']], [index, *pair, pid, os.getpid()])
             and phase.get('status', 'PASS') == 'PASS' and phase['groups_are_evaluated'] is True, 'phase identity/order/failure')
        current_time = phase['monotonic_s']
        need(type(current_time) in (int, float) and math.isfinite(current_time) and current_time >= previous_time
             and type(phase['elapsed_s']) in (int, float) and 0 <= phase['elapsed_s'] <= elapsed, 'phase time')
        previous_time = current_time
        for key in ('active_bytes', 'peak_bytes', 'cache_bytes', 'unique_buffer_bytes'):
            need(type(phase[key]) is int and phase[key] >= 0, 'invalid MLX counter')
        need(phase['active_bytes'] + phase['cache_bytes'] <= GIB and previous_peak <= phase['peak_bytes'] <= GIB
             and phase['unique_buffer_bytes'] <= phase['active_bytes'], 'MLX boundary/peak budget')
        previous_peak = phase['peak_bytes']
        need(type(phase['group_buffer_bytes']) is dict and all(type(v) is int and 0 <= v <= phase['unique_buffer_bytes'] for v in phase['group_buffer_bytes'].values()), 'invalid deduplicated group buffers')
        raw = out / f'physical-{index:03d}.raw.json'
        command = json.loads((out / f'physical-{index:03d}.command.json').read_text())
        need(command['command'] == ['/usr/bin/footprint', '-p', str(pid), '--noCategories', '-f', 'bytes', '-j', str(raw)]
             and command['returncode'] == 0 and type(command['stdout']) is str and type(command['stderr']) is str, 'self footprint command failed/differs')
        required_receipt = dict(path=raw.name, sha256=sha(raw, check), pid=pid, case=pair[0], phase=pair[1])
        need(equal(receipt, required_receipt) and equal(phase['physical_receipt'], receipt), 'physical receipt mismatch')
        raw_value = json.loads(raw.read_text())
        parsed = parse_footprint(raw_value, pid)
        auxiliary_peak = raw_value['processes'][0]['auxiliary']['phys_footprint_peak']
        need(equal([phase['physical_current_bytes'], phase['physical_peak_bytes']], [parsed['current_bytes'], parsed['peak_bytes']])
             and equal([phase['physical_top_bytes'], phase['physical_aux_current_bytes'], phase['physical_aux_peak_bytes']],
                       [parsed['top_bytes'], parsed['aux_current_bytes'], parsed['aux_peak_bytes']])
             and auxiliary_peak >= previous_physical_peak, 'physical values/auxiliary peak history differ')
        previous_physical_peak = auxiliary_peak
    release = result['release']
    need(release['layer_weakref_dead'] is True and type(result['initial_active_bytes']) is int
         and equal(release['initial_active_bytes'], result['initial_active_bytes'])
         and equal(release['final_active_bytes'], phases[-1]['active_bytes'])
         and equal(release['active_delta_bytes'], release['final_active_bytes'] - release['initial_active_bytes'])
         and equal(release['known_persistent_buffer_bytes'], phases[-1]['unique_buffer_bytes']), 'layer release evidence inconsistent')
    return {'arm': arm, 'pid': pid, 'status': 'PASS', 'complete_sha256': sha(out / 'complete.json', check),
            'cases': 8, 'physical_phases': 68, 'directory': str(out), 'cross_arm': 'not-yet-compared'}


def compare_arms(reference, candidate, arm, check=lambda: None):
    """Read every input/parameter/product byte before admitting another arm."""
    a = json.loads((reference / 'complete.json').read_text())
    b = json.loads((candidate / 'complete.json').read_text())
    need(a['arm'] == 'off' and b['arm'] == arm and arm in ARMS[1:], 'comparison arm identity')
    pairs = []
    for key in ('parameters_before', 'parameters_after'):
        need(set(a[key]) == set(b[key]), 'cross-arm parameter names')
        pairs += [(key + '/' + name, a[key][name], b[key][name]) for name in a[key]]
    need([r['case'] for r in a['cases']] == [r['case'] for r in b['cases']] == CASES, 'cross-arm case matrix')
    for left, right in zip(a['cases'], b['cases']):
        need(equal([left['offset_before'], left['offset_after']], [right['offset_before'], right['offset_after']]), 'cross-arm offsets')
        for key in ('input', 'arrays'):
            need(set(left[key]) == set(right[key]), 'cross-arm arrays/input names')
            for name in left[key]:
                x, y = left[key][name], right[key][name]
                if x is None: need(y is None, 'cross-arm absent seed'); continue
                pairs.append((left['case'] + '/' + key + '/' + name, x, y))
    compared = 0
    for label, left, right in pairs:
        check()
        try:
            need(equal({k: v for k, v in nofile(left).items() if k != 'sha256'},
                       {k: v for k, v in nofile(right).items() if k != 'sha256'}),
                 'cross-arm original-dtype descriptor differs: ' + label)
            p = raw_record(reference, left, left['dtype'], left['shape'], check)
            q = raw_record(candidate, right, right['dtype'], right['shape'], check)
            offset = 0
            with p.open('rb') as lhs, q.open('rb') as rhs:
                while x := lhs.read(8 << 20):
                    check(); y = rhs.read(len(x))
                    if x != y:
                        at = next((i for i, pair in enumerate(zip(x, y)) if pair[0] != pair[1]), min(len(x), len(y)))
                        write(candidate, 'first-difference.json', {'arm': arm, 'field': label, 'byte_offset': offset + at,
                              'reference': str(p), 'candidate': str(q), 'reason': 'original dtype bytes differ'})
                        raise RuntimeError('cross-arm byte difference: ' + label)
                    offset += len(x)
                need(not rhs.read(1), 'cross-arm extra candidate bytes')
            need(left['sha256'] == right['sha256'], 'equal bytes have conflicting raw digests')
            compared += left['bytes']
        except BaseException:
            if not (candidate / 'first-difference.json').exists():
                write(candidate, 'first-difference.json', {'arm': arm, 'field': label, 'reference': left,
                      'candidate': right, 'reason': traceback.format_exc()})
            raise
    result = {'status': 'PASS', 'reference_arm': 'off', 'arm': arm, 'raw_arrays': len(pairs),
              'bytes_compared': compared, 'method': 'full original dtype metadata+SHA+streamed byte equality; no tolerance'}
    write(candidate, 'cross-arm-comparison.json', result)
    return result


def run(args, resources=None):
    args.out.mkdir(parents=True, exist_ok=False)
    budget, child, child_out, plan = Budget(), None, None, None
    samples, errors, arms, physical = [], [], [], []
    def stop(*unused):
        raise TimeoutError('E026-layer absolute owner/arm/cleanup deadline or cancellation')
    def failed(phase):
        row = {'phase': phase, 'traceback': traceback.format_exc()}; errors.append(row)
        if len(errors) == 1: write(args.out, 'first-error.json', row)
        write(args.out, 'errors.json', errors)
    def probe(label, current=None):
        pid = current.pid if current is not None else None
        row = resources.snapshot(label, pid, suspended_pids=(74384,)); samples.append(row)
        index = len(samples) - 1
        write(args.out, f'resource-{index:03d}.json', row)
        try:
            terminal = False
            if current is not None:
                f = footprint(current, args.out, index); physical.append(f)
                row['physical_receipt'] = f'footprint-{index:03d}.json'
                terminal = current.poll() == 0
                row['terminal_exit0_observed'] = terminal
            guard(samples, pid, terminal)
        finally:
            write(args.out, f'resource-{index:03d}.json', row)
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'outer alarm already active')
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    write(args.out, 'attempt.json', {'owner_pid': os.getpid(), 'plan_sha256': args.plan_sha256,
          'arm_seconds': 120, 'work_seconds': 450, 'cleanup_seconds': 30, 'total_seconds': 480,
          'pressure_policy_reason': REASON, 'external_service_restoration': 'outer suspended_service sole owner'})
    try:
        budget.alarm()
        plan, env = prepare(args, budget.check)
        if resources is None:
            sys.path.insert(0, str(BASE)); resources = importlib.import_module('resources_v2')
            need(Path(resources.__file__).resolve() == BASE / 'resources_v2.py', 'wrong resource source')
        probe('preflight')
        for arm in ARMS:
            budget.check(); probe(arm + '/before')
            child_out = args.out / arm; child_out.mkdir()
            command = [plan['execution']['python'], '-I', '-B', str(HERE / 'layer_screen.py'), '--run', '--arm', arm,
                       '--bundle-manifest', plan['bundle']['path'], '--bundle-sha256', plan['bundle']['sha256'],
                       '--package-root', plan['package_root'], '--out', str(child_out / 'screen')]
            budget.arm()
            with (child_out / 'child.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                write(child_out, 'child.json', {'pid': child.pid, 'ppid': os.getpid(), 'command': command, 'arm': arm,
                      'spawned_monotonic': time.monotonic(), 'arm_deadline_monotonic': budget.arm_end})
                while child.poll() is None:
                    budget.check(); probe(arm + '/running', child); budget.check()
                    try: child.wait(timeout=min(1, max(.001, budget.arm_end - time.monotonic())))
                    except subprocess.TimeoutExpired: pass
                child.wait(timeout=0)
            need(child.returncode == 0, arm + ' child did not exit zero')
            result = validate_child(child_out / 'screen', arm, child.pid, plan, check=budget.check)
            write(child_out, 'validated-child.json', result)
            if arm != 'off':
                result['cross_arm'] = compare_arms(args.out / 'off/screen', child_out / 'screen', arm, budget.check)
            budget.check()
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
            budget.cleanup_alarm(); reap(child, child_out); child = None
            arms.append(result)
            signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
            budget.arm_end = None; budget.alarm(); probe(arm + '/after')
        verify_pins(plan['execution']['pins'], args.out, 'pins-after.json', budget.check)
        need(sha(args.plan, budget.check) == args.plan_sha256, 'plan changed during run')
    except BaseException:
        failed('execution')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            budget.cleanup_alarm()
            if child is not None:
                try: reap(child, child_out)
                except BaseException: failed('cleanup')
            if resources is not None:
                try: probe('postflight')
                except BaseException: failed('postflight')
            if plan is not None and not (args.out / 'pins-after.json').exists():
                try:
                    verify_pins(plan['execution']['pins'], args.out, 'pins-after.json', budget.cleanup_check)
                    need(sha(args.plan, budget.cleanup_check) == args.plan_sha256, 'plan changed during failed run')
                except BaseException: failed('post-pins')
        except BaseException:
            failed('cleanup-deadline')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in previous.items(): signal.signal(sig, handler)
        write(args.out, 'resources.json', samples)
        passed = not errors and len(arms) == 3 and time.monotonic() <= budget.total_end
        write(args.out, 'complete.json', {'schema': 'E026-layer-owner-result-v2', 'status': 'PASS' if passed else 'FAIL',
              'owner_pid': os.getpid(), 'arms': arms, 'errors': errors, 'samples': len(samples),
              'physical_samples': len(physical), 'elapsed_wall_s': time.monotonic() - budget.start,
              'pins_rechecked': (args.out / 'pins-after.json').exists(), 'pressure_policy_reason': REASON,
              'scope': 'single real-weight GDN numerical/state/backing diagnostic; no whole-model/performance result',
              'external_service_restoration': 'outer suspended_service sole owner'})
    return int(not passed)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--suspended-pid', type=int, required=True)
    a = p.parse_args(argv)
    if a.suspended_pid != 74384: p.error('only lease-authorized PID 74384 allowed')
    a.plan, a.out = a.plan.absolute(), a.out.absolute()
    return run(a)


if __name__ == '__main__':
    raise SystemExit(main())
