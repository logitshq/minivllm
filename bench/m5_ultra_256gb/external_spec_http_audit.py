"""Independent CPU audit/archive of E020, including partial and invalid attempts.

Never imports E020 measurement code, MLX, or either server. Run only after the
GPU queue is released: hashing frozen sources and tar verification are CPU/I/O
work. Weight/pack payloads and installed environments are not reread or archived.
Their expected hashes and the runner's before/after verification receipts remain
explicitly recorded evidence, not independently verified payload bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.artifacts/m5-ultra-256gb'
ORDERS = [
    ['native-none', 'omlx-none', 'omlx-mtp', 'lithos-none', 'lithos-dspark'],
    ['lithos-dspark', 'lithos-none', 'native-none', 'omlx-mtp', 'omlx-none'],
    ['omlx-none', 'omlx-mtp', 'lithos-none', 'lithos-dspark', 'native-none'],
]
MATRIX = [(pair, arm) for pair, arms in enumerate(ORDERS, 1) for arm in arms]
COUNTERS = ('Swapins', 'Swapouts', 'Pageouts', 'Compressions', 'Decompressions')
SOURCE_SUFFIXES = {'.py', '.json', '.metal', '.h', '.hpp', '.cpp', '.mm'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def equal(a, b):
    if number(a) and number(b):
        return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)
    return a == b


class Audit:
    def __init__(self):
        self.integrity_errors = []
        self.notes = []
        self.files = {}

    def check(self, value, where, detail):
        if not value:
            self.integrity_errors.append({'where': str(where), 'detail': detail})
        return bool(value)

    def load(self, path, required=False):
        path = Path(path)
        if not path.is_file():
            if required:
                self.check(False, path, 'required evidence missing')
            return None
        try:
            return read(path)
        except Exception as error:
            self.check(False, path, f'unreadable JSON: {type(error).__name__}: {error}')
            return None

    def collect(self, path, name):
        path = Path(path)
        if path.is_file() and not path.is_symlink():
            previous = self.files.get(name)
            if previous and previous != path:
                self.check(sha(previous) == sha(path), name, 'duplicate archive member differs')
            self.files[name] = path

    def collect_tree(self, root, prefix):
        root = Path(root)
        if not root.is_dir():
            return
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                self.notes.append({'scope': 'symlink not followed or archived', 'path': str(path)})
                continue
            # No weights, cached binaries, databases, or interpreter directories.
            # A pre-recorded frozen Python source under a venv-shaped path is
            # evidence, not an installed/runnable environment.
            if path.is_file() and path.suffix in {'.json', '.jsonl', '.log', '.txt', '.md', '.py', '.metal', '.h', '.hpp', '.cpp', '.mm', '.jinja'}:
                self.collect(path, prefix + '/' + str(path.relative_to(root)))


def request_audit(audit, path, row, plan, tokenizer, pair, arm, phase, index):
    """Reconstruct from original SSE data/timestamps, not stored scalar metrics."""
    bad = []
    def need(condition, detail):
        if not condition:
            bad.append(detail)
    engine = arm.split('-')[0]
    expected = plan['oracle_ids'][engine]
    body = row.get('body') or {}
    need(row.get('pair') == pair and row.get('arm') == arm and row.get('phase') == phase and row.get('index') == index,
         'request identity/order differs')
    need(row.get('method') == 'POST' and row.get('http_status') == 200, 'not a successful POST')
    need(body.get('stream') is True and body.get('stream_options', {}).get('include_usage') is True, 'SSE usage contract differs')
    need(body.get('temperature') == 0 and 'ignore_eos' not in body and 'stop' not in body, 'greedy/normal EOS request contract differs')
    if engine == 'lithos':
        need(row.get('path') == '/v1/chat/completions', 'wrong Lithos API path')
        need(body.get('messages') == [{'role': 'user', 'content': plan['user_text']}], 'different chat prompt')
        need(body.get('top_p') == 1 and body.get('max_completion_tokens') == 32, 'different Lithos sampling/budget')
    else:
        need(row.get('path') == '/v1/completions', 'wrong completion API path')
        need(body.get('max_tokens') == 32 and body.get('repetition_penalty') == 1, 'different sampling/budget')
        prompt = body.get('prompt', '')
        need(prompt == plan['prompt_text'] and tokenizer.encode(prompt, add_special_tokens=False).ids == plan['workload']['prompt_ids'],
             'prompt text/19 IDs differ')
    start = row.get('start_perf_counter_s')
    cleanup = row.get('cleanup_complete_perf_counter_s')
    need(number(start) and number(cleanup) and cleanup >= start, 'invalid start/cleanup timestamp')
    events = row.get('sse_events') or []
    texts, usage, finish, live_ids, ends = [], None, None, None, []
    stamps = []
    for ordinal, event in enumerate(events):
        stamp = event.get('client_perf_counter_s')
        need(number(stamp), 'nonfinite SSE timestamp')
        if number(stamp):
            stamps.append(stamp)
        data = event.get('data')
        if data == '[DONE]':
            ends.append((ordinal, stamp))
            continue
        try:
            payload = json.loads(data)
            need(isinstance(payload, dict) and 'error' not in payload, 'SSE error/nonobject payload')
            if not isinstance(payload, dict):
                continue
            if payload.get('usage'):
                usage = payload['usage']
            if 'benchmark_output_ids' in payload:
                live_ids = payload['benchmark_output_ids']
            choices = payload.get('choices', [])
            need(len(choices) <= 1, 'multi-choice response')
            for choice in choices:
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
                piece = choice.get('text', (choice.get('delta') or {}).get('content', ''))
                if piece:
                    need(isinstance(piece, str), 'nonstring text')
                    if isinstance(piece, str):
                        texts.append({'client_perf_counter_s': stamp, 'text': piece})
        except Exception as error:
            bad.append(f'unparseable raw SSE event {ordinal}: {type(error).__name__}')
    need(len(stamps) == len(events) and all(a <= b for a, b in zip(stamps, stamps[1:])), 'SSE clock moved backwards/missing')
    if number(start) and number(cleanup):
        need(all(start <= t <= cleanup for t in stamps), 'SSE outside measured request lifecycle')
    need(len(ends) == 1 and ends[0][0] == len(events) - 1, 'missing/repeated/nonfinal DONE')
    end = ends[0][1] if len(ends) == 1 and number(ends[0][1]) else None
    first = texts[0]['client_perf_counter_s'] if texts else None
    need(bool(texts), 'no visible text')
    text = ''.join(p['text'] for p in texts)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    first_text = texts[0]['text'] if texts else ''
    first_ids = tokenizer.encode(first_text, add_special_tokens=False).ids
    prefix = bool(first_text) and ids[:len(first_ids)] == first_ids and text.startswith(first_text)
    need(isinstance(usage, dict), 'missing/invalid usage object')
    usage = usage if isinstance(usage, dict) else {}
    output = usage.get('completion_tokens')
    cache = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    need(usage.get('prompt_tokens') == 19 and output == 32 and len(ids) == 32, 'actual input/output counts differ from 19->32')
    need(ids == expected, 'visible canonical IDs differ from engine-specific qualified raw IDs')
    need(finish == 'length', 'normal EOS ended before the qualified length / wrong finish')
    need(cache is None or cache == 0, 'reported prefix cache hit')
    need(prefix, 'first visible chunk is not canonical prefix')
    need(not row.get('error') and not row.get('finalization_error'), 'recorded transport/client error')
    if engine == 'native':
        need(live_ids == expected and cache == 0, 'native live raw IDs/cache receipt differs')
    wall = end - start if number(end) and number(start) else None
    ttft = first - start if number(first) and number(start) else None
    after_first = end - first if number(end) and number(first) else None
    need(wall is None or 0 < wall <= 300, 'E2E exceeds absolute deadline or is nonpositive')
    need(ttft is None or ttft > 0, 'nonpositive TTFT')
    count_ok = type(output) is int and output >= 0
    throughput = output / wall if count_ok and wall and wall > 0 else None
    remaining = output - len(first_ids) if count_ok else None
    x = remaining / after_first if prefix and len(ids) == output and remaining and remaining > 0 and after_first and after_first > 0 else None
    derived = {'text': text, 'text_events': texts, 'usage': usage or None, 'finish_reason': finish,
        'visible_roundtrip_ids': ids, 'output_ids_when_available': live_ids,
        'first_text_perf_counter_s': first, 'done_perf_counter_s': end, 'done': len(ends) == 1,
        'actual_visible_canonical_count': len(ids), 'actual_usage_output_count': output,
        'cached_prompt_tokens_when_available': cache, 'cache_usage_field_present': cache is not None,
        'first_chunk_text': first_text, 'first_chunk_reconstructed_tokens': len(first_ids),
        'first_chunk_prefix_verified': prefix, 'full_visible_roundtrip_count_matches_usage': len(ids) == output,
        'client_total_s': wall, 'client_ttft_text_s': ttft, 'client_post_first_chunk_s': after_first,
        'output_tok_s': throughput, 'aggregate_output_tok_s': throughput,
        'actual_request_count': 1, 'group_window_s': wall, 'total_completion_tokens': output,
        'client_estimated_interactivity_tok_s': x,
        'stream_cleanup_after_done_s': cleanup - end if number(cleanup) and number(end) else None}
    for key, value in derived.items():
        if key in row:
            audit.check(equal(row[key], value), path, 'stored field differs from raw SSE reconstruction: ' + key)
        elif not row.get('finalization_error'):
            audit.check(False, path, 'expected finalized field missing: ' + key)
    if 'contract_failures' in row:
        audit.check(bool(bad) == bool(row['contract_failures']), path, 'independent contract status disagrees with runner')
    return {'pair': pair, 'arm': arm, 'phase': phase, 'index': index, 'valid': not bad,
        'failures': bad, 'start_s': start, 'done_s': end, 'cleanup_s': cleanup,
        'E2E_s': wall, 'TTFT_s': ttft, 'aggregate_output_tok_s': throughput,
        'client_estimated_interactivity_tok_s': x, 'actual_usage_count': output,
        'actual_visible_count': len(ids), 'visible_ids': ids, 'live_raw_ids_when_available': live_ids,
        'cache_usage_available': cache is not None, 'cache_usage_value': cache,
        'raw_sse_events': len(events), 'scope': 'external visible text is not an independent live raw-ID capture'}


def route_audit(arm, text, plan):
    """Reparse original stock logs independently, including negative route evidence."""
    bad, result = [], {}
    def need(condition, message):
        if not condition:
            bad.append(message)
    engine = arm.split('-')[0]
    if engine == 'native':
        return {'valid': True, 'failures': [], 'scope': 'loader identity/raw IDs checked separately; no drafter'}
    if engine == 'omlx':
        start = []
        stats = []
        for line in text.splitlines():
            m = re.search(r'vlm_mtp decode started: request=(\S+) uid=(-?\d+) block_size=(\S+)', line)
            if m:
                start.append({'request': m[1], 'uid': int(m[2]), 'block_size': m[3]})
            m = re.search(r'vlm_mtp stats: request=(\S+) finish=(\S+) rounds=(\d+) accepted=(\d+)/(\d+) '
                          r'\(([\d.]+)%\) tokens_per_round=([\d.]+) emitted=(\d+) block_size=(\d+)', line)
            if m:
                stats.append({'request': m[1], 'finish': m[2], 'rounds': int(m[3]), 'accepted': int(m[4]),
                    'nominal_capacity': int(m[5]), 'percent': float(m[6]), 'tokens_per_round': float(m[7]),
                    'emitted': int(m[8]), 'block_size': int(m[9])})
        enabled = [line for line in text.splitlines() if 'VLM MTP enabled for' in line]
        warnings = [line for line in text.splitlines() if re.search(r'vlm.?mtp|VLM MTP', line, re.I)
                    and re.search(r'skipped|ignored|falling back|failed|fallback', line, re.I)]
        if arm == 'omlx-none':
            need(not start and not stats and not enabled, 'none arm has MTP route')
        else:
            need(any(f'VLM MTP enabled for Qwen3.8-27B-nvfp4, drafter={plan["drafters"]["omlx"]}' in line for line in enabled),
                 'wrong/missing MTP attachment')
            need(len(start) == len(stats) == 6 and len({s['request'] for s in start}) == 6, 'not six distinct MTP requests')
            need([s['request'] for s in start] == [s['request'] for s in stats], 'MTP start/stats order differs')
            need(not warnings, 'MTP fallback warning present')
            for s in stats:
                need(s['finish'] == 'length' and s['emitted'] == 32 and s['block_size'] == 3 and s['rounds'] > 0,
                     'MTP finish/width/count differs')
                need(s['nominal_capacity'] == 2 * s['rounds'] and 0 < s['accepted'] <= s['nominal_capacity'],
                     'MTP acceptance/capacity log invalid')
        result.update(starts=start, stats=stats, route_warnings=warnings,
            acceptance_scope='stock accept_lens summary; denominator is nominal capacity, not exact proposed count')
    else:
        import ast
        generations = []
        for line in text.splitlines():
            if 'Generation: ' in line:
                try:
                    generations.append(ast.literal_eval(line.split('Generation: ', 1)[1]))
                except Exception:
                    bad.append('unparseable Generation log')
        lines = [line for line in text.splitlines() if 'Serving backend=' in line]
        width = 8 if arm == 'lithos-dspark' else 1
        draft = plan['drafters']['lithos'] if width == 8 else None
        target = plan['workload']['model_dir']
        need(len(lines) == 1 and f'backend=m5_ultra_80c, target={target}, draft={draft}, verify={width}, recipe contexts=[]' in lines[0],
             'Lithos backend/target/draft/verify/recipe differs')
        need(len(generations) == 6, 'not exactly six Generation logs')
        need(all(g.get('verify_tokens') == width and g.get('cached_prompt_tokens') == 0 for g in generations),
             'Lithos cache hit/configured width mismatch')
        need(all(g.get('setup_ms') == 0 for g in generations[3:]), 'formal program setup is nonzero')
        for role, pack in plan['packs'][arm].items():
            if pack:
                need(f'Using existing {role} pack: {pack["path"]}' in text, 'exact qualified pack reuse log missing: ' + role)
        result.update(generation=generations, serving=lines,
            acceptance_scope='verify_tokens is configuration, steps diagnostic; no new actual acceptance/raw-ID trace')
    return {'valid': not bad, 'failures': bad, **result}


def parse_listener(record):
    return sorted({int(line[1:]) for line in record.get('stdout', '').splitlines() if re.fullmatch(r'p\d+', line)})


def parse_ps(record):
    fields = record.get('stdout', '').strip().split()
    return (int(fields[0]), int(fields[1])) if len(fields) >= 2 and all(s.isdigit() for s in fields[:2]) else None


def lifecycle_audit(audit, folder, controller, lease, resume):
    owned = audit.load(folder / 'owned-pid.json')
    ready = audit.load(folder / 'ready.json')
    release = audit.load(folder / 'release.json')
    result = {'valid': False, 'pid': owned.get('pid') if owned else None, 'issues': []}
    def need(condition, message):
        if not condition:
            result['issues'].append(message)
    if not owned:
        need(False, 'no server PID receipt')
        return result
    pid = owned.get('pid')
    need(owned.get('controller_pid') == controller and owned.get('direct_child') is True, 'owner declares wrong controller')
    need(parse_ps(owned.get('ps_at_launch', {})) == (pid, controller), 'launch PS does not prove direct child')
    if ready:
        listener = ready.get('listeners', {})
        need(parse_listener(listener) == listener.get('pids') == [pid], 'ready listener does not belong only to direct child')
        need(listener.get('returncode') == 0 and not listener.get('stderr'), 'ready listener inventory failed')
        need(parse_ps(ready.get('ps', {})) == (pid, controller), 'ready PS does not prove direct child')
    else:
        need(False, 'server never recorded ready')
    if release:
        errors_before_release = len(result['issues'])
        absence = release.get('absence', {})
        listeners = release.get('listeners', {})
        need(release.get('pid') == pid and release.get('parent_pid') == controller, 'release owner differs')
        need(release.get('waited_reaped') is True and type(release.get('returncode')) is int, 'server not waited/reaped')
        need(absence.get('returncode') == 1 and not absence.get('stdout') and not absence.get('stderr'), 'PID not absent after reap')
        need(parse_listener(listeners) == listeners.get('pids') == [] and listeners.get('returncode') == 1
             and not listeners.get('stderr'), 'listener remained or inventory unknown after reap')
        need(type(release.get('tcp_connect_ex_after_reap')) is int and release['tcp_connect_ex_after_reap'] != 0,
             'TCP port still accepts connections')
        if ready:
            need(release.get('port') == ready['listeners'].get('port') == listeners.get('port'), 'release checked another port')
        audit.check(release.get('valid') == (len(result['issues']) == errors_before_release), folder / 'release.json', 'release validity differs from evidence')
        result['returncode'] = release.get('returncode')
        result['kill_escalated'] = release.get('kill_escalated')
    else:
        need(False, 'release receipt missing')
    result['valid'] = not result['issues']
    result['live_listener_scope'] = 'persisted readiness listener and final absence; per-request listener assertions were not individually serialized'
    return result


def resource_audit(audit, folder, pid, controller, requests, lease, resume):
    paths = sorted(folder.glob('resource-[0-9][0-9][0-9].json'))
    raw = [audit.load(p) for p in paths]
    raw = [r for r in raw if r is not None]
    bad = []
    def need(condition, detail):
        if not condition:
            bad.append(detail)
    need(bool(raw), 'no resource snapshots')
    labels = [r.get('label') for r in raw]
    need(bool(raw) and labels[0] == 'before-launch' and labels[-1] == 'after-unload', 'resource lifecycle boundaries missing')
    times = [r.get('monotonic_s') for r in raw]
    need(all(number(t) for t in times) and all(a < b for a, b in zip(times, times[1:])), 'resource timestamps invalid/unordered')
    derived = audit.load(folder / 'resources.json')
    if derived is not None:
        audit.check(len(derived) == len(raw), folder, 'raw/derived resource counts differ')
    recomputed_deltas = []
    for i, item in enumerate(raw):
        need(item.get('process_inventory_ok') and item.get('native_process_inventory_ok'), 'unknown process inventory')
        need(item.get('pressure_level_value') is not None and item['pressure_level_value'] < 2
             and item.get('free_percent') is not None and item.get('swap_used_mb') is not None, 'unknown/unsafe resource pressure')
        need(item.get('power_metadata', {}).get('source') == 'AC Power', 'not AC power')
        need(item.get('processinfo_thermal_metadata', {}).get('code') in (0, 1), 'thermal unknown/serious')
        native = item.get('native_inference_processes', [])
        need(all(p.get('owned') == (p.get('pid') == item.get('owned_model_pid')) for p in native), 'native inventory owned flag differs from PID')
        need(all(p.get('owned') or (p.get('pid') == 74384 and p.get('authorized_suspended') and 'T' in p.get('state', '')) for p in native),
             'active foreign inference process in snapshot')
        stopped = [p for p in native if p.get('pid') == 74384]
        need(len(stopped) == 1 and stopped[0].get('authorized_suspended') and 'T' in stopped[0].get('state', ''),
             'authorized service not observed stopped')
        need(all(p.get('pid') == item.get('owned_model_pid') or p.get('pid') == 74384
                 for p in item.get('known_inference_processes', [])), 'foreign command-matched inference process')
        if item.get('label') not in ('before-launch', 'after-unload'):
            exact = item.get('exact_owned_pid_metadata', {})
            need(item.get('owned_model_pid') == pid and exact.get('pid') == pid and exact.get('ppid') == controller
                 and exact.get('status') == 'known', 'resource owner not direct GPU server')
        if lease.get('paused_at') and resume and resume.get('time'):
            try:
                when = datetime.datetime.fromisoformat(item['recorded_at']).timestamp()
                need(lease['paused_at'] <= when <= resume['time'], 'resource sample outside pause/restore interval')
            except Exception:
                need(False, 'resource wall timestamp invalid')
        if derived is not None and i < len(derived):
            for key, value in item.items():
                audit.check(derived[i].get(key) == value, folder / 'resources.json', f'raw resource field differs at {i}: {key}')
        if i:
            previous = raw[i - 1]
            swap = item['swap_used_mb'] - previous['swap_used_mb'] if number(item.get('swap_used_mb')) and number(previous.get('swap_used_mb')) else None
            delta = {k: item.get('vm_counters', {}).get(k) - previous.get('vm_counters', {}).get(k)
                     for k in COUNTERS if type(item.get('vm_counters', {}).get(k)) is int
                     and type(previous.get('vm_counters', {}).get(k)) is int}
            recomputed_deltas.append({'index': i, 'swap_used_mb_delta': swap, 'vm_counter_delta': delta})
            if derived is not None and i < len(derived):
                audit.check(equal(derived[i].get('swap_used_mb_delta'), swap) and derived[i].get('vm_counter_delta') == delta,
                            folder / 'resources.json', f'counter delta differs at {i}')
    for before, after in zip(recomputed_deltas, recomputed_deltas[1:]):
        need(not (number(before['swap_used_mb_delta']) and number(after['swap_used_mb_delta'])
                  and before['swap_used_mb_delta'] > 64 and after['swap_used_mb_delta'] > 64), 'sustained >64 MiB swap growth')
        need(not (before['vm_counter_delta'].get('Swapouts', 0) > 0 and after['vm_counter_delta'].get('Swapouts', 0) > 0),
             'sustained swapouts')
    for request in requests:
        label = f'{request["phase"]}-{request["index"]}'
        before = [s for s in raw if s.get('label') == 'before-' + label]
        after = [s for s in raw if s.get('label') == 'after-' + label]
        need(len(before) == 1, 'missing before resource sample for ' + label)
        if request['valid']:
            need(len(after) == 1, 'missing after resource sample for ' + label)
        if before and number(request['start_s']):
            need(before[0]['monotonic_s'] <= request['start_s'], 'before probe overlaps request start')
        if after and number(request['cleanup_s']):
            need(after[0]['monotonic_s'] >= request['cleanup_s'], 'after probe overlaps HTTP stream cleanup')
    totals = {k: raw[-1].get('vm_counters', {}).get(k) - raw[0].get('vm_counters', {}).get(k)
              for k in COUNTERS if raw and type(raw[-1].get('vm_counters', {}).get(k)) is int
              and type(raw[0].get('vm_counters', {}).get(k)) is int}
    return {'valid': not bad, 'issues': bad, 'samples': len(raw), 'labels': labels,
        'counter_deltas': totals, 'per_interval_deltas': recomputed_deltas,
        'swap_used_mb_delta': raw[-1]['swap_used_mb'] - raw[0]['swap_used_mb'] if raw and number(raw[-1].get('swap_used_mb')) and number(raw[0].get('swap_used_mb')) else None,
        'max_sampled_server_rss_kib': max((s.get('owned_pid_rss_kib') or 0 for s in raw), default=None),
        'first_monotonic_s': times[0] if times else None, 'last_monotonic_s': times[-1] if times else None,
        'scope': 'sampled RSS/counters, not GPU allocation peak; nonzero Swapins are reported, not erased'}


def pin_audit(audit, inputs, run, plan, tokenizer_path):
    pins, stats = plan['pins'], plan['pin_stats']
    frozen = inputs / 'frozen-sources'
    old_inputs = Path(plan['native_source_root']).parent
    counts = Counter()
    records = {}
    for filename, expected in pins.items():
        original = Path(filename)
        candidate = frozen / filename.lstrip('/')
        local = None
        if candidate.is_file():
            local = candidate
            kind = 'frozen_source_independently_hashed'
        else:
            try:
                remapped = inputs / original.relative_to(old_inputs)
            except ValueError:
                remapped = None
            if remapped is not None and remapped.is_file():
                local = remapped
                kind = 'frozen_input_independently_hashed'
            elif original.suffix in SOURCE_SUFFIXES and stats[filename]['bytes'] < 4 << 20:
                kind = 'missing_required_frozen_source'
                audit.check(False, filename, 'small planned source/config has no frozen copy')
            else:
                kind = 'recorded_payload_or_runtime_hash_only'
        if local:
            audit.check(sha(local) == expected and local.stat().st_size == stats[filename]['bytes'], local, 'frozen bytes differ from planned SHA/size')
        counts[kind] += 1
        records[filename] = {'sha256': expected, 'bytes': stats[filename]['bytes'], 'audit_scope': kind}
    # Archive enough non-weight tokenizer/config metadata for CPU-only replay.
    target = Path(plan['workload']['model_dir'])
    for name in ('config.json', 'generation_config.json', 'tokenizer_config.json', 'chat_template.jinja'):
        filename = str(target / name)
        if filename not in pins:
            continue
        path = frozen / filename.lstrip('/')
        if not path.is_file():
            path = target / name
        if path.is_file():
            audit.check(sha(path) == pins[filename], path, 'target metadata changed')
            audit.collect(path, 'model-metadata/target/' + name)
    expected_tok = pins.get(str(target / 'tokenizer.json'))
    audit.check(expected_tok is not None and sha(tokenizer_path) == expected_tok, tokenizer_path, 'audit tokenizer is not the planned tokenizer')
    audit.collect(tokenizer_path, 'model-metadata/target/tokenizer.json')
    before, after = audit.load(run / 'pins-before.json'), audit.load(run / 'pins-after.json')
    for label, receipt in [('before', before), ('after', after)]:
        if receipt is not None:
            audit.check(receipt.get('all_full_sha_match') is True and receipt.get('count') == len(pins), run / f'pins-{label}.json', 'runner full-pin receipt invalid')
    return {'counts': dict(counts), 'total_pins': len(pins), 'recorded_pins': records,
        'runner_full_sha_before': bool(before and before.get('all_full_sha_match')),
        'runner_full_sha_after': bool(after and after.get('all_full_sha_match')),
        'scope': 'Frozen source/config/tokenizer bytes independently hashed. Large weights/packs/binaries not reread; their full verification is the runner recorded receipt.'}


def lease_audit(audit, args, attempt, plan_sha):
    lease = audit.load(args.lease / 'lease.json', required=True) or {}
    resume = audit.load(args.lease / 'resume.json')
    jobs = audit.load(args.jobs, required=True) or []
    job = audit.load(args.lease / 'job-0.json')
    problems = []
    def need(condition, detail):
        if not condition:
            problems.append(detail)
    need(lease.get('service_pid') == 74384 and bool(lease.get('service_identity')), 'wrong/missing authorized service identity')
    need(lease.get('closed') is True and bool(lease.get('paused_at')), 'lease not paused and closed')
    need(lease.get('jobs') == jobs and len(jobs) == 1, 'job list does not match single direct controller')
    if attempt:
        need(attempt.get('parent_pid') == lease.get('owner_pid') and attempt.get('pid') == lease.get('child_pid'), 'controller not direct lease child')
        need(attempt.get('no_guarded_wrapper') is True, 'guarded wrapper not excluded')
    if jobs:
        command = jobs[0]
        need(not any('guarded_v2.py' in part for part in command), 'GPU-owner guard wraps a controller')
        need(any(part.endswith('/external_spec_http/run.py') for part in command), 'job is not E020 controller')
        try:
            need(command[command.index('--plan-sha256') + 1] == plan_sha, 'job uses a different plan SHA')
        except (ValueError, IndexError):
            need(False, 'job lacks explicit reviewed plan SHA')
    if job:
        need(job.get('command') == (jobs[0] if jobs else None), 'executed job command differs')
        need(job.get('pid') == (attempt or {}).get('pid') and job.get('waited') is True and type(job.get('returncode')) is int,
             'controller job was not waited/reaped')
    else:
        need(False, 'controller job completion receipt missing')
    if resume:
        need(resume.get('identity_match') is True and 'T' in resume.get('state_before', '')
             and 'T' not in resume.get('state_after', '') and bool(resume.get('state_after')), 'service identity/restore transition failed')
        need(number(resume.get('time')) and number(lease.get('paused_at')) and resume['time'] >= lease['paused_at'], 'invalid restore timestamp')
    else:
        need(False, 'service resume receipt missing')
    return {'valid': not problems, 'issues': problems, 'lease': lease, 'resume': resume, 'job': job,
            'scope': 'persisted identity and stopped-to-running transition; no signal or live service operation performed by audit'}


def launch_audit(audit, folder, arm, plan, ready):
    launch = audit.load(folder / 'launch.json') or {}
    command = launch.get('command') or []
    bad = []
    def need(condition, detail):
        if not condition:
            bad.append(detail)
    def flag(name):
        try:
            return command[command.index(name) + 1]
        except (ValueError, IndexError):
            return None
    need(not any('guarded_v2.py' in str(part) for part in command), 'wrapped server command')
    need(all(launch.get('environment', {}).get(k) == v for k, v in plan['environment'].items()), 'different MLX environment')
    if arm == 'native-none':
        need(len(command) >= 3 and command[2].endswith('/external_spec_http/native_server.py') and command[1] == '-B', 'wrong native direct entrypoint')
        need(flag('--package-root') == plan['native_source_root'] and flag('--source-manifest') == plan['native_source_manifest'], 'native source snapshot differs')
        identity = (ready or {}).get('models', {}).get('benchmark_identity') or {}
        need(identity.get('manifest_all_sources_match') and identity.get('cache_limit_gib') == 2
             and identity.get('compile_decode') is True and identity.get('fuse_decode') is True
             and identity.get('gates_compile') is True and identity.get('norm_reuse') is False
             and identity.get('drafter') is None and identity.get('step_observer') is False
             and identity.get('normal_eos') is True and identity.get('max_seqs') == 4,
             'native runtime configuration differs')
        need(identity.get('grouped_prefill') == 'unmodified core default', 'native grouped prefill default changed')
    elif arm.startswith('omlx-'):
        need(len(command) >= 2 and command[0].endswith('/omlx-baseline/.venv/bin/omlx') and command[1] == 'serve', 'wrong oMLX original CLI')
        need(all(flag in command for flag in ('--headless', '--no-cache', '--no-hf-cache')), 'missing official no-cache/headless flags')
        need(flag('--max-concurrent-requests') == '1' and flag('--sse-keepalive-mode') == 'off', 'different concurrency/keepalive')
        configs = launch.get('files', {})
        model_file = next((v for k, v in configs.items() if k.endswith('/model_settings.json')), {})
        need(model_file.get('version') == 1 and model_file.get('models') == {'Qwen3.8-27B-nvfp4': plan['qualified_model_settings'][arm]},
             'declared per-model settings differ from qualification')
        settings = next((v for k, v in configs.items() if k.endswith('/settings.json')), {})
        need(settings.get('server', {}).get('burst_decode_mode') == 'balanced'
             and settings.get('cache', {}).get('enabled') is False
             and settings.get('scheduler') == {'max_concurrent_requests': 1, 'chunked_prefill': False,
                 'prefill_priority': 'speed', 'decode_fairness': True}, 'official global scheduler/cache settings differ')
        # Official settings may add bookkeeping; all frozen qualification keys
        # must remain equal in the resulting actual model settings file.
        actual = audit.load(folder / 'state/model_settings.json')
        if actual:
            effective = actual.get('models', {}).get('Qwen3.8-27B-nvfp4', {})
            need(all(effective.get(k) == v for k, v in plan['qualified_model_settings'][arm].items()),
                 'actual model settings no longer preserve qualified values')
        else:
            need(False, 'actual oMLX model_settings.json missing')
    else:
        need(len(command) >= 4 and command[1:4] == ['-B', '-m', 'monolith.serve'], 'wrong Lithos direct entrypoint')
        need(flag('--max-context') == '2048' and flag('--prefill-chunk-size') == '128-exact'
             and flag('--keep-warm') == '0' and '--no-warmup' in command and '--local-files-only' in command,
             'Lithos context/prefill/warm policy differs')
        need(flag('--pack') == plan['packs'][arm]['target']['path'] and '--kernel-config' not in command,
             'target pack/recipe differs')
        need(launch.get('environment', {}).get('MONOLITH_ACCELERATOR') == 'on', 'accelerator environment differs')
        if arm == 'lithos-none':
            need('--no-draft' in command and '--draft' not in command, 'Lithos none enabled draft')
        else:
            need(flag('--draft') == plan['drafters']['lithos'] and flag('--draft-pack') == plan['packs'][arm]['draft']['path']
                 and flag('--draft-kind') == 'dspark' and flag('--draft-quantization') == 'none'
                 and flag('--draft-block-size') == '7' and flag('--draft-sampling') == 'argmax', 'DSpark draft/quantization/block differs')
    need(flag('--model') == plan['workload']['model_dir'] if not arm.startswith('omlx-') else
         list(launch.get('symlinks', {}).values()) == [plan['workload']['model_dir']], 'target path differs')
    return {'valid': not bad, 'issues': bad}


def independent_audit(audit, args):
    plan = audit.load(args.inputs / 'plan.json', required=True)
    if plan is None:
        raise ValueError('No readable prepared plan; retaining raw evidence only')
    plan_sha = sha(args.inputs / 'plan.json')
    declared = [(c['pair'], c['arm']) for c in plan.get('configs', [])]
    audit.check(declared == MATRIX and plan.get('concurrency') == 1 and plan.get('normal_eos') is True,
                args.inputs / 'plan.json', 'matrix or EOS/C1 contract differs from preregistration')
    audit.check(plan.get('warm_requests') == plan.get('formal_requests') == 3, args.inputs / 'plan.json', 'request counts differ')
    audit.check(plan.get('expected_counts') == {'fresh_servers': 15, 'warm_requests': 45, 'formal_requests': 45,
                                              'formal_output_tokens_if_valid': 1440}, args.inputs / 'plan.json', 'declared matrix arithmetic differs')
    observed_plan = audit.load(args.run / 'plan.json')
    if observed_plan is not None:
        audit.check(observed_plan == plan, args.run / 'plan.json', 'runtime plan differs from prepared plan')
    preparation = audit.load(args.inputs / 'preparation.json')
    if preparation:
        audit.check(preparation.get('plan_sha256') == plan_sha and preparation.get('status') == 'PASS_CPU_PREPARED_NOT_RUN',
                    args.inputs / 'preparation.json', 'preparation receipt differs')
    attempt = audit.load(args.run / 'attempt.json') or {}
    if attempt:
        audit.check(attempt.get('plan_sha256') == plan_sha, args.run / 'attempt.json', 'controller reviewed another plan SHA')
    controller = attempt.get('pid')
    service = lease_audit(audit, args, attempt, plan_sha)
    lease, resume = service['lease'], service['resume']
    tokenizer_path = args.tokenizer or Path(plan['workload']['model_dir']) / 'tokenizer.json'
    pins = pin_audit(audit, args.inputs, args.run, plan, tokenizer_path)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    for engine, file in [('omlx', 'official-mtp-comparison-v1.json'), ('lithos', 'lithos-dspark-comparison-v1.json')]:
        prior = audit.load(args.inputs / file, required=True) or {}
        audit.check(plan['oracle_ids'][engine] == prior.get('raw_ids'), args.inputs / file, 'engine-specific oracle differs')
    audit.check(plan['oracle_ids']['native'] == plan['workload']['continuation_ids'], 'native oracle', 'native expected IDs differ from fixed workload')
    native_manifest = audit.load(args.inputs / 'native-source.json', required=True) or {}
    processes, all_requests, declared_results = [], [], []
    for pair, arm in MATRIX:
        folder = args.run / f'pair{pair}-{arm}'
        if not folder.exists():
            processes.append({'pair': pair, 'arm': arm, 'state': 'not_started', 'valid': False})
            continue
        rows = []
        for phase in ('warm', 'formal'):
            for index in range(3):
                path = folder / f'{phase}-{index}.json'
                row = audit.load(path)
                if row is None:
                    continue
                try:
                    derived = request_audit(audit, path, row, plan, tok, pair, arm, phase, index)
                except Exception as error:
                    audit.check(False, path, f'cannot reconstruct malformed request: {type(error).__name__}: {error}')
                    derived = {'pair': pair, 'arm': arm, 'phase': phase, 'index': index, 'valid': False,
                               'failures': ['malformed request'], 'start_s': None, 'done_s': None, 'cleanup_s': None}
                rows.append(derived)
                all_requests.append(derived)
        ready = audit.load(folder / 'ready.json')
        launch = launch_audit(audit, folder, arm, plan, ready)
        if arm == 'native-none' and ready:
            audit.check(ready.get('models', {}).get('benchmark_identity', {}).get('source_files') == native_manifest.get('files'),
                        folder / 'ready.json', 'native actual loaded source differs from frozen manifest')
        owner = lifecycle_audit(audit, folder, controller, lease, resume)
        resources = resource_audit(audit, folder, owner.get('pid'), controller, rows, lease, resume)
        log = (folder / 'server.log').read_text(errors='replace') if (folder / 'server.log').is_file() else ''
        route = route_audit(arm, log, plan)
        recorded_route = audit.load(folder / 'route.json')
        if recorded_route:
            audit.check(recorded_route.get('valid') == route['valid'], folder / 'route.json', 'independent official-log route status differs')
        cleanup = audit.load(folder / 'cleanup.json') or {}
        done = audit.load(folder / 'result.json')
        if done:
            declared_results.append(done)
        formal = [r for r in rows if r['phase'] == 'formal']
        complete_count = len(rows) == 6 and len(formal) == 3
        valid = (complete_count and all(r['valid'] for r in rows) and owner['valid'] and resources['valid']
                 and launch['valid'] and route['valid'] and not cleanup.get('first_failure') and not cleanup.get('cleanup_errors')
                 and bool(done))
        keys = {'median_wall_s': 'E2E_s', 'median_TTFT_s': 'TTFT_s', 'median_output_tok_s': 'aggregate_output_tok_s',
                'median_client_estimated_interactivity_tok_s': 'client_estimated_interactivity_tok_s'}
        medians = {}
        for key, field in keys.items():
            values = [r.get(field) for r in formal if number(r.get(field))]
            medians[key] = statistics.median(values) if len(values) == 3 else None
        if done:
            audit.check(done.get('valid') == valid, folder / 'result.json', 'independent full process validity differs')
            audit.check(done.get('request_count') == len(rows) and done.get('formal_count') == len(formal), folder / 'result.json', 'request counts differ')
            for key, value in medians.items():
                audit.check(equal(done.get('diagnostic_medians_including_invalid_samples', {}).get(key), value), folder / 'result.json', 'raw median differs: ' + key)
                audit.check(equal(done.get(key), value if valid else None), folder / 'result.json', 'validated median differs: ' + key)
        processes.append({'pair': pair, 'arm': arm, 'state': 'complete' if done and complete_count else 'partial', 'valid': valid,
            'request_count': len(rows), 'formal_count': len(formal), 'request_contract_valid': all(r['valid'] for r in rows) and complete_count,
            'diagnostic_medians': medians, **{k: v if valid else None for k, v in medians.items()},
            'launch': launch, 'lifecycle': owner, 'resources': resources, 'route': route,
            'cleanup': cleanup, 'recorded_result_present': done is not None})
    seen = [p['lifecycle']['pid'] for p in processes if p.get('lifecycle', {}).get('pid')]
    audit.check(len(seen) == len(set(seen)), args.run, 'server PID reused inside matrix; fresh-owner proof needs closer review')
    for left, right in zip(processes, processes[1:]):
        before = left.get('resources', {}).get('last_monotonic_s')
        after = right.get('resources', {}).get('first_monotonic_s')
        if number(before) and number(after):
            audit.check(before < after, args.run, 'planned server intervals overlap/out of order')
    for before, after in zip(all_requests, all_requests[1:]):
        if number(before.get('cleanup_s')) and number(after.get('start_s')):
            audit.check(before['cleanup_s'] < after['start_s'], args.run, 'C1 requests overlap/out of order')
    pairs, aggregates = [], {}
    for engine, on_name in [('omlx', 'mtp'), ('lithos', 'dspark')]:
        for pair in (1, 2, 3):
            chosen = {p['arm']: p for p in processes if p['pair'] == pair and p['valid']}
            off, on = chosen.get(engine + '-none'), chosen.get(engine + '-' + on_name)
            both = bool(off and on)
            pairs.append({'engine': engine, 'pair': pair, 'valid': both,
                'throughput_speedup': on['median_output_tok_s'] / off['median_output_tok_s'] if both else None,
                'TTFT_speedup': off['median_TTFT_s'] / on['median_TTFT_s'] if both else None,
                'wall_speedup': off['median_wall_s'] / on['median_wall_s'] if both else None})
        points = [p['throughput_speedup'] for p in pairs if p['engine'] == engine and p['valid']]
        aggregates[engine] = {'three_pairs_complete': len(points) == 3,
            'paired_throughput_geomean': math.exp(statistics.mean(math.log(p) for p in points)) if len(points) == 3 else None,
            'every_pair_faster': all(p > 1 for p in points) if len(points) == 3 else None,
            'all_paired_points': points}
    complete = audit.load(args.run / 'complete.json')
    all_finished = len(declared_results) == 15 and all(p['state'] == 'complete' for p in processes)
    if complete:
        audit.check(all_finished, args.run / 'complete.json', 'complete claim without all 15 processes')
        audit.check(complete.get('rows') == declared_results, args.run / 'complete.json', 'complete process rows differ from original results')
        expected_status = 'PASS_CONTRACT' if all(p['valid'] for p in processes) else 'FAIL_CONTRACT_RESULTS_RETAINED'
        audit.check(complete.get('status') == expected_status, args.run / 'complete.json', 'overall status differs')
        for observed, calculated in zip(complete.get('pairs', []), pairs):
            audit.check(set(observed) == set(calculated) and all(equal(observed[k], v) for k, v in calculated.items()),
                        args.run / 'complete.json', 'paired ratio differs')
        audit.check(len(complete.get('pairs', [])) == 6, args.run / 'complete.json', 'paired ratio count differs')
        for engine, values in aggregates.items():
            observed = complete.get('descriptive_aggregates', {}).get(engine, {})
            audit.check(set(observed) == set(values) and all(equal(observed[k], v) for k, v in values.items()),
                        args.run / 'complete.json', 'descriptive aggregate differs: ' + engine)
        audit.check(pins['runner_full_sha_before'] and pins['runner_full_sha_after'], args.run, 'full-run pin receipts incomplete')
    counts = {'planned_servers': 15, 'started_servers': len(seen), 'finished_servers': sum(p['state'] == 'complete' for p in processes),
        'valid_servers': sum(p['valid'] for p in processes), 'warm_requests': sum(r['phase'] == 'warm' for r in all_requests),
        'formal_requests': sum(r['phase'] == 'formal' for r in all_requests), 'valid_requests': sum(r['valid'] for r in all_requests),
        'actual_formal_usage_tokens': sum(r.get('actual_usage_count', 0) for r in all_requests
                                        if r['phase'] == 'formal' and type(r.get('actual_usage_count')) is int)}
    return {'schema': 'E020-independent-audit-v1', 'plan_sha256': plan_sha, 'counts': counts,
        'experiment_status': ('COMPLETE_VALID_CONTRACT' if all(p['valid'] for p in processes) and complete
                              and service['valid'] and pins['runner_full_sha_before'] and pins['runner_full_sha_after'] else
                              'COMPLETE_WITH_INVALID_CONTRACTS' if all_finished and complete else 'PARTIAL_ATTEMPT'),
        'service': service, 'pins': pins, 'processes': processes, 'requests': all_requests,
        'pairs': pairs, 'descriptive_aggregates': aggregates,
        'scope': 'C1 19->32. Cross-framework performance comparison uses distinct qualified output contracts; no shared-oracle quality equivalence. All negative points retained. No adoption gate.'}


def archive(audit, out):
    manifest = {name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in sorted(audit.files.items())}
    path = out / 'evidence.tar.gz'
    with path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as tar:
                for name, source in sorted(audit.files.items()):
                    info = tarfile.TarInfo(name)
                    info.size = manifest[name]['bytes']
                    info.mode = 0o644
                    info.mtime = 0
                    with source.open('rb') as stream:
                        tar.addfile(info, stream)
    with tarfile.open(path, mode='r:gz') as tar:
        names = tar.getnames()
        if len(names) != len(manifest) or set(names) != set(manifest):
            raise ValueError('tar member set/count differs from manifest')
        for member in tar:
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError('unexpected nonregular archive member')
            h, size = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(8 << 20), b''):
                h.update(block)
                size += len(block)
            if {'bytes': size, 'sha256': h.hexdigest()} != manifest[member.name]:
                raise ValueError('tar content differs: ' + member.name)
    archive_sha = sha(path)
    write(out / 'manifest.json', {'archive': path.name, 'archive_sha256': archive_sha, 'files': manifest})
    write(out / 'tar.verify.json', {'status': 'PASS', 'members': len(manifest), 'archive_bytes': path.stat().st_size,
                                  'archive_sha256': archive_sha, 'all_members_independently_rehashed': True})
    return {'members': len(manifest), 'archive_bytes': path.stat().st_size, 'archive_sha256': archive_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=BASE / 'external-spec-http-v1-inputs')
    parser.add_argument('--run', type=Path, default=BASE / 'external-spec-http-v1')
    parser.add_argument('--lease', type=Path, default=BASE / 'external-spec-http-v1-service-lease')
    parser.add_argument('--jobs', type=Path, default=BASE / 'external-spec-http-v1-jobs.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, help='optional archived tokenizer for CPU replay')
    parser.add_argument('--extra', type=Path, action='append', default=[], help='additional CPU/preflight/failure receipt to preserve')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    audit.collect_tree(args.inputs, 'inputs')
    audit.collect_tree(args.run, 'run')
    audit.collect_tree(args.lease, 'lease')
    audit.collect(args.jobs, 'jobs.json')
    audit.collect(Path(__file__), 'audit/external_spec_http_audit.py')
    for i, path in enumerate(args.extra):
        audit.collect(path, f'extra/{i:02d}-{path.name}')
    try:
        summary = independent_audit(audit, args)
    except Exception as error:
        audit.check(False, 'audit', f'unhandled/malformed evidence: {type(error).__name__}: {error}')
        summary = {'schema': 'E020-independent-audit-v1', 'experiment_status': 'UNKNOWN_OR_PARTIAL',
                   'audit_exception': traceback.format_exc()}
    summary['audit_status'] = 'PASS_RECONCILED' if not audit.integrity_errors else 'FAIL_INDEPENDENT_RECONCILIATION'
    summary['integrity_errors'] = audit.integrity_errors
    summary['notes'] = audit.notes
    write(args.out / 'summary.json', summary)
    audit.collect(args.out / 'summary.json', 'audit/summary.json')
    proof = archive(audit, args.out)
    counts = summary.get('counts', {})
    lines = [
        '# E020 external speculative HTTP independent audit', '',
        f"Audit: **{summary['audit_status']}**. Experiment: **{summary['experiment_status']}**.", '',
        f"Started/finished/valid servers: {counts.get('started_servers', '?')} / {counts.get('finished_servers', '?')} / {counts.get('valid_servers', '?')} of 15.",
        f"Warm/formal requests: {counts.get('warm_requests', '?')} / {counts.get('formal_requests', '?')}; valid requests {counts.get('valid_requests', '?')}.", '',
        'This audit reconstructs text, counts, clocks, E2E, TTFT, C1 aggregate throughput and process medians from raw SSE; stock logs are separately checked for declared/executed routes. All negative, partial and invalid rows remain in summary.json. A failed experiment can still be faithfully reconciled; audit PASS is not a speed/adoption claim.', '',
        '| Engine | Pair | Valid contract | Throughput ratio (on/off) | E2E speedup (off/on) | TTFT speedup (off/on) |',
        '|---|---:|---|---:|---:|---:|']
    for row in summary.get('pairs', []):
        def fmt(value): return f'{value:.6f}' if number(value) else 'unavailable'
        lines.append(f"| {row['engine']} | {row['pair']} | {row['valid']} | {fmt(row['throughput_speedup'])} | {fmt(row['wall_speedup'])} | {fmt(row['TTFT_speedup'])} |")
    lines += ['',
        'Cross-framework latency/throughput may be compared with explicit output contracts: native/oMLX and Lithos do not share an identical qualified continuation. External visible text is not a new live engine raw-ID capture. MTP log denominators are nominal proposal capacity; Lithos verify width/steps do not constitute a fresh acceptance trace.', '',
        'The archive retains frozen source/config, model tokenizer metadata, all raw run/lease/jobs receipts, original official logs and recorded package/payload hashes. It excludes weight/pack payloads and runnable virtual environments. Full payload checks were performed by the runner before/after; this CPU audit independently hashes frozen source/metadata and every archive member without rereading large weights.', '',
        f"Verified archive: {proof['members']} regular members, {proof['archive_bytes']} compressed bytes. SHA256 `{proof['archive_sha256']}`.", '',
        'Use manifest.json and tar.verify.json to verify the evidence. For offline CPU replay, extract the archive and pass the extracted inputs/, run/, lease/, jobs.json plus --tokenizer model-metadata/target/tokenizer.json to the archived audit script; choose a new --out. No server or GPU is required.', '',
        'Nonzero Swapins/other paging counters and sampled RSS limits are retained per process. Service restoration is established only from the outer lease identity/stop-to-running receipt. No zero-paging, GPU allocation-peak or global Pareto claim is made.', '']
    (args.out / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'audit_status': summary['audit_status'], 'experiment_status': summary['experiment_status'],
                      'counts': counts, 'integrity_errors': len(audit.integrity_errors), **proof}))


if __name__ == '__main__':
    main()
