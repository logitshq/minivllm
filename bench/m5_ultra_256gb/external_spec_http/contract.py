"""E020 constants and pure validation; safe to import without a model runtime."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / '.artifacts/m5-ultra-256gb'
LITHOS = Path('/Users/ryanli/code/lithos-metal')
WORKLOAD = BASE / 'fullqual-build2/workload.json'
WORKLOAD_SHA = '65be876cac2397bf9afa76105c8e3fcce23de5e6c8652a28cb1b5f39640c153c'
LITHOS_HEAD = '223658090ae549e59220bdec459957cbdad7926a'
USER_TEXT = '用一句话解释什么是 KV cache。'
MODEL_ID = 'Qwen3.8-27B-nvfp4'
ENV = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
       'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}
QUALIFICATIONS = {
    'omlx-none': BASE / 'official-mtp-none-v1',
    'omlx-mtp': BASE / 'official-mtp-mtp-v1',
    'lithos-none': BASE / 'lithos-dspark-none-v1',
    'lithos-dspark': BASE / 'lithos-dspark-dspark-v1',
}
ORDERS = [
    ['native-none', 'omlx-none', 'omlx-mtp', 'lithos-none', 'lithos-dspark'],
    ['lithos-dspark', 'lithos-none', 'native-none', 'omlx-mtp', 'omlx-none'],
    ['omlx-none', 'omlx-mtp', 'lithos-none', 'lithos-dspark', 'native-none'],
]
CONFIGS = [{'pair': pair, 'arm': arm} for pair, order in enumerate(ORDERS, 1) for arm in order]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_request(row, expected_ids, *, native=False):
    """Return every failed condition. Never replace a failed request with a retry."""
    failures = []
    def need(condition, message):
        if not condition:
            failures.append(message)
    usage = row.get('usage') or {}
    ids = row.get('visible_roundtrip_ids', [])
    need(row.get('http_status') == 200, 'HTTP status is not 200')
    need(not row.get('error'), 'stream/client error')
    need(row.get('done') is True, 'missing SSE DONE')
    need(row.get('finish_reason') == 'length', 'normal EOS/finish differs from the fixed length workload')
    need(usage.get('prompt_tokens') == 19, 'prompt usage is not 19')
    need(usage.get('completion_tokens') == 32, 'actual completion usage is not 32')
    need(len(ids) == usage.get('completion_tokens') == 32, 'canonical visible count differs from usage')
    need(ids == expected_ids, 'visible IDs differ from this engine qualified raw-ID sequence')
    need(row.get('first_text_perf_counter_s') is not None, 'no visible text')
    need(row.get('first_chunk_prefix_verified') is True, 'first chunk canonical prefix is ambiguous')
    cached = row.get('cached_prompt_tokens_when_available')
    need(cached is None or cached == 0, 'HTTP reports a prefix-cache hit')
    if native:
        need(row.get('output_ids_when_available') == expected_ids, 'native live raw IDs differ from native oracle')
        need(cached == 0, 'native cache absence not explicitly reported')
    return failures


def official_log_evidence(arm, text, *, target, draft, packs):
    """Official logs only. These checks do not instrument the model or claim raw IDs."""
    failures, evidence = [], {'scope': 'ordinary official server logs, no observer hooks'}
    def need(condition, message):
        if not condition:
            failures.append(message)
    if arm.startswith('omlx-'):
        starts = re.findall(r'vlm_mtp decode started: request=(\S+) uid=(-?\d+) block_size=(\S+)', text)
        pattern = (r'vlm_mtp stats: request=(\S+) finish=(\S+) rounds=(\d+) '
                   r'accepted=(\d+)/(\d+) \(([\d.]+)%\) tokens_per_round=([\d.]+) '
                   r'emitted=(\d+) block_size=(\d+)')
        stats = [{'request': m[0], 'finish': m[1], 'rounds': int(m[2]), 'accepted': int(m[3]),
                  'nominal_capacity_denominator': int(m[4]), 'reported_percent': float(m[5]),
                  'reported_tokens_per_round': float(m[6]), 'emitted': int(m[7]), 'block_size': int(m[8])}
                 for m in re.findall(pattern, text)]
        evidence.update(starts=starts, stats=stats,
            denominator_scope='rounds * (block_size - 1); NOT actual proposed tokens in the final shortened round')
        if arm == 'omlx-mtp':
            need(f'VLM MTP enabled for {MODEL_ID}, drafter={draft}' in text, 'official MTP attachment log absent')
            need(len(starts) == len(stats) == 6, 'not exactly six official MTP starts and completions')
            need(len({s[0] for s in starts}) == 6, 'MTP request IDs are not distinct')
            need([s[0] for s in starts] == [s['request'] for s in stats], 'MTP start/stats order differs')
            need(all(s['finish'] == 'length' and s['emitted'] == 32 and s['block_size'] == 3
                     and s['rounds'] > 0 and 0 < s['accepted'] <= s['nominal_capacity_denominator']
                     and s['nominal_capacity_denominator'] == s['rounds'] * 2 for s in stats),
                 'MTP route/finish/block/acceptance log contract failed')
            warnings = [line for line in text.splitlines()
                        if re.search(r'(vlm_mtp|VLM MTP|vlm-mtp)', line, re.I)
                        and re.search(r'skipped|ignored|falling back|failed|fallback', line, re.I)]
            evidence['mtp_route_warning_lines'] = warnings
            need(not warnings, 'official MTP fallback/skip warning present')
        else:
            need(not starts and not stats and 'VLM MTP enabled for' not in text, 'none arm executed MTP')
    elif arm.startswith('lithos-'):
        dspark = arm == 'lithos-dspark'
        rows = [ast.literal_eval(line.split('Generation: ', 1)[1]) for line in text.splitlines()
                if 'Generation: ' in line]
        serving = [line for line in text.splitlines() if 'Serving backend=' in line]
        evidence.update(generation=rows, serving=serving,
            acceptance_scope='verify_tokens is configured width; steps are diagnostics, not an acceptance trace')
        expected_verify = 8 if dspark else 1
        need(len(serving) == 1 and f'backend=m5_ultra_80c, target={target},' in serving[0]
             and f'verify={expected_verify}, recipe contexts=[]' in serving[0], 'wrong backend/target/verify/recipes')
        if serving:
            need(f'draft={draft if dspark else None},' in serving[0], 'unexpected drafter')
        need(len(rows) == 6, 'not exactly six official Generation logs')
        need(all(row.get('cached_prompt_tokens') == 0 and row.get('verify_tokens') == expected_verify
                 for row in rows), 'cache-hit or configured verify-width mismatch')
        need(all(row.get('setup_ms') == 0 for row in rows[3:]), 'formal request includes program setup')
        for role, pack in packs.items():
            if pack is not None:
                need(f'Using existing {role} pack: {pack["path"]}' in text, f'qualified {role} pack reuse log absent')
    return {'valid': not failures, 'failures': failures, **evidence}


def summarize(rows):
    """Only complete validated fresh pairs yield ratios; retain every process row."""
    valid = [row for row in rows if row.get('valid')]
    pairs = []
    for engine, candidate in [('omlx', 'mtp'), ('lithos', 'dspark')]:
        for pair in (1, 2, 3):
            found = {row['arm']: row for row in valid if row['pair'] == pair}
            off, on = found.get(engine + '-none'), found.get(engine + '-' + candidate)
            pairs.append({'engine': engine, 'pair': pair, 'valid': bool(off and on),
                'throughput_speedup': on['median_output_tok_s'] / off['median_output_tok_s'] if off and on else None,
                'TTFT_speedup': off['median_TTFT_s'] / on['median_TTFT_s'] if off and on else None,
                'wall_speedup': off['median_wall_s'] / on['median_wall_s'] if off and on else None})
    aggregates = {}
    for engine in ('omlx', 'lithos'):
        points = [p['throughput_speedup'] for p in pairs if p['engine'] == engine and p['valid']]
        aggregates[engine] = {'three_pairs_complete': len(points) == 3,
            'paired_throughput_geomean': math.exp(statistics.mean(map(math.log, points))) if len(points) == 3 else None,
            'every_pair_faster': all(p > 1 for p in points) if len(points) == 3 else None,
            'all_paired_points': points}
    return {'rows': rows, 'pairs': pairs, 'descriptive_aggregates': aggregates,
            'scope': 'C1 19->32 only. Cross-framework performance may be compared with engine-specific output contracts explicitly labelled; no shared-oracle quality equivalence or adoption threshold.'}
