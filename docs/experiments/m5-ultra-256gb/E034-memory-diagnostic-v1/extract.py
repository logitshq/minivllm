"""Read retained metadata only; no MLX import, model load or service action."""
import hashlib, json, re
from pathlib import Path
ROOT = Path('/Users/ryanli/code/minivllm')
OWNER = ROOT / '.artifacts/m5-ultra-256gb/glm-flash-v3-owner-v1'
OUT = Path(__file__).parent
pins = {}
def read(path):
    raw = path.read_bytes()
    pins[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return raw.decode()
def parse_vm(raw):
    match = re.search(r'page size of (\d+) bytes', raw)
    assert match and int(match[1]) == 16384
    fields = ('Pages free', 'Pages active', 'Pages inactive', 'Pages wired down',
              'Pages stored in compressor', 'Pages occupied by compressor',
              'File-backed pages', 'Anonymous pages', 'Swapouts', 'Swapins')
    return {'page_size': int(match[1]), 'pages': {
        k: int(re.search(r'^' + re.escape(k) + r':\s*(\d+)\.', raw, re.M)[1]) for k in fields}}
events = [json.loads(x) for x in read(OWNER/'phases/glm-official-load/probe/events.jsonl').splitlines()]
start = next(e for e in events if e.get('phase') == 'official_vlm_load:before')
monitor = json.loads(read(OWNER/'monitor/monitor.json'))
failed = [r for r in monitor['lanes']['physical']['samples'] if r['status'] != 'COMPLETE']
assert len(failed) == 1 and failed[0]['index'] == 31 and failed[0]['error']['message'] == 'runtime swap growth'
paths = [OWNER/'cold-0/sample.json', OWNER/'cold-1/sample.json'] + sorted(OWNER.glob('physical-*/sample.json'))
assert len(paths) == 34
rows = []
for path in paths:
    s = json.loads(read(path)); assert s['status'] == 'MEASURED'
    vmraw = next(c['stdout'] for c in s['commands'] if c['command'] == ['/usr/bin/vm_stat'])
    vm = parse_vm(vmraw); p = s['parsed']
    assert vm['pages']['Swapouts'] == p['swapouts']
    rows.append({'observation': path.parent.name, 'source': str(path), 'pid': s['pid'],
        'sample_status': s['status'], 'guard_status': 'REJECTED' if path.parent.name == 'physical-0031' else 'ACCEPTED',
        'started_monotonic': s['started_monotonic'], 'ended_monotonic': s['ended_monotonic'],
        'elapsed_start_s': s['started_monotonic'] - start['monotonic'],
        'elapsed_end_s': s['ended_monotonic'] - start['monotonic'],
        'sample_seconds': s['ended_monotonic'] - s['started_monotonic'],
        'child_footprint_bytes': None if path.parent.name.startswith('cold-') else p['current_bytes'],
        'system_footprint_bytes': p['system_bytes'], 'pressure': p['pressure'],
        'swap_used_bytes': p['swap_bytes'], 'swapouts': p['swapouts'], 'vm_stat': vm})
base = next(r for r in rows if r['observation'] == 'cold-1')
for r in rows:
    r['swap_growth_bytes'] = r['swap_used_bytes'] - base['swap_used_bytes']
    r['swapouts_growth_pages'] = r['swapouts'] - base['swapouts']
data = {'schema': 'E034-raw-memory-diagnostic-v1', 'qualification': 'DIAGNOSTIC_ONLY_FAILED_LOAD',
    'origin': {'event': 'official_vlm_load:before', 'monotonic': start['monotonic'], 'pid': start['pid']},
    'guard_rejection': {'monotonic': failed[0]['ended_monotonic'],
        'elapsed_s': failed[0]['ended_monotonic'] - start['monotonic'], 'error': failed[0]['error'],
        'source': str(OWNER/'monitor/monitor.json'), 'failed_sample_retained': True},
    'baseline': {'observation': 'cold-1', 'swap_used_bytes': base['swap_used_bytes'], 'swapouts': base['swapouts']},
    'rows': rows, 'source_pins': pins,
    'scope': ['34 actual observations including failed last guard observation; no post-abort physical samples.',
        'Points use measurement completion time; commands within each sample were sequential, not simultaneous.',
        'Cold samples target the owner, not the future child, so child footprint is null.',
        'Footprint fields are accounting observations; not added to each other or vm_stat categories.',
        'Compressor occupied is physical pages; stored is logical pages represented by the compressor.',
        'Swap/Swapouts are system-global and cannot identify which process caused them.',
        'No safe-fit, correctness, or inference performance qualification.']}
with (OUT/'observations.json').open('x') as f: json.dump(data, f, indent=2); f.write('\n')
print(json.dumps({'status': 'EXTRACTED', 'observations': len(rows), 'sources': len(pins), 'failed_last_sample': rows[-1]['guard_status']}))
