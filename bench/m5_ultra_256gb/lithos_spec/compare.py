"""CPU-only E017 route/receipt validation and fresh-process token comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def validate_request(row, mode):
    generation, state = row['generation'], row['step_state']
    assert len(row['prompt_ids']) == 19 and len(generation['tokens']) == 32
    assert generation['cached_prompt_tokens'] == 0 and generation['decode_tokens'] == 31
    assert sum(item['tokens'] for item in generation['prefill_timings']) == 19
    assert state['error'] == 0 and state['done'] == 1 and state['stop_at'] == 32
    assert not row['eos_in_returned_prefix'], 'normal EOS ended before the qualified 32-token contract'
    if mode == 'none':
        assert all(generation[key] is None for key in ('accepted', 'committed', 'verify_len'))
        assert row['accept_log_rows'] is None
        assert state['ring_head'] == 32 and generation['steps'] >= 31
        return
    accepted, committed, verified = [generation[key] for key in ('accepted', 'committed', 'verify_len')]
    assert len(accepted) == len(committed) == len(verified) == generation['steps'] > 0
    assert all(type(v) is int for sequence in (accepted, committed, verified) for v in sequence)
    assert all(v == 7 for v in verified), 'fixed gamma 7 must actually verify seven drafts at this short context'
    assert all(0 <= a <= v and 1 <= c <= a + 1 for a, c, v in zip(accepted, committed, verified))
    assert all(c == a + 1 for a, c in zip(accepted[:-1], committed[:-1]))
    assert sum(accepted) > 0, 'no useful speculation observed'
    # accept_scan commits the complete final block, then sets done. Generation
    # truncates the returned prefix; accepted/committed retain actual GPU work.
    assert sum(committed[:-1]) < 31 <= sum(committed) <= 38
    assert state['ring_head'] == 1 + sum(committed)
    assert state['step'] == len(generation['prefill_timings']) + generation['steps']
    words = row['accept_log_rows']
    assert accepted == [v & 255 for v in words]
    assert verified == [(v >> 8) & 255 for v in words]
    assert committed == [v >> 16 for v in words]


def validate_arm(plan, result):
    mode = result['mode']
    assert mode in ('none', 'dspark') and plan['mode'] == mode
    assert result['status'] == 'PASS_ARM_REQUIRES_FRESH_CONTROL'
    route = result['route']
    assert route['device']['name'] == 'Apple M5 Ultra' and route['device']['gpu_cores'] == 80
    assert route['backend'] == 'm5_ultra_80c' and not route['recipes'] and route['recipe_key'] is None
    assert route['capacity'] == (2054 if mode == 'dspark' else 2048)
    assert route['prefix_cache'] is False and route['prefill_chunk_size'] == 128
    assert route['requested_prefill_exact'] is True
    assert route['effective_prefill_exact'] is False, '128 rows are already the exact reference size'
    assert route['gamma'] == (7 if mode == 'dspark' else 0)
    assert route['has_drafter'] is (mode == 'dspark')
    if mode == 'dspark':
        assert route['verify'] == 'fixed'
    assert result['packs']['target']['quantize'] is None
    if mode == 'dspark':
        assert result['packs']['draft']['quantize'] is None
        assert route['drafter_class'] == 'monolith.spec.dspark.model.DSparkDrafter'
        assert route['verify_length'] == 7
    else:
        assert result['packs']['draft'] is None
    decodes = [item for item in result['programs'] if not item['prefill']]
    assert decodes and all(item['backend'] == 'm5_ultra_80c' for item in result['programs'])
    for program in decodes:
        assert program['bound'] == (8 if mode == 'dspark' else 1)
        assert program['dynamic'] is (mode == 'dspark')
        functions = set(program['kernel_functions'])
        if mode == 'dspark':
            assert {'accept_scan', 'verify_select'} <= functions and program['draft_ops'] > 0
        else:
            assert 'accept_scan' not in functions and program['draft_ops'] == 0
        assert program['op_count'] > 0 and len(program['program_sha256']) == 64
    requests = result['requests']
    assert len(requests) == 3 and [row['index'] for row in requests] == [0, 1, 2]
    for row in requests:
        assert row['prompt_ids'] == plan['prompt_ids']
        validate_request(row, mode)
    assert all(row['generation']['tokens'] == requests[0]['generation']['tokens'] for row in requests)


def compare(control, candidate):
    plans = [json.loads((path / 'plan.json').read_text()) for path in (control, candidate)]
    results = [json.loads((path / 'result.json').read_text()) for path in (control, candidate)]
    assert [result['mode'] for result in results] == ['none', 'dspark']
    assert plans[0]['pid'] != plans[1]['pid'], 'fresh processes required'
    for key in ('target_identity', 'drafter_identity', 'source_sha256', 'harness_sha256', 'environment', 'prompt_ids', 'workload_sha256', 'lithos_head'):
        assert plans[0][key] == plans[1][key], 'identity mismatch: ' + key
    for plan, result, path in zip(plans, results, (control, candidate)):
        validate_arm(plan, result)
        for program in result['programs']:
            raw = (path / f"program-{program['index']}.json").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == program['program_sha256']
        if result['mode'] == 'dspark':
            for row in result['requests']:
                raw = (path / f"accept-log-{row['index']}.bin").read_bytes()
                words = struct.unpack('<' + 'I' * (len(raw) // 4), raw)
                decoded = [word for i, word in enumerate(words[:row['step_state']['step']])
                           if i >= len(row['generation']['prefill_timings']) and (word & 0xFFFF) != 0xFFFF]
                assert decoded == row['accept_log_rows'], 'raw accept log differs from receipt'
        cleanup = json.loads((path / 'cleanup.json').read_text())
        assert cleanup['released_engines'] and cleanup['buffers_cleared'] and cleanup['last_engine_cleared']
        assert result['pins_after_match']
    ids = results[0]['requests'][0]['generation']['tokens']
    assert all(row['generation']['tokens'] == ids for result in results for row in result['requests'])
    return {'status': 'PASS_LITHOS_DSPARK_ROUTE_AND_FRESH_CONTROL_IDS', 'requests': 6, 'raw_ids_compared': 192,
            'raw_ids': ids, 'scope': '19->32 greedy engine qualification only; no full-state, HTTP or performance claim',
            'receipts': {str(path / 'result.json'): hashlib.sha256((path / 'result.json').read_bytes()).hexdigest() for path in (control, candidate)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.control, args.candidate)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(result['status'])


if __name__ == '__main__':
    main()
