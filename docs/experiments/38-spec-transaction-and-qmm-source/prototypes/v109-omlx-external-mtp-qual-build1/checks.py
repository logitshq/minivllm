"""Pure receipt checks; no vendor/numeric imports."""
def inspect_observations(rows, mode, requests, config):
    terminals = [x for x in rows if x['kind'] == 'terminal']
    assert len(terminals) == len(requests) == 3
    assert len({x['request_id'] for x in terminals}) == 3
    assert [x['kind'] for x in rows].count('observer_installed') == 1
    result = []
    for terminal, request in zip(terminals, requests):
        assert terminal['finish_reason'] == 'length' and terminal['error'] is None
        assert terminal['prompt_tokens'] == 19 and terminal['completion_tokens'] == 32
        assert terminal['cached_tokens'] == 0
        assert len(terminal['raw_output_ids']) == 32
        assert terminal['raw_output_ids'] == request['visible_roundtrip_ids']
        assert not set(terminal['raw_output_ids']) & set(config['eos_token_ids'])
        rid = terminal['request_id']
        related = [x for x in rows if x.get('request_id') == rid]
        route = [x for x in related if x['kind'] == 'route']
        verifies = [x for x in related if x['kind'] == 'verify']
        rounds = [x for x in related if x['kind'] == 'round']
        stats = [x for x in related if x['kind'] == 'stats']
        if mode == 'none':
            assert not route and not verifies and not rounds and not stats
            result.append({'request_id': rid, 'raw_output_ids': terminal['raw_output_ids'], 'verify_calls': 0})
            continue
        assert len(route) == len(stats) == 1 and route[0]['routed']
        assert route[0]['prompt_ids'] == config['prompt_ids'] and route[0]['max_tokens'] == 32
        assert route[0]['sampler_temperature'] == 0 and route[0]['rollback_capability'] is True
        assert not set(terminal['raw_output_ids']) & set(route[0]['stop_ids'])
        assert route[0]['scheduler_drafter_source'] == config['draft_model_dir']
        assert verifies and len(verifies) == len(rounds) == len(stats[0]['accept_lens']) == len(stats[0]['draft_lens'])
        assert [x['drafted'] for x in rounds] == stats[0]['draft_lens']
        assert [x['accepted'] for x in rounds] == stats[0]['accept_lens']
        assert [x['actual_draft_token_count'] for x in verifies] == stats[0]['draft_lens']
        assert all(x['input_shape'] == [1, x['actual_draft_token_count'] + 1] for x in verifies)
        assert all(0 <= x['accepted'] <= x['drafted'] for x in rounds)
        assert sum(x['drafted'] for x in rounds) > 0
        assert stats[0]['finish_reason'] == 'length' and stats[0]['emitted'] == 32
        result.append({'request_id': rid, 'raw_output_ids': terminal['raw_output_ids'],
                       'verify_calls': len(verifies), 'drafted': sum(x['drafted'] for x in rounds),
                       'accepted': sum(x['accepted'] for x in rounds),
                       'round_drafts': stats[0]['draft_lens'], 'round_accepts': stats[0]['accept_lens']})
    loads = [x for x in rows if x['kind'] == 'load']
    attach = [x for x in rows if x['kind'] == 'attach']
    if mode == 'none':
        assert not loads and not attach
    else:
        assert len(loads) == len(attach) == 1
        load, attached = loads[0], attach[0]
        assert load['loaded'] and load['source_path'] == config['draft_model_dir'] and load['draft_kind'] == 'mtp'
        assert attached['engine_same_object'] and attached['drafter_source'] == config['draft_model_dir']
        flags = attached['flags']
        assert flags['vlm_mtp_enabled'] and flags['vlm_mtp_draft_model'] == config['draft_model_dir']
        assert flags['vlm_mtp_draft_block_size'] is None
        assert flags['mtp_enabled'] is False and flags['dflash_enabled'] is False and flags['specprefill_enabled'] is False
        assert load['block_size'] == config['draft_config_block_size']
    return result


def equal_streams(control, mtp):
    assert len(control) == len(mtp) == 3
    canonical = control[0]['raw_output_ids']
    assert all(x['raw_output_ids'] == canonical for x in control), 'ordinary target nondeterminism'
    assert all(x['raw_output_ids'] == canonical for x in mtp), 'speculative raw32 differs from fresh ordinary target'
    return canonical
