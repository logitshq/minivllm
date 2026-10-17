"""Pure launch/config construction: official entrypoints, no wrapper grandchildren."""
from pathlib import Path

from contract import ENV, HERE, LITHOS, MODEL_ID, REPO


def specification(plan, arm, out, port, key):
    out = Path(out)
    target = plan['workload']['model_dir']
    environment = dict(ENV, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    files, symlinks = {}, {}
    if arm == 'native-none':
        command = [str(REPO / '.venv/bin/python'), '-B', str(HERE / 'native_server.py'),
                   '--model', target, '--port', str(port), '--package-root', plan['native_source_root'],
                   '--source-manifest', plan['native_source_manifest']]
        cwd = REPO
    elif arm.startswith('omlx-'):
        state, models = out / 'state', out / 'models'
        symlinks[str(models / MODEL_ID)] = target
        files[str(state / 'settings.json')] = {
            'server': {'burst_decode_mode': 'balanced'},
            'scheduler': {'max_concurrent_requests': 1, 'chunked_prefill': False,
                          'prefill_priority': 'speed', 'decode_fairness': True},
            'cache': {'enabled': False}}
        # Exact qualification model settings, not a partial set merged with user state.
        files[str(state / 'model_settings.json')] = {
            'version': 1, 'models': {MODEL_ID: plan['qualified_model_settings'][arm]}}
        command = [str(REPO / '.artifacts/omlx-baseline/.venv/bin/omlx'), 'serve',
                   '--model-dir', str(models), '--base-path', str(state), '--host', '127.0.0.1',
                   '--port', str(port), '--api-key', key, '--headless', '--no-hf-cache', '--no-cache',
                   '--max-concurrent-requests', '1', '--sse-keepalive-mode', 'off']
        cwd = REPO
    else:
        assert arm in ('lithos-none', 'lithos-dspark')
        packs = plan['packs'][arm]
        command = [str(LITHOS / '.venv/bin/python'), '-B', '-m', 'monolith.serve',
                   '--model', target, '--pack', packs['target']['path'], '--local-files-only',
                   '--host', '127.0.0.1', '--port', str(port), '--max-context', '2048',
                   '--prefill-chunk-size', '128-exact', '--keep-warm', '0', '--no-warmup']
        if arm == 'lithos-none':
            command += ['--no-draft']
        else:
            command += ['--draft', plan['drafters']['lithos'], '--draft-kind', 'dspark',
                        '--draft-pack', packs['draft']['path'], '--draft-quantization', 'none',
                        '--draft-block-size', '7', '--draft-sampling', 'argmax']
        environment.update(MONOLITH_ACCELERATOR='on', LITHOS_METAL_API_KEY=key)
        cwd = LITHOS
    return {'command': command, 'cwd': str(cwd), 'environment': environment,
            'files': files, 'symlinks': symlinks}
