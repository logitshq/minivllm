"""E017 read-only metadata preparation; stdlib only, never imports GPU runtimes.

Fetches small JSON/model-card metadata, not weights. Refuses existing output dirs.
No pack, model loader, Metal device, server, or benchmark is launched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
REPO_ID = 'LithosAI/Qwen3.8-27B-DSpark-NVFP4'
REVISION = 'b169bc4c8dc805a9e6178fa82f6a7b84b169a457'
LITHOS_COMMIT = '223658090ae549e59220bdec459957cbdad7926a'
TARGET_REVISION = '5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6'
SMALL_FILES = ('config.json', 'conversion.json', 'validation.json', 'model.safetensors.index.json')
SOURCE_FILES = ('monolith/serve.py', 'monolith/models/catalog.py', 'monolith/serving/setup.py',
                'monolith/serving/cache.py', 'monolith/spec/dspark/model.py', 'monolith/spec/dspark/config.py',
                'monolith/spec/dspark/weights.py', 'monolith/spec/dspark/optimization.py', 'monolith/nn/pack_plan.py',
                'monolith/models/qwen3_5/weights.py', 'monolith/backends/metal/base.py',
                'monolith/backends/metal/m5_ultra_80c/backend.py', 'monolith/backends/metal/m5_ultra_80c/config.json',
                'monolith/backends/metal/m5_max_40c/serving.py',
                'monolith/backends/metal/m5_max_40c/recipes/dspark/selected-nvfp4-endpoints.json',
                'monolith/generate.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        body = response.read(2 * 1024**2 + 1)
        if len(body) > 2 * 1024**2:
            raise ValueError('Metadata response exceeded 2 MiB; refusing further download')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lithos', type=Path, default=Path('/Users/ryanli/code/lithos-metal'))
    parser.add_argument('--workload', type=Path, default=ROOT / '.artifacts/m5-ultra-256gb/fullqual-build2/workload.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir()
    try:
        write(args.out / 'plan.json', {'experiment': 'E017 metadata only', 'repo_id': REPO_ID, 'revision': REVISION,
              'maximum_http_response_bytes': 2 * 1024**2, 'download_weight_files': False,
              'gpu_runtime_imports': False, 'pack_or_server_launch': False})
        head = subprocess.check_output(['git', '-C', str(args.lithos), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(args.lithos), 'status', '--porcelain'], text=True)
        assert head == LITHOS_COMMIT and not status, 'Lithos source identity changed; inspect before updating pins'
        api_url = f'https://huggingface.co/api/models/{REPO_ID}/revision/{REVISION}?blobs=true'
        raw_api = fetch(api_url)
        api = json.loads(raw_api)
        assert api['sha'] == REVISION
        write(args.out / 'hub-metadata.json', api)
        sources = {'hub-metadata': {'url': api_url, 'sha256': sha(raw_api)}}
        fetched = {}
        for name in SMALL_FILES:
            url = f'https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{name}'
            body = fetch(url)
            fetched[name] = json.loads(body)
            (args.out / name).write_bytes(body)
            sources[name] = {'url': url, 'bytes': len(body), 'sha256': sha(body)}
        card_url = f'https://huggingface.co/{REPO_ID}/raw/{REVISION}/README.md'
        card = fetch(card_url)
        sources['README.md'] = {'url': card_url, 'bytes': len(card), 'sha256': sha(card), 'body_saved': False}
        conversion = fetched['conversion.json']
        for name in ('config.json', 'model.safetensors.index.json'):
            assert sources[name]['sha256'] == conversion['files'][name]['sha256']
        inventory = {r['rfilename']: r for r in api['siblings']}
        shards = sorted(set(fetched['model.safetensors.index.json']['weight_map'].values()))
        shard_rows = []
        for name in shards:
            item = inventory[name]
            declared = conversion['files'][name]
            assert item['size'] == declared['bytes'] and item['lfs']['sha256'] == declared['sha256']
            shard_rows.append({'name': name, 'bytes': item['size'], 'remote_declared_sha256': item['lfs']['sha256'], 'downloaded': False})
        workload = json.loads(args.workload.read_text())
        target = Path(workload['model_dir'])
        assert target.name == TARGET_REVISION
        target_config = json.loads((target / 'config.json').read_text())
        text = target_config.get('text_config', target_config)
        draft = fetched['config.json']
        ds = draft['dspark_config']
        names = fetched['model.safetensors.index.json']['weight_map']
        static = {
            'target_architecture': target_config['architectures'],
            'hidden_size_match': text['hidden_size'] == draft['hidden_size'] == 5120,
            'vocab_size_match': text['vocab_size'] == draft['vocab_size'] == 248320,
            'target_num_layers': text['num_hidden_layers'], 'target_taps': ds['target_layer_ids'],
            'taps_in_layer_range': all(0 <= i < text['num_hidden_layers'] for i in ds['target_layer_ids']),
            'checkpoint_embedding_absent': 'embed_tokens.weight' not in names,
            'checkpoint_lm_head_absent': 'lm_head.weight' not in names,
            'markov_head_present': 'markov_head.markov_w1.weight' in names and 'markov_head.markov_w2.weight' in names,
            'confidence_head_present': 'confidence_head.proj.weight' in names,
            'default_catalog_matches_target': False,
            'scope': 'Static source/config evidence only; target binding, tokens, state and GPU routes still unqualified',
        }
        assert all(static[k] for k in ('hidden_size_match', 'vocab_size_match', 'taps_in_layer_range',
                   'checkpoint_embedding_absent', 'checkpoint_lm_head_absent', 'markov_head_present', 'confidence_head_present'))
        pack = ROOT / '.artifacts/m5-ultra-256gb/lithos-adaptation-v1/packs/target-b6042c39f78b9df33a19ac1c/manifest.json'
        old_pack = json.loads(pack.read_text())
        root = ROOT / '.artifacts/m5-ultra-256gb/lithos-spec-packs-v1'
        command = [str(args.lithos / '.venv/bin/python'), '-B', '-m', 'monolith.serve', '--model', str(target),
                   '--draft', REPO_ID, '--draft-revision', REVISION, '--draft-kind', 'dspark',
                   '--draft-quantization', 'none', '--draft-block-size', '7', '--draft-sampling', 'argmax',
                   '--max-context', '2048', '--prefill-chunk-size', '128-exact', '--pack', str(root),
                   '--local-files-only', '--host', '127.0.0.1', '--port', '18017', '--keep-warm', '0']
        result = {'status': 'PASS_STATIC_METADATA_ONLY', 'source_commit': head,
                  'source_sha256': {f: sha((args.lithos / f).read_bytes()) for f in SOURCE_FILES},
                  'harness_sha256': sha(Path(__file__).read_bytes()), 'upstream_sources': sources,
                  'target': {'path': str(target), 'revision': TARGET_REVISION,
                             'config_sha256': sha((target / 'config.json').read_bytes()),
                             'index_sha256': sha((target / 'model.safetensors.index.json').read_bytes()),
                             'workload_sha256': sha(args.workload.read_bytes()), 'weight_bytes_hashed_in_this_preparation': False},
                  'draft': {'repo': REPO_ID, 'revision': REVISION, 'shards': shard_rows,
                            'total_weight_bytes_to_download': sum(r['bytes'] for r in shard_rows),
                            'weight_bytes_downloaded': 0, 'weight_format': draft['lithos_quantization'],
                            'rope': draft['rope_parameters'], 'block_size_proposals': draft['block_size'],
                            'sliding_window': draft.get('sliding_window')},
                  'static_compatibility': static,
                  'pack': {'old_manifest_sha256': sha(pack.read_bytes()), 'old_capacity': old_pack['options']['max_context'],
                           'target_formats': sorted({s['format'] for s in old_pack['slabs']}),
                           'required_spec_capacity': 2054, 'old_pack_reusable_at_requested_capacity': False,
                           'new_cache_root': str(root)},
                  'launch_plan': {'cwd': str(args.lithos), 'command': command,
                                  'environment': {'MONOLITH_ACCELERATOR': 'on', 'HF_HUB_OFFLINE': '1'},
                                  'executed': False, 'requires_prior_pinned_weight_download': True,
                                  'external_service_lease_required': True, 'exclusive_gpu_queue_required': True},
                  'recipe_scope': 'Native M5 Ultra backend has no validated serving recipes; existing same-target pack is all NVFP4, whereas Max mixed-target selector needs NVFP4 and FP8. Do not silently substitute target or label fallback as best-tuned Lithos.',
                  'next_gate': 'Pinned weights + direct pack validation, hardware identity, actual DSpark/verify routes, raw greedy IDs and committed-state correctness before paired timing'}
        write(args.out / 'complete.json', result)
        print(json.dumps({'status': result['status'], 'revision': REVISION, 'weight_bytes_downloaded': 0,
                          'weight_bytes_to_download': result['draft']['total_weight_bytes_to_download'], 'next_gate': result['next_gate']}))
    except BaseException:
        write(args.out / 'failure.json', {'status': 'STOP', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
