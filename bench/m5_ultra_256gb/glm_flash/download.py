"""Fetch the pinned GLM Flash checkpoint; no model import, conversion or GPU.

Run outside any inference benchmark. All shards are independently checked
against the publisher's LFS SHA256 values; metadata and receipts stay separate
from the large, ignored model cache. Failed attempts are preserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.artifacts/models/huggingface')
    args = parser.parse_args()
    args.out = args.out.resolve(); args.cache_dir = args.cache_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = HERE / 'metadata-v1/manifest.json'
    meta = json.loads(manifest_path.read_text())
    plan = {'repo': meta['repo'], 'revision': meta['revision'], 'weight_bytes': meta['weight_bytes'],
            'shards': meta['shards'], 'source_sha256': sha(__file__), 'metadata_sha256': sha(manifest_path),
            'cache_dir': str(args.cache_dir), 'pid': os.getpid(), 'started': time.time(),
            'max_download_workers': 3, 'GPU_work': False, 'quantization_unchanged': True}
    write(args.out / 'plan.json', plan)
    try:
        free = shutil.disk_usage(args.cache_dir).free
        required = meta['weight_bytes'] + 64 * 1024**3
        write(args.out / 'disk-before.json', {'free_bytes': free, 'minimum_free_bytes': required})
        assert free >= required, 'insufficient free space for checkpoint and staging reserve'
        os.environ['HF_XET_CACHE'] = str(args.cache_dir.parent / 'xet')
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        from huggingface_hub import snapshot_download
        print('DOWNLOAD', meta['repo'], meta['revision'], meta['weight_bytes'], flush=True)
        snapshot = Path(snapshot_download(repo_id=meta['repo'], revision=meta['revision'],
            cache_dir=str(args.cache_dir), max_workers=3,
            allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', 'README.md', 'LICENSE*']))
        assert snapshot.name == meta['revision']
        write(args.out / 'downloaded.json', {'snapshot': str(snapshot), 'time': time.time(),
                                            'not_yet_SHA_verified': True})
        records = []
        for entry in meta['shards']:
            path = snapshot / entry['name']
            assert path.stat().st_size == entry['bytes'], entry['name']
            actual = sha(path)
            assert actual == entry['sha256'], entry['name']
            records.append({**entry, 'sha256_verified': True})
            write(args.out / 'verification-progress.json', {'verified_shards': len(records), 'shards': records})
            print('VERIFIED', len(records), '/', len(meta['shards']), entry['name'], flush=True)
        for name, entry in meta['metadata'].items():
            assert (snapshot / name).stat().st_size == entry['bytes']
            assert sha(snapshot / name) == entry['sha256'], name
        index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
        assert set(index['weight_map'].values()) == {x['name'] for x in records}
        result = {'status': 'PASS_PINNED_DOWNLOAD_AND_SHA_ONLY', 'repo': meta['repo'],
                  'revision': meta['revision'], 'snapshot': str(snapshot), 'weight_bytes': meta['weight_bytes'],
                  'shards': records, 'completed': time.time(), 'no_runtime_load_correctness_or_speed_claim': True}
        write(args.out / 'complete.json', result)
        print('PASS', len(records), 'verified shards', str(snapshot), flush=True)
    except BaseException:
        write(args.out / 'failure.json', {'traceback': traceback.format_exc(), 'time': time.time(),
              'resumable_cache_preserved': True, 'GPU_work': False})
        raise


if __name__ == '__main__':
    main()
