"""Download the pinned E017 drafter and verify every declared weight shard."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    metadata_path = Path(__file__).parent / 'metadata-v1/complete.json'
    metadata = json.loads(metadata_path.read_text())
    draft = metadata['draft']
    plan = {'repo': draft['repo'], 'revision': draft['revision'],
            'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'shards': draft['shards'], 'gpu_work': False}
    (args.out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    try:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(repo_id=draft['repo'], revision=draft['revision'],
                                         allow_patterns=['*.json', '*.safetensors', 'README.md'], max_workers=3))
        assert snapshot.name == draft['revision']
        rows = []
        for shard in draft['shards']:
            path = snapshot / shard['name']
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert path.stat().st_size == shard['bytes']
            assert digest == shard['remote_declared_sha256'], shard['name']
            rows.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest})
        result = {'status': 'PASS_PINNED_DOWNLOAD_ONLY', 'snapshot': str(snapshot), 'shards': rows,
                  'no_model_load_or_gpu_qualification': True}
        (args.out / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    except BaseException:
        (args.out / 'failure.json').write_text(json.dumps({'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
