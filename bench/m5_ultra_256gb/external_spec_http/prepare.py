"""E020 CPU-only plan: no model imports, server, GPU, weight download or repacking.

Inherits already qualified model/pack digests. Weights are fully rehashed by the
scheduled runner before and after the matrix, outside every request timer.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from contract import (BASE, CONFIGS, ENV, HERE, LITHOS, LITHOS_HEAD, QUALIFICATIONS,
                      REPO, USER_TEXT, WORKLOAD, WORKLOAD_SHA, digest, read, write)
from launch import specification


def runtime_files(python, names):
    # importlib.metadata reads distribution metadata; never imports the packages.
    code = """import importlib.metadata as m,json,sys
from pathlib import Path
out={'versions':{},'files':[]}
for name in sys.argv[1:]:
 d=m.distribution(name);out['versions'][name]=d.version
 out['files'] += [str(Path(d.locate_file(p)).resolve()) for p in d.files or []
                 if str(p).endswith(('.py','.so','.dylib','.metallib','METADATA')) and Path(d.locate_file(p)).is_file()]
print(json.dumps(out))"""
    import json
    return json.loads(subprocess.check_output([str(python), '-B', '-c', code, *names], text=True))


def prepare(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'attempt.json', {'pid': os.getpid(), 'stage': 'CPU preparation only',
                                'no_runtime_or_model_import': True})
    assert digest(WORKLOAD) == WORKLOAD_SHA
    workload = read(WORKLOAD)
    assert len(workload['prompt_ids']) == 19 and len(workload['continuation_ids']) == 32
    plans, evidence, packs, model_settings = {}, {}, {}, {}
    pins = {}
    def pin(path, expected=None):
        path = Path(path).absolute()
        actual = expected or digest(path)
        if str(path) in pins:
            assert pins[str(path)] == actual, f'conflicting inherited SHA: {path}'
        pins[str(path)] = actual
    def model(record):
        for name, info in record['files'].items():
            path = Path(record['path']) / name
            size = info.get('bytes', info.get('size'))
            assert path.stat().st_size == size
            pin(path, info['sha256'])
    for engine, name, status in [
        ('omlx', 'official-mtp-comparison-v1.json', 'PASS_OFFICIAL_MTP_ENGINE_IDS_AND_ACTUAL_ROUTE'),
        ('lithos', 'lithos-dspark-comparison-v1.json', 'PASS_LITHOS_DSPARK_ROUTE_AND_FRESH_CONTROL_IDS')]:
        path = BASE / name
        comparison = read(path)
        assert comparison['status'] == status and comparison['raw_ids_compared'] == 192
        for receipt, sha in comparison['receipts'].items():
            assert digest(REPO / receipt) == sha
        evidence[engine] = comparison
        pin(path)
        shutil.copyfile(path, out / name)
    for arm, source in QUALIFICATIONS.items():
        plan = plans[arm] = read(source / 'plan.json')
        assert plan['workload_sha256'] == WORKLOAD_SHA
        for name in ('plan.json', 'result.json', 'request-0.json', 'request-1.json', 'request-2.json'):
            pin(source / name)
        if arm.startswith('omlx-'):
            model(plan['target'])
            model(plan['drafter'])
            for path, sha in plan['distribution_identity']['files'].items():
                pin(path, sha)
            model_settings[arm] = read(source / 'settings.json')['model']
            pin(source / 'settings.json')
            assert model_settings[arm]['vlm_mtp_enabled'] == (arm == 'omlx-mtp')
            assert not model_settings[arm]['mtp_enabled'] and not model_settings[arm]['dflash_enabled']
            assert model_settings[arm].get('vlm_mtp_draft_block_size') is None
            for i in range(3):
                raw = read(source / f'request-{i}.json')['raw']
                assert raw['raw_ids'] == evidence['omlx']['raw_ids'] and raw['prompt_ids'] == workload['prompt_ids']
        else:
            assert plan['lithos_head'] == LITHOS_HEAD
            model(plan['target_identity'])
            model(plan['drafter_identity'])
            for path, sha in plan['source_sha256'].items():
                pin(path, sha)
            packs[arm] = read(source / 'packs.json')
            pin(source / 'packs.json')
            for pack in packs[arm].values():
                if pack is None:
                    continue
                manifest = Path(pack['path']) / 'manifest.json'
                assert digest(manifest) == pack['manifest_sha256']
                assert pack['quantize'] is None and pack['capacity'] == (2054 if arm == 'lithos-dspark' else 2048)
                payload = manifest.parent / read(manifest)['pack']
                assert payload.stat().st_size == pack['bytes']
                pin(manifest, pack['manifest_sha256'])
                pin(payload, pack['weights_sha256'])
            for i in range(3):
                row = read(source / f'request-{i}.json')
                assert row['generation']['tokens'] == evidence['lithos']['raw_ids']
                assert row['prompt_ids'] == workload['prompt_ids']
    assert subprocess.check_output(['git', '-C', str(LITHOS), 'rev-parse', 'HEAD'], text=True).strip() == LITHOS_HEAD
    assert not subprocess.check_output(['git', '-C', str(LITHOS), 'status', '--porcelain'], text=True).strip()
    # Standalone native snapshot prevents another agent's later edits changing the anchor.
    native_root = out / 'frozen-native'
    source = REPO / 'tutorial_code/miniengine'
    for path in source.rglob('*.py'):
        destination = native_root / 'tutorial_code/miniengine' / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        pin(destination)
    manifest = out / 'native-source.json'
    write(manifest, {'files': {str(p.relative_to(source)): digest(p) for p in source.rglob('*.py')}})
    pin(manifest)
    # Pin HTTP/tokenizer dependencies as well as the qualified numerical runtimes.
    runtime = {}
    for name, python, names in [
        ('controller-native', REPO / '.venv/bin/python', ['mlx', 'mlx-metal', 'httpx', 'httpcore', 'tokenizers']),
        ('omlx-http', REPO / '.artifacts/omlx-baseline/.venv/bin/python',
         ['fastapi', 'starlette', 'uvicorn', 'pydantic', 'transformers', 'tokenizers']),
        ('lithos-http', LITHOS / '.venv/bin/python',
         ['fastapi', 'starlette', 'uvicorn', 'pydantic', 'transformers', 'tokenizers'])]:
        info = runtime_files(python, names)
        runtime[name] = {'python': str(python), 'versions': info['versions']}
        for path in info['files']:
            # Existing qualified hashes take precedence and are verified at runtime.
            if path not in pins:
                pin(path)
        pin(python.resolve())
    pin(REPO / '.artifacts/omlx-baseline/.venv/bin/omlx')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(workload['model_dir']) / 'tokenizer.json'))
    prompt = tok.decode(workload['prompt_ids'], skip_special_tokens=False)
    assert tok.encode(prompt, add_special_tokens=False).ids == workload['prompt_ids']
    oracle = {'native': workload['continuation_ids'], **{k: v['raw_ids'] for k, v in evidence.items()}}
    for ids in oracle.values():
        assert tok.encode(tok.decode(ids, skip_special_tokens=True), add_special_tokens=False).ids == ids
    # Same official template settings as Lithos Backend.complete; CPU tokenizer only.
    code = """import json,sys
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=False)
print(json.dumps(t.apply_chat_template([{'role':'user','content':sys.argv[2]}],tokenize=True,
    return_dict=False,add_generation_prompt=True,enable_thinking=False)))"""
    import json
    env = dict(os.environ, HF_HUB_OFFLINE='1', USE_TORCH='0', USE_TF='0', USE_FLAX='0')
    chat_ids = json.loads(subprocess.check_output([str(LITHOS / '.venv/bin/python'), '-B', '-c', code,
                                                 workload['model_dir'], USER_TEXT], text=True, env=env))
    assert chat_ids == workload['prompt_ids'], 'official Lithos chat template does not match the 19-ID workload'
    for path in HERE.glob('*.py'):
        pin(path)
    for name in ('resources_v2.py', 'resources.py', 'resource_base.py', 'suspended_service.py'):
        pin(HERE.parent / name)
    pin(HERE.parent / 'lithos/client.py')
    pin(WORKLOAD)
    # Freeze small source/config files for later auditing before any GPU execution.
    sources = out / 'frozen-sources'
    for filename, sha in pins.items():
        path = Path(filename)
        if path.suffix in ('.py', '.json', '.metal', '.h', '.hpp', '.cpp', '.mm') and path.stat().st_size < 4 << 20:
            assert digest(path) == sha, f'small source changed since qualification: {path}'
            destination = sources / str(path).lstrip('/')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    plan = {'schema': 'E020-v1', 'status': 'PREPARED_NOT_RUN', 'configs': CONFIGS,
        'workload': workload, 'prompt_text': prompt, 'user_text': USER_TEXT, 'oracle_ids': oracle,
        'normal_eos': True, 'concurrency': 1, 'warm_requests': 3, 'formal_requests': 3,
        'expected_counts': {'fresh_servers': 15, 'warm_requests': 45, 'formal_requests': 45,
                            'formal_output_tokens_if_valid': 1440},
        'environment': ENV, 'runtime_versions': runtime, 'qualified_model_settings': model_settings,
        'drafters': {'omlx': plans['omlx-mtp']['drafter']['path'],
                     'lithos': plans['lithos-dspark']['drafter_identity']['path']}, 'packs': packs,
        'native_source_root': str(native_root), 'native_source_manifest': str(manifest),
        'native_configuration': {'compile_decode': True, 'fuse_decode': True, 'cache_limit_gib': 2,
            'drafter': None, 'gates_compile': True, 'norm_reuse': False,
            'grouped_prefill': 'unmodified adopted default; C1 is ineligible because group length < 2'},
        'pins': pins, 'pin_stats': {p: {'bytes': Path(p).stat().st_size, 'mtime_ns': Path(p).stat().st_mtime_ns}
                                  for p in pins},
        'pin_policy': 'Full SHA before and after all servers; every arm rechecks small sources and all file stats. No hash in request timers.',
        'source_roots': {'lithos': str(LITHOS), 'native_frozen': str(native_root)},
        'lithos_head': LITHOS_HEAD, 'authorized_suspended_service_pid': 74384,
        'cooldown_seconds_after_each_request': 2,
        'resource_policy': 'resources_v2 guard plus AC/nominal-or-fair/known direct server PPID; all swap counters retained, no zero-paging claim',
        'failures': 'No retries or outlier deletion. Contract failures retained through all 6 requests; safety/transport/lifecycle failures abort matrix.',
        'scope': 'Fresh within-engine C1 none/spec pairs plus current native anchor. Visible text matches previous qualified engine IDs; external live raw IDs unavailable. Cross-framework latency/throughput comparison is allowed with these distinct output contracts, not shared-oracle quality equivalence.',
        'decision': 'Descriptive paired ratios only. Three independent process medians and all 45 formal rows retained; no adoption or global fastest claim.'}
    write(out / 'plan.json', plan)
    previews = []
    for config in CONFIGS:
        previews.append({**config, **specification(plan, config['arm'], Path('<new-output>') /
                            f'pair{config["pair"]}-{config["arm"]}', '<ephemeral-port>', '<ephemeral-key>')})
    write(out / 'launch-preview.json', previews)
    write(out / 'preparation.json', {'status': 'PASS_CPU_PREPARED_NOT_RUN', 'plan_sha256': digest(out / 'plan.json'),
                                   'pins': len(pins), 'large_payload_hashes_deferred_to_scheduled_run': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    existed_before = args.out.exists()
    try:
        prepare(args.out)
    except BaseException:
        import traceback
        if not existed_before and args.out.is_dir():
            write(args.out / 'failure.json', {'traceback': traceback.format_exc(), 'no_GPU_started': True})
        raise


if __name__ == '__main__':
    main()
