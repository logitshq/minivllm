"""E017 native Lithos engine qualification; all runtime imports follow CLI parsing.

Uses official serving.prepare and load_session without modifying their routing,
quantization or EOS rules. This observer saves timings only as raw diagnostics;
hashing, source serialization and pack creation make this unsuitable for timing.
"""
import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import traceback
from unittest.mock import patch

from compare import validate_arm

EXPECTED_HEAD = '223658090ae549e59220bdec459957cbdad7926a'
EXPECTED_DRAFT = 'b169bc4c8dc805a9e6178fa82f6a7b84b169a457'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def identity(root):
    root = Path(root).resolve()
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix in ('.json', '.safetensors', '.jinja'))
    assert any(path.suffix == '.safetensors' for path in paths)
    return {'path': str(root), 'files': {path.name: {'bytes': path.stat().st_size, 'sha256': digest(path)} for path in paths}}


def source_pins(root):
    files = {}
    for folder in ('monolith', 'kernels', 'runtime'):
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.json', '.metal', '.h', '.hpp', '.mm', '.cpp', '.so', '.dylib'):
                files[str(path.resolve())] = digest(path)
    assert any(path.endswith(('.so', '.dylib')) for path in files), 'native binary must be pinned before runtime import'
    return files


def pack_record(directory, output, label):
    if directory is None:
        return None
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    write(output / f'{label}-pack-manifest.json', manifest)
    payload = directory / manifest['pack']
    return {'path': str(directory.resolve()), 'manifest_sha256': digest(directory / 'manifest.json'),
            'weights_sha256': digest(payload), 'bytes': payload.stat().st_size,
            'quantize': manifest.get('quantize'), 'capacity': manifest['options']['max_context']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('none', 'dspark'), required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--drafter', type=Path, required=True)
    parser.add_argument('--download-receipt', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--pack-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'attempt.json', {'mode': args.mode, 'pid': os.getpid(), 'stage': 'source_freeze_before_runtime_import'})
    session = assets = None
    programs, requests = [], []
    try:
        root = args.source_root.resolve()
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        assert head == EXPECTED_HEAD, head
        workload = json.loads(args.workload.read_text())
        assert len(workload['prompt_ids']) == 19
        target, drafter = Path(workload['model_dir']).resolve(), args.drafter.resolve()
        assert drafter.name == EXPECTED_DRAFT
        target_identity, drafter_identity = identity(target), identity(drafter)
        receipt = json.loads(args.download_receipt.read_text())
        assert receipt['status'] == 'PASS_PINNED_DOWNLOAD_ONLY' and Path(receipt['snapshot']).resolve() == drafter
        assert len(receipt['shards']) == 3
        for shard in receipt['shards']:
            assert drafter_identity['files'][shard['name']] == {'bytes': shard['bytes'], 'sha256': shard['sha256']}
        pins = source_pins(root)
        harness = {str(Path(__file__).parent / name): digest(Path(__file__).parent / name) for name in ('qualify.py', 'compare.py', 'test_compare.py')}
        plan = {'mode': args.mode, 'pid': os.getpid(), 'python': sys.executable, 'lithos_head': head,
                'source_sha256': pins, 'harness_sha256': harness, 'target_identity': target_identity,
                'drafter_identity': drafter_identity, 'workload_sha256': digest(args.workload),
                'download_receipt_sha256': digest(args.download_receipt), 'prompt_ids': workload['prompt_ids'],
                'environment': {key: value for key, value in os.environ.items() if key.startswith(('MONOLITH_', 'LITHOS_', 'METAL_', 'MLX_'))},
                'scope': '3 serial 19->32 greedy requests, normal EOS, native M5 Ultra backend; qualification, not performance/full-state correctness'}
        write(args.out / 'plan.json', plan)
        sys.path.insert(0, str(root))
        import monolith
        from monolith.runtime import _native
        from monolith.generate import Session, load_session
        from monolith.serving.setup import prepare
        assert Path(monolith.__file__).resolve().is_relative_to(root)
        assert Path(_native.__file__).resolve().as_posix() in pins
        device = _native.Device().info()
        device_record = {name: getattr(device, name) for name in ('name', 'gpu_cores', 'apple_family', 'recommended_working_set')}
        assert device.name == 'Apple M5 Ultra' and device.gpu_cores == 80
        mode_pack_root = args.pack_root.resolve() / args.mode
        mode_pack_root.mkdir(parents=True, exist_ok=False)
        config = argparse.Namespace(model=str(target), revision=None, download_dir=None, local_files_only=True,
            draft=str(drafter) if args.mode == 'dspark' else None, no_draft=args.mode == 'none',
            draft_revision=EXPECTED_DRAFT, max_context=2048, draft_block_size=7 if args.mode == 'dspark' else None,
            draft_quantization='none', draft_pack=None, pack=str(mode_pack_root),
            kernel_config=None, kernel_config_key=None, draft_sampling='argmax')
        write(args.out / 'prepare-options.json', vars(config))
        assets = prepare(config)
        recipe_key, options = assets.options(19)
        assert assets.profile.backend == 'm5_ultra_80c' and not assets.recipes and recipe_key is None
        assert assets.capacity == (2054 if args.mode == 'dspark' else 2048)
        assert bool(assets.draft_dir) is (args.mode == 'dspark')
        packs = {'target': pack_record(assets.pack_dir, args.out, 'target'),
                 'draft': pack_record(assets.draft_pack, args.out, 'draft')}
        write(args.out / 'packs.json', packs)
        original_compile = Session._compile
        def observe_compile(owner, bound, *, dynamic, prefill):
            program = original_compile(owner, bound, dynamic=dynamic, prefill=prefill)
            number = len(programs)
            path = args.out / f'program-{number}.json'
            program.save(path)
            record = {'index': number, 'bound': bound, 'dynamic': dynamic, 'prefill': prefill,
                'backend': program.backend_id, 'context_capacity': program.context_capacity,
                'program_sha256': digest(path), 'config_digest': program.config_digest, 'op_count': len(program.ops),
                'draft_ops': sum('draft.' in op.name for op in program.ops),
                'kernel_functions': sorted({kernel.function for kernel in program.kernels.values()}),
                'kernels': {key: {'source_sha256': hashlib.sha256(kernel.source.encode()).hexdigest(),
                                  'function': kernel.function, 'macros': kernel.macros, 'language_version': kernel.language_version}
                            for key, kernel in program.kernels.items()}}
            programs.append(record)
            write(args.out / 'programs.json', programs)
            return program
        with patch.object(Session, '_compile', observe_compile):
            session = load_session(str(assets.model_dir), str(assets.pack_dir), **options,
                temperature=0.0, top_p=1.0, autotune=False, measured_choices=True,
                prefill_chunk_size=128, prefix_cache=False, prefill_exact=True)
            route = {'device': device_record, 'backend': session.profile.backend, 'profile': dataclasses.asdict(session.profile),
                'recipes': assets.recipes, 'recipe_key': recipe_key, 'capacity': assets.capacity,
                'has_drafter': session.drafter is not None, 'drafter_class': type(session.drafter).__module__ + '.' + type(session.drafter).__qualname__ if session.drafter is not None else None,
                'gamma': session.drafter.gamma if session.drafter else 0, 'verify': session.verify, 'verify_length': session.verify_length,
                'prefix_cache': session.prefix_cache is not None, 'prefill_chunk_size': session.prefill_chunk_size,
                'requested_prefill_exact': True, 'effective_prefill_exact': session.prefill_exact,
                'eos': session.eos, 'normal_eos': True, 'decode_t_max': session.decode_t_max}
            write(args.out / 'route.json', route)
            eos_ids = [session.eos] if isinstance(session.eos, int) else list(session.eos)
            expected_eos = session.model.config.eos_token_id
            assert session.eos == expected_eos, 'EOS must come unchanged from model config'
            for index in range(3):
                generation = session.generate(list(workload['prompt_ids']), 32)
                state = session._last_engine.state()
                words = None
                if args.mode == 'dspark':
                    from monolith.compiler.emit import ACCEPT_LOG, CONF_LOG
                    raw = session._last_engine.read(ACCEPT_LOG)
                    (args.out / f'accept-log-{index}.bin').write_bytes(raw)
                    (args.out / f'confidence-log-{index}.bin').write_bytes(session._last_engine.read(CONF_LOG))
                    all_words = struct.unpack('<' + 'I' * (len(raw) // 4), raw)
                    words = [word for i, word in enumerate(all_words[:int(state['step'])])
                             if i >= len(generation.prefill_timings) and (word & 0xFFFF) != 0xFFFF]
                row = {'index': index, 'prompt_ids': workload['prompt_ids'], 'generation': dataclasses.asdict(generation),
                       'step_state': state, 'accept_log_rows': words,
                       'eos_in_returned_prefix': any(token in eos_ids for token in generation.tokens[:-1]),
                       'live_programs': [{'key': str(key), 'backend': engine.program.backend_id,
                                          'op_count': len(engine.program.ops)} for key, engine in session.engines.items()]}
                requests.append(row)
                write(args.out / f'request-{index}.json', row)
                print('REQUEST', args.mode, index, 'tokens', len(generation.tokens), 'steps', generation.steps, flush=True)
        # Rehash the real checkpoint payloads and runtime source after execution.
        assert identity(target) == target_identity and identity(drafter) == drafter_identity
        assert all(digest(path) == expected for path, expected in pins.items())
        assert all(digest(path) == expected for path, expected in harness.items())
        for pack in packs.values():
            if pack is not None:
                assert digest(Path(pack['path']) / 'manifest.json') == pack['manifest_sha256']
                assert digest(Path(pack['path']) / 'weights.pack') == pack['weights_sha256']
        result = {'status': 'PASS_ARM_REQUIRES_FRESH_CONTROL', 'mode': args.mode, 'route': route,
                  'packs': packs, 'programs': programs, 'requests': requests, 'pins_after_match': True,
                  'scope': plan['scope']}
        validate_arm(plan, result)
        write(args.out / 'result.json', result)
    except BaseException as exc:
        write(args.out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
              'requests_saved': len(requests), 'programs_saved': len(programs)})
        raise
    finally:
        if session is not None:
            try:
                session.release_engines()
                cleanup = {'released_engines': not session.engines, 'buffers_cleared': session.buffers is None,
                           'last_engine_cleared': session._last_engine is None, 'owner_reap_required_for_process_release': True}
                session._pipelines.clear()
                session._programs.clear()
                session = None
                assets = None
                gc.collect()
                write(args.out / 'cleanup.json', cleanup)
                assert cleanup['released_engines'] and cleanup['buffers_cleared'] and cleanup['last_engine_cleared']
            except BaseException:
                write(args.out / 'cleanup-failure.json', {'traceback': traceback.format_exc()})
                raise


if __name__ == '__main__':
    main()
