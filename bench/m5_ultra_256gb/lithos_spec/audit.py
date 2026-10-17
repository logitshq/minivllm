"""Independent stdlib E017 receipt/Program/log audit and immutable archive."""
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import tarfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '.artifacts/m5-ultra-256gb'
OUT = ROOT / 'docs/experiments/m5-ultra-256gb/E017-lithos-dspark-v1'
LITHOS = Path('/Users/ryanli/code/lithos-metal')


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    assert not OUT.exists(), 'Immutable evidence: select a new archive version'
    files, stats, all_ids, all_resources, owner_records = {}, {}, [], [], []
    def collect(path, name=None):
        path = Path(path)
        if name is None:
            name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else 'external/lithos-metal/' + str(path.relative_to(LITHOS))
        data = path.read_bytes()
        if name in files:
            assert files[name] == data
        files[name] = data
        return data
    plans = [read(BASE / f'lithos-dspark-{mode}-v1/plan.json') for mode in ('none', 'dspark')]
    assert plans[0]['pid'] != plans[1]['pid']
    for key in ('lithos_head', 'source_sha256', 'harness_sha256', 'target_identity', 'drafter_identity', 'prompt_ids', 'environment', 'workload_sha256', 'download_receipt_sha256', 'python'):
        assert plans[0][key] == plans[1][key], key
    assert plans[0]['lithos_head'] == '223658090ae549e59220bdec459957cbdad7926a'
    assert plans[0]['environment']['MONOLITH_ACCELERATOR'] == 'on'
    for key in ('source_sha256', 'harness_sha256'):
        for path, expected in plans[0][key].items():
            assert sha(collect(path)) == expected, path
    # Weight payload hashes were checked before AND after by each sole-owner run.
    # Recheck all small metadata now; do not reread 30+ GiB of model/pack bytes.
    for role in ('target', 'drafter'):
        identity = plans[0][role + '_identity']
        for name, metadata in identity['files'].items():
            if not name.endswith('.safetensors'):
                data = collect(Path(identity['path']) / name, f'model-metadata/{role}/{name}')
                assert len(data) == metadata['bytes'] and sha(data) == metadata['sha256']
    target_config = json.loads(files['model-metadata/target/config.json'])
    eos = target_config['eos_token_id']
    eos_set = set(eos if isinstance(eos, list) else [eos])
    download = read(BASE / 'lithos-dspark-download-v1/complete.json')
    assert download['status'] == 'PASS_PINNED_DOWNLOAD_ONLY'
    assert download['snapshot'] == plans[0]['drafter_identity']['path']
    assert len(download['shards']) == 3 and sum(x['bytes'] for x in download['shards']) == 1136258502
    for shard in download['shards']:
        assert plans[0]['drafter_identity']['files'][shard['name']] == {key: shard[key] for key in ('bytes', 'sha256')}
    assert sha((BASE / 'lithos-dspark-download-v1/complete.json').read_bytes()) == plans[0]['download_receipt_sha256']
    workload = BASE / 'fullqual-build2/workload.json'
    assert sha(collect(workload)) == plans[0]['workload_sha256']
    assert read(workload)['prompt_ids'] == plans[0]['prompt_ids'] and len(plans[0]['prompt_ids']) == 19
    program_stats, pack_stats = [], []
    for mode, plan in zip(('none', 'dspark'), plans):
        run = BASE / f'lithos-dspark-{mode}-v1'
        result = read(run / 'result.json')
        assert result['status'] == 'PASS_ARM_REQUIRES_FRESH_CONTROL' and result['pins_after_match']
        assert result['mode'] == plan['mode'] == mode
        route = result['route']
        assert route == read(run / 'route.json')
        assert route['device']['name'] == 'Apple M5 Ultra' and route['device']['gpu_cores'] == 80 and route['device']['apple_family'] == 10
        assert route['backend'] == route['profile']['backend'] == 'm5_ultra_80c'
        assert route['recipes'] == {} and route['recipe_key'] is None
        assert route['capacity'] == (2054 if mode == 'dspark' else 2048)
        assert route['eos'] == eos and route['normal_eos'] and route['prefix_cache'] is False
        assert route['prefill_chunk_size'] == 128 and route['requested_prefill_exact'] and not route['effective_prefill_exact']
        assert route['has_drafter'] is (mode == 'dspark') and route['gamma'] == (7 if mode == 'dspark' else 0)
        if mode == 'dspark':
            assert route['drafter_class'] == 'monolith.spec.dspark.model.DSparkDrafter'
            assert route['verify'] == 'fixed' and route['verify_length'] == 7
        else:
            assert route['drafter_class'] is None and result['packs']['draft'] is None
        options = read(run / 'prepare-options.json')
        assert options['no_draft'] is (mode == 'none') and options['draft_quantization'] == 'none'
        assert options['draft'] == (plan['drafter_identity']['path'] if mode == 'dspark' else None)
        for role, record in result['packs'].items():
            if record is None:
                continue
            raw = (run / f'{role}-pack-manifest.json').read_bytes()
            manifest = json.loads(raw)
            actual_manifest = Path(record['path']) / 'manifest.json'
            assert sha(actual_manifest.read_bytes()) == record['manifest_sha256']
            # Saved JSON was pretty-printed; compare parsed content, not formatting.
            assert manifest == read(actual_manifest)
            assert manifest['nbytes'] == record['bytes'] == (Path(record['path']) / manifest['pack']).stat().st_size
            assert manifest['options']['max_context'] == record['capacity'] == route['capacity']
            identity = manifest['cache_identity']
            assert identity['backend'] == 'm5_ultra_80c' and identity['capacity'] == route['capacity']
            assert identity['role'] == role and identity['quantization'] is None and record['quantize'] is None
            source_identity = plan['drafter_identity' if role == 'draft' else 'target_identity']
            assert identity['checkpoint']['path'] == source_identity['path']
            assert identity['checkpoint']['config_sha256'] == source_identity['files']['config.json']['sha256']
            assert identity['checkpoint']['index_sha256'] == source_identity['files']['model.safetensors.index.json']['sha256']
            assert identity['layout']['lane_order'] == 'interleaved16' and identity['layout']['scale_placement'] == 'block'
            formats = dict(Counter(s['format'] for s in manifest['slabs']))
            assert formats == ({'nvfp4': 27, 'bf16': 1} if role == 'draft' else {'nvfp4': 258})
            pack_stats.append({'mode': mode, 'role': role, **record, 'slab_formats': formats, 'layout': identity['layout']})
            collect(actual_manifest, f'pack-metadata/{mode}-{role}-manifest-original.json')
        assert result['programs'] == read(run / 'programs.json') and len(result['programs']) == 2
        for record in result['programs']:
            raw = (run / f"program-{record['index']}.json").read_bytes()
            assert sha(raw) == record['program_sha256']
            program = json.loads(raw)
            assert program['backend_id'] == record['backend'] == 'm5_ultra_80c'
            assert program['context_capacity'] == record['context_capacity'] == 2048
            assert program['config_digest'] == record['config_digest']
            assert len(program['ops']) == record['op_count']
            assert sum('draft.' in op['name'] for op in program['ops']) == record['draft_ops']
            assert sorted({k['function'] for k in program['kernels'].values()}) == record['kernel_functions']
            kernels = {key: {'source_sha256': sha(value['source'].encode()), 'function': value['function'],
                       'macros': value['macros'], 'language_version': value['language_version']} for key, value in program['kernels'].items()}
            assert kernels == record['kernels']
            eos_kernels = [k for k in program['kernels'].values() if k['function'] == ('accept_scan' if mode == 'dspark' else 'advance')]
            assert eos_kernels and all(k['macros']['EOS_TEST'] == '(tok==248044||tok==248046)' for k in eos_kernels)
            if not record['prefill']:
                assert record['bound'] == (8 if mode == 'dspark' else 1) and record['dynamic'] is (mode == 'dspark')
                assert record['draft_ops'] == (43 if mode == 'dspark' else 0)
                if mode == 'dspark':
                    assert {'accept_scan', 'verify_select'} <= set(record['kernel_functions'])
                    assert any('draft.markov_w2' in op['name'] for op in program['ops'])
                else:
                    assert 'accept_scan' not in record['kernel_functions']
            program_stats.append({'mode': mode, **{k: v for k, v in record.items() if k != 'kernels'}, 'kernel_count': len(kernels)})
        rows = result['requests']
        assert len(rows) == 3 and [r['index'] for r in rows] == [0, 1, 2]
        details = []
        for row in rows:
            assert row == read(run / f"request-{row['index']}.json")
            gen, state = row['generation'], row['step_state']
            assert row['prompt_ids'] == plan['prompt_ids'] and len(gen['tokens']) == 32
            assert not eos_set.intersection(gen['tokens']) and not row['eos_in_returned_prefix']
            assert gen['cached_prompt_tokens'] == 0 and gen['decode_tokens'] == 31
            assert sum(item['tokens'] for item in gen['prefill_timings']) == 19
            assert state['error'] == 0 and state['done'] == 1 and state['stop_at'] == 32
            assert state['position'] == 50 and state['ring_head'] == state['ring_tail'] == 32
            all_ids.append(gen['tokens'])
            if mode == 'dspark':
                raw = (run / f"accept-log-{row['index']}.bin").read_bytes()
                words = struct.unpack('<' + 'I' * (len(raw) // 4), raw)
                indices = [i for i, word in enumerate(words[:state['step']]) if i >= len(gen['prefill_timings']) and (word & 65535) != 65535]
                accepted = [words[i] & 255 for i in indices]
                verified = [(words[i] >> 8) & 255 for i in indices]
                committed = [words[i] >> 16 for i in indices]
                assert gen['accepted'] == accepted and gen['verify_len'] == verified and gen['committed'] == committed
                assert row['accept_log_rows'] == [words[i] for i in indices]
                assert len(indices) == gen['steps'] == 14 and state['step'] == 15
                assert all(v == 7 for v in verified) and all(c == a + 1 for a, c in zip(accepted, committed))
                assert sum(accepted) == 17 and sum(committed) == 31 and sum(verified) == 98
                conf = (run / f"confidence-log-{row['index']}.bin").read_bytes()
                conf_values = struct.unpack('<' + 'f' * (len(conf) // 4), conf)
                conf_rows = [list(conf_values[i * 16:i * 16 + 7]) for i in indices]
                assert conf_rows == gen['confidences'] and all(math.isfinite(v) for r in conf_rows for v in r)
                details.append({'request': row['index'], 'decode_rounds': 14, 'accepted': sum(accepted), 'verified': sum(verified),
                                'committed': sum(committed), 'acceptance_histogram': dict(Counter(accepted)), 'budget_overshoot': 0})
            else:
                assert gen['steps'] == 31 and state['step'] == 32
                assert all(gen[k] is None for k in ('accepted', 'committed', 'verify_len', 'confidences'))
                assert row['accept_log_rows'] is None
                details.append({'request': row['index'], 'decode_rounds': 31, 'committed': 31})
        stats[mode] = details
        cleanup = read(run / 'cleanup.json')
        assert cleanup['released_engines'] and cleanup['buffers_cleared'] and cleanup['last_engine_cleared']
        owner = BASE / f'lithos-dspark-{mode}-v1-owner'
        release, child, complete = [read(owner / name) for name in ('release.json', 'child.json', 'complete.json')]
        assert release['pid'] == child['pid'] == plan['pid'] and release['returncode'] == 0
        assert release['waited_reaped'] and release['strict_absence'] and complete['status'] == 'PASS'
        owner_records.append({'mode': mode, 'owner_pid': complete['owner_pid'], **release})
        resources = [read(path) for path in sorted(owner.glob('resource-*.json'))]
        assert len(resources) == (9 if mode == 'none' else 10)
        all_resources.extend(resources)
        for folder in (run, owner):
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    collect(path)
    assert len(all_ids) == 6 and all(ids == all_ids[0] for ids in all_ids)
    for resource in all_resources:
        assert resource['pressure_level_value'] == 1 and resource['processinfo_thermal_metadata']['code'] == 0
        assert resource['no_active_foreign_native_processes'] and resource['power_metadata']['source'] == 'AC Power'
        paused = [p for p in resource['native_inference_processes'] if p['pid'] == 74384]
        assert len(paused) == 1 and paused[0]['authorized_suspended'] and 'T' in paused[0]['state']
    assert len({r['swap_used_mb'] for r in all_resources}) == len({r['vm_counters']['Swapouts'] for r in all_resources}) == 1
    lease_dir = BASE / 'lithos-dspark-service-lease-v1'
    lease, resume = read(lease_dir / 'lease.json'), read(lease_dir / 'resume.json')
    jobs = read(BASE / 'lithos-dspark-jobs-v1.json')
    assert lease['closed'] and lease['service_pid'] == 74384 and lease['jobs'] == jobs
    assert resume['identity_match'] and resume['state_before'] == 'T' and resume['state_after'] == 'S'
    for index, owner in enumerate(owner_records):
        job = read(lease_dir / f'job-{index}.json')
        assert job['command'] == jobs[index] and job['pid'] == owner['owner_pid'] and job['returncode'] == 0 and job['waited']
    for folder in (lease_dir, BASE / 'lithos-dspark-download-v1', ROOT / 'bench/m5_ultra_256gb/lithos_spec/metadata-v1'):
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                collect(path)
    for name in ('lithos-dspark-jobs-v1.json', 'lithos-dspark-comparison-v1.json'):
        collect(BASE / name)
    comparison = read(BASE / 'lithos-dspark-comparison-v1.json')
    assert comparison['status'] == 'PASS_LITHOS_DSPARK_ROUTE_AND_FRESH_CONTROL_IDS' and comparison['raw_ids_compared'] == 192
    assert comparison['raw_ids'] == all_ids[0]
    for path, expected in comparison['receipts'].items():
        assert sha(Path(path).read_bytes()) == expected
    collect(Path(__file__))
    for name in ('guarded_v2.py', 'resources_v2.py', 'suspended_service.py'):
        collect(ROOT / 'bench/m5_ultra_256gb' / name)
    summary = {'status': 'PASS_INDEPENDENT_LITHOS_DSPARK_QUALIFICATION_AUDIT', 'requests': 6, 'raw_ids_compared': 192,
               'modes': stats, 'programs': program_stats, 'packs': pack_stats, 'eos_ids': eos,
               'no_eos_token_observed': True, 'scope': 'One 19-token raw prompt repeated three times in each of two fresh processes; route and returned-ID qualification only, no full-state, HTTP or speed ranking',
               'source_hashes_checked': len(plans[0]['source_sha256']), 'harness_hashes_checked': len(plans[0]['harness_sha256']),
               'resource_samples': len(all_resources), 'pressure': 1, 'thermal': 0, 'swap_used_mb': all_resources[0]['swap_used_mb'],
               'swapouts': all_resources[0]['vm_counters']['Swapouts'], 'owner_release': owner_records, 'service_resume': resume,
               'model_and_pack_payload_hashes': 'Each runtime checked payloads before/after; audit independently checked metadata/manifests/Program/log/source bytes without rereading or archiving large weight payloads'}
    OUT.mkdir(parents=True)
    archive = OUT / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, data in sorted(files.items()):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            tar.addfile(item, io.BytesIO(data))
    manifest = {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in sorted(files.items())}
    with tarfile.open(archive) as tar:
        assert len(tar.getnames()) == len(manifest) and set(tar.getnames()) == set(manifest)
        for name, item in manifest.items():
            data = tar.extractfile(name).read()
            assert len(data) == item['bytes'] and sha(data) == item['sha256']
    archive_sha = sha(archive.read_bytes())
    write(OUT / 'manifest.json', {'archive_sha256': archive_sha, 'files': manifest})
    write(OUT / 'tar.verify.json', {'status': 'PASS', 'archive_sha256': archive_sha, 'members': len(manifest), 'all_member_hashes_and_sizes_verified': True})
    write(OUT / 'summary.json', summary)
    (OUT / 'README.md').write_text(f'''# E017：Lithos DSpark 实际路由与短请求资格通过

两个 fresh process 各跑三次相同 19→32 greedy 请求，全部 **192 个 raw token IDs** 与普通 Lithos 控制一致。正常 EOS 配置为 `{eos}`，没有关闭 EOS；本请求未生成 EOS，因此不宣称覆盖 EOS 分支。prefix cache 未命中。

每次 DSpark 请求实际 **14 轮、98 proposals verified、17 accepted、31 decode tokens committed**，加上 prefill 产出的 1 token，共返回 32 tokens。接受 histogram 为 0×7、1×2、2×4、7×1，覆盖零接受、部分拒绝与全接受。本轮最后一块刚好到达预算，没有 overshoot；一般允许完整块多提交的语义仍与返回数分开记录。普通臂每次 31 decode rounds。这些计数不等于速度收益。

实际设备为 Apple M5 Ultra / 80 cores / Apple10，backend `m5_ultra_80c`，recipe 为空；没有套用 Max 配置。DSpark decode program 为 dynamic bound 8、586 ops，其中 43 个 draft ops，存在 Markov W2、verify_select 与 accept_scan；普通 decode 为 bound 1、535 ops，无 draft ops。完整 Program、每 kernel source SHA/macros 和原始 accept/conf 二进制日志已独立重算核对。

两臂 target pack capacity 为 2048 / 2054，DSpark draft capacity 为 2054；两个 program 的有效 target context 上限均为 2048。target 为 258 个 NVFP4 slabs；draft 为 27 个 NVFP4 + 1 个 BF16 slab，合并后的 packed slab 数不能冒充 checkpoint 矩阵数。quantization=None，原生 layout interleaved16/block；权重原字节的前后 SHA 由两次运行保存，CPU 审计不重复读取 30+ GiB payload。

19 个资源快照 pressure=1、thermal=0，swap=268.81 MiB 与 Swapouts=303419 没有增加。两个子进程 exit 0，wait/reap 与严格不存在检查通过。外部 omlx-server PID 74384 身份匹配并从 T 恢复 S。

`evidence.tar.gz` 保存完整 run/owner/lease/jobs、download/comparison/metadata、221 份被测 Lithos source/native binary、3 份 harness、4 份完整 Program、模型非权重 metadata 和原始 pack manifest；`manifest.json` / `tar.verify.json` 逐字节校验全部 {len(manifest)} 个成员。`summary.json` 为独立 CPU 审计重算结论。

资格只覆盖同一个短 prompt 的重复请求、真实路由及 returned IDs，未验证全部 attention/GDN/conv/draft state，也没有 HTTP、长生成或速度排名结论。观察器、pack 构建、哈希与 Program 序列化都在此运行中，后续性能必须用独立无观察器流程。
''')
    print(json.dumps({'status': summary['status'], 'members': len(manifest), 'archive_bytes': archive.stat().st_size}))


if __name__ == '__main__':
    main()
