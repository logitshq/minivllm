"""Offline, correctness-gated v3 C2..64 / E020 C1 scientific figure.

Only audited chartdata and their frozen source provenance are read. No server,
model, MLX, network, or performance test is run. Output must be a new directory.
Use --snapshot PREVIOUS_OUTPUT/sources to redraw entirely from copied inputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics as st

ROOT = Path(__file__).resolve().parents[3]
PROGRESS = 'docs/optimization-guide/progress/2026-10-17-'
PINS = {
    'v3': {'chartdata.json': '6b6664e0cdf9dc5108779f96b093072e6079f082b4be969facdd14712d6ba43a',
           'source-manifest.json': '1e633f637af827b0dfa7e9f92c79dcc69743cec0ba2809bfb61787bdb8e4e43c'},
    'v4': {'chartdata.json': 'f5fab1581521f5eac6b3ab13870846abc8f9324dac0529fd8e41931d148795be',
           'source-manifest.json': '7f39ad30ec3fc260aca69a69f8f22f5532732ebd8c59472a13b3c04cf3e94840'},
}
AUDIT = 'bench/reviews/pareto-v3-v4-audit.md'
SUPPORT = ('docs/experiments/m5-ultra-256gb/E020-external-spec-http-v1/summary.json',
           '.artifacts/m5-ultra-256gb/glm-flash-physical-preflight-v1/raw.json',
           '.artifacts/m5-ultra-256gb/glm-flash-physical-preflight-v1/service-location.json')
ARMS = ('native-none', 'omlx-none', 'omlx-mtp', 'lithos-none', 'lithos-dspark')
COLORS = {'native': '#127E6B', 'omlx': '#D76C28', 'native-none': '#127E6B',
          'omlx-none': '#B8672D', 'omlx-mtp': '#DF922F', 'lithos-none': '#526BB1', 'lithos-dspark': '#9B63A4'}
LABELS = {'native': 'minivllm', 'omlx': 'oMLX balanced', 'native-none': 'minivllm target-only',
          'omlx-none': 'oMLX target-only', 'omlx-mtp': 'oMLX + MTP',
          'lithos-none': 'Lithos target-only', 'lithos-dspark': 'Lithos + DSpark'}
MARKERS = {'native': 'o', 'omlx': 's', 'native-none': 'o', 'omlx-none': 's',
           'omlx-mtp': '^', 'lithos-none': 'D', 'lithos-dspark': 'P'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


class Sources:
    def __init__(self, output, snapshot):
        self.output, self.snapshot, self.manifest = output, snapshot, {}

    def copy(self, original, relative, expected=None):
        source = self.snapshot / relative if self.snapshot else ROOT / original
        target = self.output / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if expected is not None:
            assert actual == expected, (original, actual, expected)
        target.write_bytes(payload)
        self.manifest[original] = {'snapshot_path': relative, 'sha256': actual, 'bytes': len(payload)}
        return payload


def inputs(sources):
    audit = sources.copy(AUDIT, 'audit/pareto-v3-v4-audit.md').decode()
    reviewed = dict((path, sha) for sha, path in re.findall(r'^([0-9a-f]{64})  (.+)$', audit, re.M))
    for index, original in enumerate(SUPPORT):
        sources.copy(original, f'audit/support-{index}.json', reviewed[original])
    datasets, manifests = {}, {}
    for version in ('v3', 'v4'):
        root = PROGRESS + version
        for name in ('chartdata.json', 'source-manifest.json', 'README.md', 'summary.json'):
            original = f'{root}/{name}'
            payload = sources.copy(original, f'inputs/{version}/{name}', reviewed[original])
            if name in PINS[version]:
                assert hashlib.sha256(payload).hexdigest() == PINS[version][name]
            if name == 'chartdata.json':
                datasets[version] = json.loads(payload)
            if name == 'source-manifest.json':
                manifests[version] = json.loads(payload)
        for meta in manifests[version].values():
            relative = meta['snapshot_path']
            sources.copy(f'{root}/sources/{relative}', f'inputs/{version}/sources/{relative}', meta['sha256'])
    return datasets, manifests


def dominates(left, right):
    return (left['latency_ms'] <= right['latency_ms'] and left['throughput'] >= right['throughput']
            and (left['latency_ms'] < right['latency_ms'] or left['throughput'] > right['throughput']))


def collect(data, manifests, output):
    def raw(version, original):
        return json.loads((output / 'sources' / 'inputs' / version / 'sources' / manifests[version][original]['snapshot_path']).read_text())
    history = [p for p in data['v3']['processes'] if p['engine'] in ('native', 'omlx') and p['C'] in (2, 4, 8, 16, 32, 64)]
    assert len(history) == 36
    n_history = 0
    for process in history:
        original = raw('v3', process['source'])
        assert len(process['groups']) == len(original['groups']['measure']) == 3
        for group, observed in zip(process['groups'], original['groups']['measure']):
            requests = observed['requests']
            assert len(requests) == process['C'] == len(group['requests'])
            latencies, tokens = [], 0
            for recorded, request in zip(group['requests'], requests):
                assert recorded['request_id'] == request['request_id']
                assert request['finish_reason'] == 'length' and request['usage']['completion_tokens'] == 32
                assert request['usage']['prompt_tokens'] == 19 and len(request['visible_roundtrip_ids']) == 32
                assert request['full_visible_roundtrip_count_matches_usage']
                if request['output_ids_when_available'] is not None:
                    assert request['output_ids_when_available'] == request['visible_roundtrip_ids']
                s, f, d = (request[k] for k in ('start_perf_counter_s', 'first_text_perf_counter_s', 'done_perf_counter_s'))
                assert s <= f <= d <= request['cleanup_complete_perf_counter_s']
                latency = (d - s) * 1000
                same(latency, recorded['latency_ms'])
                same((f - s) * 1000, recorded['ttft_ms'])
                latencies.append(latency)
                tokens += request['usage']['completion_tokens']
            window = max(r['done_perf_counter_s'] for r in requests) - min(r['start_perf_counter_s'] for r in requests)
            same(st.median(latencies), group['latency_ms'])
            same(tokens / window, group['throughput'])
            n_history += len(requests)
        for metric in ('latency_ms', 'throughput', 'ttft_ms'):
            same(st.median(g[metric] for g in process['groups']), process[metric])
    assert n_history == 2268
    points = []
    for engine in ('native', 'omlx'):
        for c in (2, 4, 8, 16, 32, 64):
            rows = [r for r in history if r['engine'] == engine and r['C'] == c]
            assert len(rows) == 3 and {r['replicate'] for r in rows} == {1, 2, 3}
            point = {'engine': engine, 'C': c, **{m: st.median(r[m] for r in rows) for m in ('latency_ms', 'throughput', 'ttft_ms')}}
            previous = next(p for p in data['v3']['points'] if p['engine'] == engine and p['C'] == c)
            for metric in ('latency_ms', 'throughput', 'ttft_ms'):
                same(point[metric], previous[metric])
            points.append(point)
    for point in points:
        point['own_frontier'] = not any(dominates(other, point) for other in points if other['engine'] == point['engine'])
        point['panel_frontier'] = not any(dominates(other, point) for other in points)
    current, current_points = data['v4']['processes'], []
    assert len(current) == 15 and data['v4']['status'] == 'PASS_CONTRACT'
    n_current = 0
    for process in current:
        assert process['valid'] and not process['contract_failures'] and process['fatal_failure'] is None
        assert len(process['requests']['formal']) == 3
        for request in process['requests']['formal']:
            observed = raw('v4', request['source'])
            assert observed['done'] and observed['http_status'] == 200 and not observed['contract_failures'] and observed['error'] is None
            assert observed['finish_reason'] == 'length' and observed['usage']['completion_tokens'] == 32
            latency = observed['done_perf_counter_s'] - observed['start_perf_counter_s']
            same(latency, request['latency_s'])
            same(observed['usage']['completion_tokens'] / latency, request['throughput'])
            n_current += 1
        for metric in ('latency_s', 'throughput', 'ttft_s'):
            same(st.median(r[metric] for r in process['requests']['formal']), process['medians'][metric])
    assert n_current == 45
    for arm in ARMS:
        rows = [p for p in current if p['arm'] == arm]
        assert len(rows) == 3 and {r['pair'] for r in rows} == {1, 2, 3}
        point = {'arm': arm, 'C': 1, 'latency_ms': 1000 * st.median(p['medians']['latency_s'] for p in rows),
                 'throughput': st.median(p['medians']['throughput'] for p in rows)}
        old = next(p for p in data['v4']['points'] if p['arm'] == arm)
        same(point['latency_ms'], old['medians']['latency_s'] * 1000)
        same(point['throughput'], old['medians']['throughput'])
        same(point['throughput'], 32000 / point['latency_ms'])
        current_points.append(point)
    for point in current_points:
        point['panel_frontier'] = not any(dominates(other, point) for other in current_points)
    excluded = [{'engine': p['engine'], 'C': p['C'], 'replicate': p['replicate'], 'source': p['source'],
                 'reason': ('native historical C1 is outside the historical C2..64 panel; current C1 remains a separate round'
                            if p['engine'] == 'native' else 'historical external C1 lacks evidence excluding PID74384 interference; activity and causal contamination are NOT established')}
                for p in data['v3']['processes'] if p['C'] == 1]
    return {'schema': 'qualified-pareto-v5', 'panels': {
        'historical_C2_C64': {'processes': history, 'points': points, 'process_count': 36, 'formal_requests': n_history,
                             'aggregation': data['v3']['aggregation'], 'quality_contract': data['v3']['quality_contract']},
        'E020_C1': {'processes': current, 'points': current_points, 'process_count': 15, 'formal_requests': n_current,
                    'aggregation': data['v4']['aggregation'], 'quality_contract': 'E020 independent PASS_RECONCILED / COMPLETE_VALID_CONTRACT; external visible IDs versus prior same-engine raw-ID qualification'}},
        'exclusions': excluded, 'scope': 'Independent historical and E020 panels. No pooling, cross-round Pareto, cross-framework token identity, GLM point or E022 benefit.',
        'axes': {'x': 'full request POST/start to SSE DONE p50, ms; lower is better',
                 'y': 'actual group completion tokens / (last DONE - first POST), tokens/s/Mac; higher is better'},
        'audit': AUDIT, 'c1_identity': 'At C1 with fixed 32 tokens, throughput = 32000 / latency_ms.'}


def render(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'figure.facecolor': 'white',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'qualified-pareto-v5'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [1.25, 1]})
    fig.subplots_adjust(left=.065, right=.975, top=.73, bottom=.25, wspace=.23)
    fig.text(.065, .955, 'M5 Ultra · throughput vs complete-request latency', size=23, weight='bold')
    fig.text(.065, .907, 'Qwen3.8-27B NVFP4 · 19 input / 32 output · correctness-gated evidence · 256 GiB', size=13)
    fig.text(.065, .869, 'Better = upper left. Historical concurrency and latest C1 are separate measurements.', size=12, color='#46505A')
    hist, latest = data['panels']['historical_C2_C64'], data['panels']['E020_C1']
    left.set_title('A  Historical target-only · C2–64\n36 processes · 2,268 formal requests', loc='left', size=13, pad=42)
    right.set_title('B  E020 same-round · C1\n15 processes · 45 formal requests', loc='left', size=13, pad=42)
    for engine in ('native', 'omlx'):
        rows = [p for p in hist['processes'] if p['engine'] == engine]
        points = [p for p in hist['points'] if p['engine'] == engine]
        left.scatter([p['latency_ms'] for p in rows], [p['throughput'] for p in rows], s=28,
                     color=COLORS[engine], marker=MARKERS[engine], alpha=.27, zorder=2)
        front = sorted([p for p in points if p['own_frontier']], key=lambda p: p['latency_ms'])
        left.plot([p['latency_ms'] for p in front], [p['throughput'] for p in front], color=COLORS[engine], lw=1.6, zorder=3)
        for point in points:
            left.scatter(point['latency_ms'], point['throughput'], s=85, marker=MARKERS[engine],
                         facecolor=COLORS[engine] if point['own_frontier'] else 'white',
                         edgecolor=COLORS[engine], linewidth=1.7, zorder=4)
            offset = (7, 10) if engine == 'native' else (7, -17)
            if point['C'] == 64:
                offset = (7, -18)
            left.annotate('C' + str(point['C']), (point['latency_ms'], point['throughput']), xytext=offset,
                          textcoords='offset points', color=COLORS[engine], fontsize=10.5)
    left.legend(handles=[Line2D([], [], marker=MARKERS[e], color=COLORS[e], label=LABELS[e], lw=1.6) for e in ('native', 'omlx')],
                loc='lower left', bbox_to_anchor=(-.01, 1.005), frameon=False, ncol=2, fontsize=10.5)
    left.set_xscale('log'); left.set_xlim(650, 28500); left.set_ylim(0, 215)
    left.set_xticks([1000, 2000, 5000, 10000, 20000])
    left.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    left.set_xlabel('Full-request p50 latency (ms ↓; logarithmic x)', labelpad=11)
    left.set_ylabel('Total output throughput (tokens/s/Mac ↑)', labelpad=10)
    label_positions = {'native-none': (690, 44.1), 'omlx-none': (698, 47.4), 'omlx-mtp': (776, 42.6),
                       'lithos-none': (784, 38.6), 'lithos-dspark': (697, 41.3)}
    for point in latest['points']:
        arm = point['arm']
        rows = [r for r in latest['processes'] if r['arm'] == arm]
        right.scatter([r['medians']['latency_s'] * 1000 for r in rows], [r['medians']['throughput'] for r in rows],
                      color=COLORS[arm], marker=MARKERS[arm], s=38, alpha=.30, zorder=2)
        right.scatter(point['latency_ms'], point['throughput'], s=110, color=COLORS[arm],
                      marker=MARKERS[arm], edgecolor='white', linewidth=.8, zorder=4)
        right.annotate(LABELS[arm], (point['latency_ms'], point['throughput']), xytext=label_positions[arm],
                       color=COLORS[arm], fontsize=10.5, ha='left', va='center',
                       arrowprops={'arrowstyle': '-', 'color': COLORS[arm], 'lw': .8}, zorder=5)
    right.text(.02, 1.025, 'All configurations C1 · filled marks = configuration medians',
               transform=right.transAxes, fontsize=10, color='#46505A')
    right.set_xlim(655, 853); right.set_ylim(37.7, 49)
    right.set_xticks([675, 700, 725, 750, 775, 800, 825, 850])
    right.set_xlabel('Full-request p50 latency (ms ↓; linear x)', labelpad=11)
    right.set_ylabel('Total output throughput (tokens/s/Mac ↑)', labelpad=10)
    for ax in (left, right):
        ax.grid(color='#E4E7E9', linewidth=.7)
        ax.set_axisbelow(True)
    fig.text(.065, .125,
        'Large marks: medians of three fresh process medians. Faint marks: every process; no confidence intervals.\n'
        'A: filled marks and lines show each engine’s measured nondominated configurations; hollow marks retain regressions.\n'
        'B: fixed 32-token C1 requests make throughput = 32 / latency(s); this is a slice, not a concurrent sweep.',
        fontsize=10.5, color='#46505A', linespacing=1.6)
    fig.text(.065, .025,
        'Independent axes and rounds; historical native and external runs were not contemporaneously paired.\n'
        'Historical external C1 excluded because interference exclusion is unproven. Per-engine correctness contracts apply; cross-engine token identity is not established.',
        fontsize=10.3, color='#46505A', linespacing=1.55)
    for extension in ('png', 'svg'):
        fig.savefig(output / f'qualified-throughput-latency.{extension}', dpi=180,
                    metadata={'Date': None} if extension == 'svg' else None)
    plt.close(fig)
    return matplotlib.__version__


def readme(data, output):
    history, latest = data['panels']['historical_C2_C64'], data['panels']['E020_C1']
    left = '\n'.join(f"| {LABELS[p['engine']]} | {p['C']} | {p['latency_ms']:.3f} | {p['throughput']:.6f} | {'是' if p['own_frontier'] else '否'} |" for p in history['points'])
    right = '\n'.join(f"| {LABELS[p['arm']]} | {p['latency_ms']:.3f} | {p['throughput']:.6f} | {'是' if p['panel_frontier'] else '否'} |" for p in latest['points'])
    (output / 'README.md').write_text(f'''# Correctness-gated throughput–latency · 2026-10-17 v5

横轴是完整请求从 POST/start 到 SSE DONE 的实际 p50 延迟，单位 ms；纵轴是实际输出 token 总数除以整组首 POST 到末 DONE 的窗口，单位 tok/s/Mac。正确性是入图前提。图中两轮分别展示、分别计算支配关系；没有拼成跨轮前沿。

![Qualified throughput vs complete-request latency](qualified-throughput-latency.png)

[SVG](qualified-throughput-latency.svg) · [全部进程/分组数据](chartdata.json) · [输入与来源 SHA](source-manifest.json) · [独立审计副本](sources/audit/pareto-v3-v4-audit.md)

左面板保留历史 native/oMLX target-only 的 C2、4、8、16、32、64：36 个 fresh process、2,268 条正式请求。先取每组请求延迟 p50，再取每进程三个组的中位数，最后取三个进程的中位数；吞吐按组实际 token/window 计算后采用同样聚合层级。浅色点是全部进程，不是置信区间。仅连接各引擎已测非支配配置；空心点保留较差配置，没有删除 C16/C64。左 x 为 log；两面板轴范围独立。

| 历史配置 | C | 完整请求 latency ms ↓ | 整机 tok/s ↑ | 引擎内前沿 |
|---|---:|---:|---:|---|
{left}

右面板完整保留 E020 同轮 C1 的五配置、15 个 fresh process、45 条正式请求。每进程三次正式请求，配置点为三个进程中位数的中位数。oMLX target-only 是这一同轮切面唯一非支配配置；不据此外推其他并发、长度或模型。C1 每次恰好32输出 token，所以 `T=32000/L_ms`，它只是一张切面，不能替代并发扫描。TTFT 不作为本图横轴。

| E020 同轮 C1 | 完整请求 latency ms ↓ | 整机 tok/s ↑ | 本面板前沿 |
|---|---:|---:|---|
{right}

历史外部 C1 的排除依据是资源来源证据不足，不是按性能挑选。PID74384 启动于 2026-10-16 15:01:58 UTC；旧外部 C1 资源记录在 15:03:12.541121–15:05:18.369638 UTC，旧检测器只匹配已知 benchmark 命令，无法证明该 `omlx-server` 不存在、被暂停或 GPU 空闲。证据同样不能证明它当时一定活跃推理，也不能把差值归因为干扰。旧 C1 数据仍完整保存在输入快照；历史 native C1 也不拼接到新 C1 面板。

历史外部 C2–64 资源记录为 14:34:32.449459–14:59:06.383925 UTC，早于该 PID 启动；这只排除这个具体 PID 的时间重叠，不能把旧检测器追认为完整 GPU 独占证明。native 与外部仍是不同时间测量，左图只称历史比较。

正确性范围严格沿用 [v3](../2026-10-17-v3/README.md)、[v4](../2026-10-17-v4/README.md) 和独立审计：native 在固定输入/同调度下通过 logits/cache/输出精确资格；历史外部 HTTP 只有已记录的计数、visible-text/roundtrip 合同，没有 live raw-ID oracle。E020 外部 visible IDs 与此前各自引擎 raw-ID 资格核对，native 当次直接核 raw IDs。跨框架逐 token 相同、相同下游任务质量、每条自由到达调度的完整数值等价均未建立。不得把这些范围扩大为统一质量 oracle 或“全网最快”。

均限定 M5 Ultra 256 GiB、同一 Qwen3.8-27B NVFP4 snapshot、19-token 输入、greedy、normal EOS、cap32、无 prefix-cache 命中；历史左图仅 target-only，右图另含官方 MTP/DSpark。GLM、未通过的 E022 与未获得该合同的实验没有点。C 表示 HTTP outstanding，不保证 GPU batch 恒等于 C；历史组是有限 burst，不是持续混合业务流量。

新 renderer 逐请求重算选中数据的 timestamps/usage，并核对全部原始快照 SHA；没有修改 v3/v4。输入副本保留所有 204 个原始来源文件（含已排除 C1 便于复核）、两个 chartdata、manifests 与审计证据。未把两个面板的请求或进程池化。

离线重画，输出必须是新目录（Python + Matplotlib）：

```sh
.artifacts/m5-ultra-256gb/progress-render-venv/bin/python \\
  docs/optimization-guide/progress/2026-10-17-v5/sources/renderer/qualified_pareto.py \\
  --snapshot docs/optimization-guide/progress/2026-10-17-v5/sources \\
  --output /tmp/minivllm-qualified-pareto-redraw
```
''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.snapshot:
        existing = json.loads((args.snapshot.parent / 'source-manifest.json').read_text())
        for meta in existing.values():
            assert digest(args.snapshot / meta['snapshot_path']) == meta['sha256']
    sources = Sources(args.output, args.snapshot)
    data, manifests = inputs(sources)
    result = collect(data, manifests, args.output)
    write(args.output / 'chartdata.json', result)
    target = args.output / 'sources/renderer/qualified_pareto.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, target)
    sources.manifest['bench/m5_ultra_256gb/progress/qualified_pareto.py'] = {
        'snapshot_path': 'renderer/qualified_pareto.py', 'sha256': digest(target), 'bytes': target.stat().st_size}
    write(args.output / 'source-manifest.json', sources.manifest)
    version = render(result, args.output)
    readme(result, args.output)
    summary = {'status': 'RECOMPUTED_FROM_AUDITED_INPUTS_AND_HTTP_TIMESTAMPS',
        'matplotlib': version, 'panels': {key: {'processes': p['process_count'], 'formal_requests': p['formal_requests']} for key, p in result['panels'].items()},
        'source_files': len(sources.manifest), 'chartdata_sha256': digest(args.output / 'chartdata.json'),
        'excluded_historical_C1_processes': len(result['exclusions'])}
    write(args.output / 'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
