"""Frozen HTTP throughput versus measured request latency; CPU-only redraw."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics as st

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / 'docs/optimization-guide/progress/2026-10-17-v3'
RAW = '.artifacts/m5-ultra-256gb'
ENGINES = ('native', 'omlx', 'omlx_aggressive', 'lithos')
COLORS = {'native': '#127E6B', 'omlx': '#D76C28', 'omlx_aggressive': '#AB6427', 'lithos': '#947E21'}
LABELS = {'native': 'minivllm', 'omlx': 'oMLX balanced', 'omlx_aggressive': 'oMLX aggressive', 'lithos': 'Lithos'}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


class Sources:
    def __init__(self, out, snapshot):
        self.out, self.snapshot, self.manifest = out, snapshot, {}

    def read(self, original):
        relative = original.replace('.artifacts/', 'raw-artifacts/', 1)
        source = self.snapshot / relative if self.snapshot else ROOT / original
        target = self.out / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        self.manifest[original] = {'snapshot_path': relative, 'sha256': digest(target), 'bytes': target.stat().st_size}
        return json.loads(target.read_text())


def dominates(a, b, metric='latency_ms'):
    return a[metric] <= b[metric] and a['throughput'] >= b['throughput'] and (a[metric] < b[metric] or a['throughput'] > b['throughput'])


def collect(sources):
    frontier = sources.read('docs/experiments/m5-ultra-256gb/prefill-nax-v2/frontier.json')
    audit = sources.read('docs/experiments/m5-ultra-256gb/prefill-nax-v2/audit.json')
    assert audit['status'] == 'PASS' and audit['fixed_schedule_numerical_equality']
    assert frontier['status'] == 'PASS'
    specs = []
    for row in frontier['current_native_process_rows']:
        folder = 'prefill-http-v2' if row['C'] in (1, 8, 32, 64) else 'prefill-frontier-extension-v1'
        specs.append(('native', row['C'], row['pair'], f"{RAW}/{folder}/C{row['C']}-pair{row['pair']}-candidate/result.json", None))
    for row in frontier['historical_external_process_rows']:
        original = str(Path(row['path']).relative_to(ROOT))
        specs.append((row['engine'], row['C'], row['pair'], original, row['sha256']))
    assert len(specs) == 48
    processes, request_count = [], 0
    for engine, c, pair, original, expected in specs:
        result = sources.read(original)
        if expected:
            assert sources.manifest[original]['sha256'] == expected
            assert result['normal_EOS'] is True and result['draft'] is None and result['prefix_cache'] == 0
        else:
            ident = result['runtime_identity']
            assert ident['actual_source_files_sha256'] == frontier['source_files']
            assert ident['grouped_prefill'] and ident['compile_decode'] and ident['fuse_decode']
        groups = []
        assert len(result['groups']['measure']) == 3 and len(result['groups']['warm']) == 3
        for index, group in enumerate(result['groups']['measure']):
            requests = group['requests']
            assert len(requests) == group['actual_request_count'] == c
            records = []
            for r in requests:
                start, first, done = [r[k] for k in ('start_perf_counter_s', 'first_text_perf_counter_s', 'done_perf_counter_s')]
                assert start <= first <= done <= r['cleanup_complete_perf_counter_s']
                latency, ttft = done - start, first - start
                same(latency, r['client_total_s']); same(ttft, r['client_ttft_text_s'])
                assert r['finish_reason'] == 'length'
                assert r['usage']['completion_tokens'] == 32 and r['usage']['prompt_tokens'] == 19
                assert len(r['visible_roundtrip_ids']) == 32 and r['full_visible_roundtrip_count_matches_usage']
                if r['output_ids_when_available'] is not None:
                    assert r['output_ids_when_available'] == r['visible_roundtrip_ids']
                assert r['text'] == ''.join(e['text'] for e in r['events'])
                assert first == r['events'][0]['client_perf_counter_s']
                records.append({'request_id': r['request_id'], 'latency_ms': latency * 1000, 'ttft_ms': ttft * 1000})
            window = max(r['done_perf_counter_s'] for r in requests) - min(r['start_perf_counter_s'] for r in requests)
            throughput = c * 32 / window
            same(window, group['group_window_s']); same(throughput, group['y_total_output_throughput_tok_s_mac'])
            assert group['total_completion_tokens'] == c * 32
            groups.append({'latency_ms': st.median(r['latency_ms'] for r in records), 'ttft_ms': st.median(r['ttft_ms'] for r in records),
                           'throughput': throughput, 'requests': records, 'source_pointer': f'/groups/measure/{index}'})
            request_count += c
        row = {'engine': engine, 'C': c, 'replicate': pair, 'source': original, 'groups': groups,
               **{k: st.median(g[k] for g in groups) for k in ('latency_ms', 'ttft_ms', 'throughput')}}
        same(row['throughput'], result['y']); same(row['ttft_ms'] / 1000, result['TTFT_s'])
        processes.append(row)
    points = []
    for engine in ENGINES:
        for c in sorted({r['C'] for r in processes if r['engine'] == engine}):
            rows = [r for r in processes if r['engine'] == engine and r['C'] == c]
            assert len(rows) == 3
            point = {'engine': engine, 'C': c, **{k: st.median(r[k] for r in rows) for k in ('latency_ms', 'ttft_ms', 'throughput')}}
            old = next(p for p in frontier['points'] if p['engine'] == engine and p['C'] == c)
            same(point['throughput'], old['y']); same(point['ttft_ms'] / 1000, old['TTFT_s'])
            points.append(point)
    for p in points:
        p['joint_frontier'] = not any(dominates(q, p) for q in points)
        p['own_frontier'] = not any(dominates(q, p) for q in points if q['engine'] == p['engine'])
    coverage = [{'engine': p['engine'], 'C': p['C'], 'covered_by_native_C': [q['C'] for q in points if q['engine'] == 'native' and dominates(q, p)]}
                for p in points if p['engine'] != 'native']
    return {'date': '2026-10-17', 'points': points, 'processes': processes, 'request_count': request_count,
            'process_count': len(processes), 'coverage': coverage,
            'latency_definition': 'HTTP client done_perf_counter_s minus start_perf_counter_s; includes prefill, queuing and full 32-token delivery; excludes stream cleanup after done',
            'aggregation': 'Per group request p50; per process median of 3 measured groups; per configuration median of 3 fresh processes. Throughput uses group total tokens / wall window then same hierarchy.',
            'quality_contract': 'Only previously qualified target-only configurations. Native fixed-input/fixed-schedule logits/cache equality passed. External visible IDs/counts checked; cross-framework bitwise output equality not claimed. No known failed numerical candidate plotted.',
            'scope': 'M5 Ultra 256GiB, Qwen3.8-27B NVFP4, 19 input / 32 output, target-only, normal EOS. External runs archived earlier. No MTP/DSpark performance or unmeasured workload claim.'}


def render(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'minivllm-throughput-latency-v3', 'figure.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.subplots_adjust(left=.085, right=.97, bottom=.255, top=.76)
    fig.text(.085, .95, 'M5 Ultra · throughput–latency Pareto', size=23, weight='bold')
    fig.text(.085, .905, 'Qwen3.8-27B NVFP4 · 19 input / 32 output · target-only · 256 GiB', size=13)
    fig.text(.085, .87, 'Better = upper left. Latency is measured from request start to complete response.', size=12, color='#454B50')
    markers = {'native': 'o', 'omlx': 's', 'omlx_aggressive': '^', 'lithos': 'D'}
    for engine in ENGINES:
        rows = [r for r in data['processes'] if r['engine'] == engine]
        points = sorted([r for r in data['points'] if r['engine'] == engine], key=lambda p: p['latency_ms'])
        ax.scatter([r['latency_ms'] for r in rows], [r['throughput'] for r in rows], s=21, alpha=.22, color=COLORS[engine], marker=markers[engine])
        front = [p for p in points if p['own_frontier']]
        if len(front) > 1:
            ax.plot([p['latency_ms'] for p in front], [p['throughput'] for p in front], color=COLORS[engine], lw=1.7, alpha=.9)
        for i, p in enumerate(points):
            ax.scatter(p['latency_ms'], p['throughput'], s=85, marker=markers[engine], edgecolor=COLORS[engine], linewidth=1.8,
                       facecolor=COLORS[engine] if p['own_frontier'] else 'white', zorder=4,
                       label=LABELS[engine] if i == 0 else None)
            # C1 is explained in the table; avoid overlapping labels.
            if p['C'] != 1:
                offset = (8, 9) if p['C'] != 64 else (8, -16)
                ax.annotate(f"C{p['C']}", (p['latency_ms'], p['throughput']), xytext=offset, textcoords='offset points', color=COLORS[engine])
    ax.set_xscale('log'); ax.set_xlim(550, 28000); ax.set_ylim(0, 210)
    ax.set_xticks([600, 1000, 2000, 5000, 10000, 20000]); ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.set_xlabel('Median complete-request latency (ms; logarithmic scale; lower is better)', labelpad=11)
    ax.set_ylabel('Total output throughput (tokens/s/Mac; higher is better)', labelpad=10)
    ax.grid(which='major', color='#E4E7E9');ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(-.005, 1.13), ncol=4, frameon=False)
    front=', '.join('C'+str(p['C']) for p in data['points'] if p['engine']=='native' and p['joint_frontier'])
    fig.text(.085,.035,f'Joint measured frontier: minivllm {front}. Large marks: medians; faint marks: all 48 process results.\n'
        'Filled marks: within-engine frontier. Lines connect measured frontier configurations only; no interpolation claim or confidence intervals.\n'
        'Each point: 3 fresh processes × 3 measured groups after warmup. Group latency is request p50, then median across groups/processes.\n'
        'External oMLX/Lithos points are archived earlier runs. Lithos: C1 only. MTP/DSpark and long-context results are not yet included.',size=10.5,linespacing=1.5,color='#50575E')
    fig.savefig(out/'throughput-latency-pareto.png',dpi=180)
    fig.savefig(out/'throughput-latency-pareto.svg',metadata={'Date':None})
    plt.close(fig)
    return matplotlib.__version__


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=DEFAULT);ap.add_argument('--snapshot',type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.snapshot:
        manifest=json.loads((args.snapshot.parent/'source-manifest.json').read_text())
        for meta in manifest.values(): assert digest(args.snapshot/meta['snapshot_path'])==meta['sha256']
    sources=Sources(args.output,args.snapshot);data=collect(sources);write(args.output/'chartdata.json',data)
    relative='bench/m5_ultra_256gb/progress/latency_pareto.py';copy=args.output/'sources'/relative;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(__file__,copy)
    sources.manifest[relative]={'snapshot_path':relative,'sha256':digest(copy),'bytes':copy.stat().st_size}
    write(args.output/'source-manifest.json',sources.manifest)
    version=render(data,args.output)
    write(args.output/'summary.json',{'status':'RECOMPUTED_FROM_HTTP_TIMESTAMPS','matplotlib':version,'processes':48,'measured_requests':data['request_count'],
          'coverage':data['coverage'],'source_files':len(sources.manifest),'joint_frontier':[{k:p[k] for k in ('engine','C','latency_ms','throughput')} for p in data['points'] if p['joint_frontier']]})
    table='\n'.join(f"| {LABELS[p['engine']]} | {p['C']} | {p['latency_ms']:.1f} | {p['throughput']:.2f} | {p['ttft_ms']:.1f} | {'是' if p['joint_frontier'] else '否'} |" for p in data['points'])
    (args.output/'README.md').write_text(f'''# Throughput–latency Pareto · 2026-10-17 v3

横轴为真实 **HTTP 请求开始到完整响应完成的延迟**，单位 ms；纵轴为总输出吞吐 tok/s。越靠左上越好。正确性是进入候选集合的前提，不是可牺牲的优化轴。

![Throughput–latency Pareto](throughput-latency-pareto.png)

[SVG](throughput-latency-pareto.svg) · [逐请求重算数据](chartdata.json) · [SHA](source-manifest.json)

| 引擎 | 并发 C | 完整请求延迟 ms | 总吞吐 tok/s | 首文本 TTFT ms | 联合前沿 |
|---|---:|---:|---:|---:|---|
{table}

全部48个进程、{data['request_count']}条正式请求从原始timestamps重算。每组延迟取request p50，每进程取三个正式组的中位数，每配置再取三个fresh进程的中位数。吞吐为组输出token总数除以最早开始到最晚完成的窗口，随后按相同层次聚合。不是用旧交付速率取倒数，也没有把TTFT当作完整请求延迟。原始逐请求记录和全部进程点均保存，没有删去回退点；小点不是置信区间。

限定同机Qwen3.8-27B NVFP4、19输入/32输出、greedy、normal EOS、无prefix cache、target-only。native是E010最终核心，外部为此前归档的oMLX/Lithos实测；本图没有重新跑外部引擎。Lithos当前只有C1点，MTP/DSpark不在此性能图。长上下文与GLM尚无这组证据。

正确性沿用已有资格：native在相同输入和固定调度下logits/cache/输出精确一致；HTTP自由到达可导致不同batch划分，不能声称跨异步进程或跨框架全部token逐位一致。每请求的usage、正常结束、32个visible IDs及native raw IDs可用时的roundtrip均重新核对；详细数值资格见E010归档。已知数值失败或尚无HTTP资格的实验模块不进入这张图。

离线重画（Python + Matplotlib 3.9.4；输出必须是新目录）：

```sh
python3 bench/m5_ultra_256gb/progress/latency_pareto.py --snapshot docs/optimization-guide/progress/2026-10-17-v3/sources --output /tmp/minivllm-latency-redraw
```

旧图保留。本快照把之前的交付速率横轴改为实际请求latency，并按 latency越小、throughput越大的规则重新计算Pareto支配关系。
''')
    print(json.dumps({'out':str(args.output),'processes':48,'requests':data['request_count'],'points':data['points']},ensure_ascii=False))


if __name__=='__main__': main()
