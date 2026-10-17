#!/usr/bin/env python3
"""Render the frozen E018 screen + three fresh-process confirmation results.

CPU-only; imports Matplotlib only after collecting and validating JSON.
The prior 2026-10-17-v1 snapshot is never modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / 'docs/optimization-guide/progress/2026-10-17-v2'
RAW = '.artifacts/m5-ultra-256gb'
LENGTHS = (1, 3, 5, 8)
COLORS = {'reference': '#356F9F', 'candidate': '#C67424'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


class Sources:
    def __init__(self, output, snapshot=None):
        self.output, self.snapshot, self.manifest = output, snapshot, {}

    def read(self, original):
        relative = original.replace('.artifacts/', 'raw-artifacts/', 1)
        source = (self.snapshot / relative) if self.snapshot else (ROOT / original)
        target = self.output / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        self.manifest[original] = {'snapshot_path': relative, 'sha256': digest(target), 'bytes': target.stat().st_size}
        return json.loads(target.read_text())


def collect(sources):
    processes = []
    for i, folder in enumerate(['gdn-verify-v1'] + [f'gdn-verify-repeat-v1-run{n}' for n in (1, 2, 3)]):
        base = f'{RAW}/{folder}'
        plan = sources.read(f'{base}/plan.json')
        summary = sources.read(f'{base}/summary.json')
        qualification = sources.read(f'{base}/qualification.json')
        child = sources.read(f'{base}-owner/child.json')
        assert plan['lengths'] == list(LENGTHS) and plan['pool_limit_bytes'] == 2 * 1024**3
        assert plan['warmups_per_arm'] == 2 and plan['abba_blocks'] == 3
        assert plan['gate']['T3_T5_T8_geomean_speedup_min'] == 1.25
        assert plan['gate']['T1_speedup_min'] == plan['gate']['no_T_regression_below'] == .97
        assert plan['identity']['mlx'] == '0.32.3'
        assert plan['identity']['device']['device_name'] == 'Apple M5 Ultra'
        assert summary['qualification_passed'] is True and len(qualification) == summary['qualification_cases'] == 12
        assert {(c['state'], c['T']) for c in qualification} == {(state, t) for state in ('real', 'zero', 'independent_random') for t in LENGTHS}
        count = 0
        for case in qualification:
            assert case['passed'] is True and case['initial_state_unchanged'] is True
            t = case['T']
            assert {c['label'] for c in case['comparisons']} == ({f'output-{n}' for n in range(t)} |
                    {f'history-{n}' for n in range(t)} | {f'independent-prefix-keep-{n}' for n in range(t + 1)})
            for comparison in case['comparisons']:
                assert comparison['exact'] is True
                assert comparison['reference_sha256'] == comparison['candidate_sha256']
                count += 1
        assert count == summary['exact_comparisons'] == 165
        points = []
        for t in LENGTHS:
            original = f'{base}/timing-T{t}.json'
            row = sources.read(original)
            assert row == next(r for r in summary['timing'] if r['T'] == t)
            assert [(r['block'], r['arm']) for r in row['rounds']] == [
                (b, arm) for b in range(3) for arm in ('reference', 'candidate', 'candidate', 'reference')]
            assert all(math.isfinite(r['ms']) and r['ms'] > 0 for r in row['rounds'])
            medians = {arm: statistics.median(r['ms'] for r in row['rounds'] if r['arm'] == arm) for arm in COLORS}
            for arm in COLORS:
                same(medians[arm], row['medians_ms'][arm])
                assert len(row['warmups_ms'][arm]) == 2
            ratio = medians['reference'] / medians['candidate']
            same(ratio, row['speedup'])
            points.append({'T': t, 'median_ms': medians, 'speedup': ratio,
                           'formal_samples': row['rounds'], 'first_ms': row['first_post_qualification_ms'],
                           'warmup_ms': row['warmups_ms'], 'source': original,
                           'source_pointer': '/rounds'})
        gm = statistics.geometric_mean(p['speedup'] for p in points if p['T'] in (3, 5, 8))
        same(gm, summary['T3_T5_T8_geomean_speedup'])
        gate = gm >= 1.25 and all(p['speedup'] >= .97 for p in points)
        assert summary['advance'] is gate
        worst = min(points, key=lambda p: p['speedup'])
        processes.append({'label': 'Initial screen' if i == 0 else f'Fresh repeat {i}',
            'phase': 'screen' if i == 0 else 'confirmation', 'pid': child['pid'],
            'exact_comparisons': count, 'geomean_T3_T5_T8': gm, 'gate': gate,
            'worst_T': worst['T'], 'worst_speedup': worst['speedup'], 'points': points,
            'source_plan': f'{base}/plan.json', 'scope': plan['scope']})
    dspark = sources.read('docs/experiments/m5-ultra-256gb/E017-lithos-dspark-v1/summary.json')
    assert dspark['status'] == 'PASS_INDEPENDENT_LITHOS_DSPARK_QUALIFICATION_AUDIT'
    assert dspark['requests'] == 6 and dspark['raw_ids_compared'] == 192
    return {'date': '2026-10-17', 'experiment': 'E018', 'processes': processes,
            'all_fresh_confirmation_gates_pass': all(p['gate'] for p in processes[1:]),
            'numeric_comparisons': sum(p['exact_comparisons'] for p in processes),
            'formal_samples': 192, 'warmup_samples': 64, 'first_samples': 32,
            'scope': 'One GDN layer helper: literal serial per-token helper vs same arithmetic fused across T. Not old chunk-WY equivalence, whole-model inference, rollback, or HTTP.',
            'E017': {'status': dspark['status'], 'raw_ids_compared': 192, 'requests': 6,
                     'no_speed_claim': True, 'scope': dspark['scope']}}


def render(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12,
        'axes.labelsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#858B91',
        'text.color': '#242A30', 'axes.labelcolor': '#242A30', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'svg.fonttype': 'none', 'svg.hashsalt': 'minivllm-progress-2026-10-17-v2'})
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey='row')
    fig.subplots_adjust(left=.055, right=.98, top=.75, bottom=.21, hspace=.34, wspace=.19)
    fig.suptitle('E018 · GDN verify helper across four independent processes', x=.055, y=.97,
                 ha='left', fontsize=22, weight='bold')
    passed = data['all_fresh_confirmation_gates_pass']
    fig.text(.055, .925, f"165 exact comparisons × 4 = {data['numeric_comparisons']}; fresh confirmation gates: " +
             ' / '.join('PASS' if p['gate'] else 'FAIL' for p in data['processes'][1:]) +
             f" → overall {'PASS' if passed else 'FAIL'}. Single-layer helper only; no HTTP claim.", fontsize=12)
    handles = [Line2D([], [], color=COLORS['reference'], marker='o', label='Reference: per-token serial helper'),
               Line2D([], [], color=COLORS['candidate'], marker='s', label='Candidate: fused T-token serial chain'),
               Line2D([], [], color='#888E94', marker='.', lw=0, label='Small dots: every formal observation (no CI)')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.05, .892), ncol=3, frameon=False, fontsize=11)
    max_ms = max(r['ms'] for p in data['processes'] for point in p['points'] for r in point['formal_samples'])
    ratios = [point['speedup'] for p in data['processes'] for point in p['points']]
    ratio_low, ratio_high = min(.80, min(ratios) - .12), max(1.75, max(ratios) + .15)
    for col, proc in enumerate(data['processes']):
        top, bottom = axes[0,col], axes[1,col]
        label = 'PASS' if proc['gate'] else 'FAIL'
        top.set_title(f"{proc['label']} · {label}\nGM₃,₅,₈ {proc['geomean_T3_T5_T8']:.3f}× · worst T{proc['worst_T']} {proc['worst_speedup']:.3f}×",
                      fontsize=11.5, pad=11)
        for arm, color in COLORS.items():
            for p in proc['points']:
                top.scatter([p['T']] * 6, [r['ms'] for r in p['formal_samples'] if r['arm'] == arm],
                            color=color, alpha=.32, s=17, edgecolors='none')
            top.plot(LENGTHS, [p['median_ms'][arm] for p in proc['points']], color=color,
                     marker='o' if arm=='reference' else 's', ms=5.5, lw=1.5)
        top.set_ylim(0, max_ms * 1.12)
        bottom.plot(LENGTHS, [p['speedup'] for p in proc['points']], color=COLORS['candidate'], marker='D', ms=6, lw=1.5)
        bottom.axhline(1, color='#41474D', lw=1)
        bottom.axhline(.97, color='#737980', ls=':', lw=1.4)
        for p in proc['points']:
            offset = (4, -16) if p['T'] == 1 and p['speedup'] < 1 else (0, 9)
            bottom.annotate(f"{p['speedup']:.3f}×", (p['T'], p['speedup']), xytext=offset,
                            textcoords='offset points', ha='left' if offset[0] else 'center', fontsize=10,
                            color='#AD526C' if p['speedup'] < 1 else '#30363C')
        bottom.set_ylim(ratio_low, ratio_high)
        for ax in (top,bottom):
            ax.set_xticks(LENGTHS);ax.set_xlim(.5,8.5)
            ax.grid(axis='y',color='#E5E7EA',linewidth=.7);ax.set_axisbelow(True)
            ax.set_xlabel('Verify length T (tokens)');ax.tick_params(labelbottom=True)
        if col == 0:
            top.set_ylabel('Wall time (ms; lower is faster)')
            bottom.set_ylabel('Reference / candidate (×)')
    fig.text(.055, .025,
        'Shared scales within each row. Top: process medians plus all 192 formal timings; 3 ABBA rounds = six observations per arm and T. Bottom: ratio of medians.\n'
        'One screen + three predeclared fresh confirmations; no pooling, confidence intervals, or selective reruns. Gate per process: GM(T3,T5,T8) ≥1.25× AND every T ≥0.97× (dotted line).\n'
        'Pinned M5 Ultra / MLX 0.32.3 / 2 GiB pool; one GDN layer, B1/H48/D128. Real inputs captured from fixed Qwen target; model released before micro timing.\n'
        'Exact output, per-token state history, all kept prefixes, and unchanged initial state. Literal helper arithmetic only: old chunk-WY equivalence and end-to-end speculation remain unqualified.',
        fontsize=10.5,color='#50575E',linespacing=1.5)
    fig.savefig(output/'e018-helper-progress.png',dpi=180)
    fig.savefig(output/'e018-helper-progress.svg',metadata={'Date':None})
    plt.close(fig)
    return matplotlib.__version__


def readme(data, output):
    table='\n'.join(f"| {p['label']} | {p['pid']} | " + ' | '.join(f"{v['speedup']:.6f}×" for v in p['points']) +
        f" | {p['geomean_T3_T5_T8']:.6f}× | {'PASS' if p['gate'] else 'FAIL'} |" for p in data['processes'])
    decision = '三个 fresh confirmation 全部达到独立预注册门，可进入下一层资格；不是生产采用。' if data['all_fresh_confirmation_gates_pass'] else '至少一个 fresh confirmation 未达独立预注册门；保留全部结果，不授予下一层采用资格。'
    (output/'README.md').write_text(f"""# Progress snapshot v2 · E018 · 2026-10-17

**{decision}** 初筛和三次 fresh process 均独立展示；165×4 = **660** 个数值比较通过。没有池化进程中位数，也未筛掉负收益。

![E018 single-layer helper](e018-helper-progress.png)

[可编辑 SVG](e018-helper-progress.svg) · [图数据](chartdata.json) · [来源 SHA](source-manifest.json) · [摘要](summary.json)

| Process | PID | T1 | T3 | T5 | T8 | GM(T3,T5,T8) | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
{table}

每个进程的门是 GM(T3,T5,T8)≥1.25× 且每个 T≥0.97×，T1 的负收益原样保留。图顶排显示真实 ms，底排 reference/candidate 倍数；横轴为实际 T1/3/5/8，同行共用尺度。每点每臂六个正式观测，三组 ABBA；四进程共 192 个正式样本，64 warm、32 首样本全部随 chartdata 保存。小点是观察样本，不是独立进程或置信区间；连接线只辅助阅读。

本实验只比较 **单层 GDN helper**：原 per-token literal helper 串行调用，对比在一个 kernel 中保持同样串行数学的 T-token chain。设备为 M5 Ultra / MLX0.32.3 / 2 GiB allocator pool，shape B1/H48/D128。模型只用于抓取真实输入，micro timing 前已释放。三个初态 real/zero/independent_random 分别检查 output、每 token state history、所有可保留 prefix，以及初始 state 不变。

**没有证明旧 chunk-WY 路径等价、完整模型正确性、target rollback、投机生成或 HTTP 性能。** 不把本图倍数乘到 E010/E012/E014，不作全网最快结论。

E017 Lithos DSpark 已通过正式路线与返回 IDs 资格：两个 fresh processes、同一 19-token prompt 各三请求，共比较 192 raw IDs；这次 **没有速度结论**，见 [E017 归档](../../../experiments/m5-ultra-256gb/E017-lithos-dspark-v1/README.md)。[v1 的七张进展图](../2026-10-17-v1/README.md) 保持冻结，包含 E014 core 采用失败等完整历史。

## 离线重画

用 Python + Matplotlib 3.9.4（本机独立 CPU venv `.artifacts/m5-ultra-256gb/progress-render-venv`），仓库根运行：

```sh
python3 bench/m5_ultra_256gb/progress/gdn.py
python3 bench/m5_ultra_256gb/progress/gdn.py --snapshot docs/optimization-guide/progress/2026-10-17-v2/sources --output /tmp/minivllm-e018-rebuild
```

生成器重算每点六样本中位数、speedup、GM 和每个独立门；检查全部 660 个 strict 字段及两臂 SHA 一致，再画图。`sources/raw-artifacts/` 保存原 JSON 的副本（避免被 `.artifacts` ignore 规则排除），source-manifest 保留原路径和副本映射。此图核验不替代正式实验的 source/resource/owner 审计。
""")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot',type=Path)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.snapshot:
        pinned=json.loads((args.snapshot.parent/'source-manifest.json').read_text())
        for original,meta in pinned.items():
            assert digest(args.snapshot/meta['snapshot_path']) == meta['sha256'],original
    sources=Sources(args.output,args.snapshot)
    data=collect(sources);write(args.output/'chartdata.json',data)
    relative='bench/m5_ultra_256gb/progress/gdn.py';copy=args.output/'sources'/relative
    copy.parent.mkdir(parents=True,exist_ok=True)
    if Path(__file__).resolve()!=copy.resolve():shutil.copyfile(__file__,copy)
    sources.manifest[relative]={'snapshot_path':relative,'sha256':digest(copy),'bytes':copy.stat().st_size}
    write(args.output/'source-manifest.json',sources.manifest)
    version=render(data,args.output);readme(data,args.output)
    summary={'status':'CHART_NUMERIC_RECONCILIATION_PASS','matplotlib':version,
        'source_files':len(sources.manifest),'numeric_comparisons':data['numeric_comparisons'],
        'formal_samples':192,'warmup_samples':64,'first_samples':32,
        'process_gates':[p['gate'] for p in data['processes']],
        'all_fresh_confirmation_gates_pass':data['all_fresh_confirmation_gates_pass'],
        'visual_inspection':'Recorded separately after view_image', 'no_HTTP_or_whole_model_claim':True}
    write(args.output/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
