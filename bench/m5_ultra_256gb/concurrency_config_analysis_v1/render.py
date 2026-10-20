"""Render a closed E045 eight-request screen; default inert, never engine Pareto.

Run only after the frozen independent raw-wire auditor has finished. No backend
or service access. The two panels use independently focused scales; C is HTTP
concurrency, not GPU batch. Lines are label leaders only, never interpolation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

AUDITOR_SHA = '17b28a40adc01bb793d3fdcaabbf7697f7432a601e637d55faf9d1555e5379d0'
LABELS = ('c1-start50', 'c1-mb200', 'c1-mb16', 'c1-mb800', 'c1-end50',
          'c8-start50', 'c8-mb800', 'c8-mb16', 'c8-mb200', 'c8-end50')
STYLES = {16: ('#246b9d', 's'), 50: ('#596675', 'o'),
          200: ('#a86818', 'D'), 800: ('#a74f87', '^')}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_audit(path):
    raw = path.read_bytes()
    audit = json.loads(raw)
    need(audit['schema'] == 'E045-independent-screen-audit-v1'
         and audit['status'] in ('PASS_SCREEN_ONLY', 'FAIL_TOKEN_EQUIVALENCE')
         and audit['errors'] == [] and audit['cycle_closed'] is True,
         'A complete closed-cycle independent audit is required.')
    need(audit['audit_source']['sha256'] == AUDITOR_SHA,
         'Unexpected independent auditor source.')
    need(audit['diagnostic_only'] is True and audit['requests_reparsed'] == 160
         and all(audit[k] is False for k in ('full_state_qualified',
             'numerical_qualified', 'performance_qualified',
             'resources_restoration_independently_audited_here')),
         'Preserve the diagnostic-only scope.')
    need(set(audit['points']) == set(audit['token_checks']) == set(LABELS),
         'Require all ten observed settings, including both bookends.')
    rows = []
    for name in LABELS:
        point = audit['points'][name]
        C = int(name[1])
        mb = 50 if '50' in name else int(name.split('mb')[1])
        need(point['concurrency'] == C and point['requests'] == 8
             and point['committed_output_tokens'] == 8 * 512,
             '512/p50 requires eight complete 512-token outputs.')
        for key in ('aggregate_output_tok_s', 'e2e_p50_s'):
            need(type(point[key]) in (int, float)
                 and math.isfinite(point[key]) and point[key] > 0,
                 'Invalid plotted metric: ' + key)
        anchors = [audit['points'][f'c{C}-{end}50'] for end in ('start', 'end')]
        rank = audit['ranking'][str(C)]
        token_ok = audit['token_checks'][name]['equal_to_same_C_start50_warm']
        rows.append(dict(label=name, concurrency=C, mb=mb,
            display=f'{mb} MB' + (' (start)' if 'start' in name else ' (end)' if 'end' in name else ''),
            x_512_over_p50_tok_s_user=512 / point['e2e_p50_s'],
            aggregate_output_tok_s=point['aggregate_output_tok_s'], e2e_p50_s=point['e2e_p50_s'],
            exact_returned_ids=token_ok, bookends_stable=rank['bookends_stable'],
            timing_bookends_stable=all(v <= .02 for v in rank['bookend_drift'].values()),
            worst_throughput_ratio=min(point['aggregate_output_tok_s'] / a['aggregate_output_tok_s'] for a in anchors),
            worst_p50_speed_ratio=min(a['e2e_p50_s'] / point['e2e_p50_s'] for a in anchors),
            eligible_for_confirmation=name in rank['ranked_for_confirmation']))
    return audit, rows, raw


def render(audit_path, out):
    audit, rows, raw = read_audit(audit_path)
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/minivllm-plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'svg.hashsalt': 'E045-screen-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 10.6))
    fig.subplots_adjust(left=.085, right=.97, top=.79, bottom=.43, wspace=.27)
    for ax, C, entry in zip(axes, (1, 8), ('graph + GDN', 'graph only')):
        subset = [r for r in rows if r['concurrency'] == C]
        xs = [r['x_512_over_p50_tok_s_user'] for r in subset]
        ys = [r['aggregate_output_tok_s'] for r in subset]
        dx = max(max(xs) - min(xs), max(xs) * .01)
        dy = max(max(ys) - min(ys), max(ys) * .01)
        ax.set_xlim(min(xs) - dx * .22, max(xs) + dx * .85)
        ax.set_ylim(min(ys) - dy * .23, max(ys) + dy * .23)
        ax.set_title(f'HTTP C{C} · {entry}', loc='left', fontsize=14, y=1.10, pad=0)
        ax.set_xlabel('512 / median end-to-end latency (tok/s/user)', labelpad=10)
        ax.set_ylabel('Aggregate committed output (tok/s)', labelpad=10)
        ax.grid(color='#e2e8f0', linewidth=.7)
        ax.set_axisbelow(True)
        ax.ticklabel_format(style='plain', useOffset=False)
        # Fixed separate label slots retain even nearly coincident bookends.
        for index, row in enumerate(sorted(subset, key=lambda r: r['aggregate_output_tok_s'], reverse=True)):
            color, marker = STYLES[row['mb']]
            if 'end' in row['label']:
                marker = 'X'
            if not row['exact_returned_ids']:
                marker = 'x'
            x, y = row['x_512_over_p50_tok_s_user'], row['aggregate_output_tok_s']
            ax.scatter([x], [y], s=85, marker=marker, color=color, zorder=4)
            label = row['display'] + ('' if row['exact_returned_ids'] else ' [IDs FAIL]')
            ax.annotate(label, xy=(x, y), xytext=(.985, .91 - index * .195),
                textcoords='axes fraction', ha='right', va='center', color=color, fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', pad=2),
                arrowprops=dict(arrowstyle='-', color=color, linewidth=.65), zorder=5)
        drift = audit['ranking'][str(C)]['bookend_drift']
        ax.text(0, 1.035, f"50-MB drift: throughput {drift['aggregate_output_tok_s']:.2%}, p50 {drift['e2e_p50_s']:.2%}",
                transform=ax.transAxes, fontsize=10, color='#596675')
    fig.text(.085, .947, 'E045 · C1 / C8 command-buffer configuration screen', fontsize=20, weight='bold', color='#172536')
    fig.text(.085, .905, '8 measured requests + 8 warm requests per fresh process · screen only, NOT a formal Pareto frontier', fontsize=12, color='#8a4d18')
    fig.text(.085, .864, 'MLX_MAX_MB_PER_BUFFER = 16 / 50 / 200 / 800; 50 MB repeated before and after each C. Other settings fixed.', fontsize=11, color='#475569')
    table_ax = fig.add_axes([.085, .15, .885, .225]); table_ax.axis('off')
    table_rows = []
    for row in rows:
        baseline = '50' in row['label']
        decision = 'baseline' if baseline else 'confirm next' if row['eligible_for_confirmation'] else 'not selected'
        if not row['exact_returned_ids']:
            decision = 'IDs FAIL'
        elif not row['timing_bookends_stable']:
            decision += '; drift gate fails'
        elif not row['bookends_stable']:
            decision = 'baseline IDs differ'
        table_rows.append([f"C{row['concurrency']} · {row['display']}",
            f"{row['aggregate_output_tok_s']:.3f}", f"{row['e2e_p50_s']:.4f}",
            '—' if baseline else f"{row['worst_throughput_ratio']:.4f}×",
            '—' if baseline else f"{row['worst_p50_speed_ratio']:.4f}×", decision])
    table = table_ax.table(cellText=table_rows, colLabels=['Configuration', 'Output tok/s', 'p50 E2E (s)',
        'Throughput ratio¹', 'p50 speed ratio¹', 'Screen decision'],
        colWidths=[.22, .13, .13, .16, .16, .20], cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(9.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#dfe5eb'); cell.set_linewidth(.45)
        if r == 0:
            cell.set_facecolor('#eef2f6'); cell.set_text_props(weight='bold')
    fig.text(.085, .112, '¹ Minimum ratio against both 50-MB bookends. Selection requires both ratios ≥1.03 and both bookend drifts ≤2%, plus exact returned IDs.', fontsize=10, color='#475569')
    fig.text(.085, .077, 'Axes use separate focused scales; label leaders are not performance curves. 512/p50 is latency-derived, not P90 interactivity or measured ITL.', fontsize=10, color='#475569')
    fig.text(.085, .042, f"Audit: {audit['status']}. C8 baseline also differs between runs; cause is unclassified. No full-state or engine-frontier claim.", fontsize=10, color='#475569')
    (out / 'audit-source.json').write_bytes(raw)
    (out / 'chartdata.json').write_text(json.dumps(dict(rows=rows, source_path=str(audit_path),
        source_sha256=sha(raw), schema='E045-screen-chart-v1', engine_pareto_point=False), indent=2, allow_nan=False) + '\n')
    for ext in ('png', 'svg'):
        fig.savefig(out / f'concurrency-config-screen.{ext}', dpi=170,
            metadata={'Date': None} if ext == 'svg' else None)
    plt.close(fig)
    receipt = dict(status='RENDERED', visual_review='PENDING', audit_path=str(audit_path),
        audit_sha256=sha(raw), render_source_sha256=sha(Path(__file__).read_bytes()),
        artifacts={p.name: dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in sorted(out.iterdir())})
    need(audit_path.read_bytes() == raw, 'Audit changed during render.')
    (out / 'render-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], out=str(out), visual_review='PENDING')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.audit is None and args.out is None:
        print('Inert: pass --audit closed-independent-audit.json --out fresh-directory after the GPU cycle closes.')
        return 0
    need(args.audit is not None and args.out is not None, 'Supply --audit and --out together.')
    render(args.audit.resolve(), args.out.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
