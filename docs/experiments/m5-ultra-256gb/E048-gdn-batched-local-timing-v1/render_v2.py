"""Reproducible static E048 local layer ratio figure; audited data only."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

AUDIT_SHA='22141e12f3eda94f762ec77fbbe51d87c825853b62152510dee3985055b8ba1a'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def render(audit_path,out):
    assert sha(audit_path)==AUDIT_SHA
    data=json.loads(audit_path.read_bytes())
    assert data['status']=='PASS' and data['local_timing_screen_audited'] is True
    assert (data['case_count'],data['group_count'],data['timing_count'])==(12,60,1200)
    assert all(data[k] is False for k in ('full_model_qualified','http_qualified','performance_qualified','engine_pareto_point'))
    rows=data['cases'];assert len({r['case_id'] for r in rows})==12
    for row in rows:
        assert [g['group'] for g in row['groups']]==list(range(5))
        assert row['median_group_ratio']==statistics.median(g['native_over_candidate'] for g in row['groups'])
        assert all(g['native_over_candidate']==g['median_ns']['native']/g['median_ns']['candidate'] for g in row['groups'])
    out.mkdir(exist_ok=False,parents=True)
    chart=dict(source=dict(path=str(audit_path),sha256=AUDIT_SHA),cases=rows,all_groups_displayed=60,
        ratio='median(native ns in group) / median(candidate ns in group)',median='Median of five group ratios',
        scope='Local eager GDN layer, warm resident inputs/weights; not engine throughput or Pareto.')
    (out/'chartdata.json').write_text(json.dumps(chart,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelcolor':'#252a31',
        'text.color':'#252a31','axes.edgecolor':'#9da3aa','xtick.color':'#4b535c','ytick.color':'#4b535c',
        'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(14.8,7.6),sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.27,top=.76,wspace=.09)
    colors={2:'#3273a8',8:'#a96c12'};offsets=[-.22,-.11,0,.11,.22]
    for ax,batch in zip(axes,(2,8)):
        chosen=[r for r in rows if r['batch_size']==batch]
        assert [(r['decode_step'],r['layer']) for r in chosen]==[(s,l) for s in (1,129) for l in (0,32,62)]
        for x,row in enumerate(chosen):
            ys=[g['native_over_candidate'] for g in row['groups']]
            ax.scatter([x+j for j in offsets],ys,s=52,facecolor='white',edgecolor=colors[batch],linewidth=1.5,zorder=3)
            ax.plot([x-.27,x+.27],[row['median_group_ratio']]*2,color='#252a31',lw=2.1,zorder=4)
            ax.scatter([x],[row['median_group_ratio']],marker='D',s=44,color='#252a31',zorder=5)
            ax.text(x,1.177,f"{row['median_group_ratio']:.3f}×",ha='center',va='center',fontsize=12,fontweight='bold')
        ax.axhline(1,color='#4b535c',lw=1.3,ls='--',zorder=1)
        ax.axvline(2.5,color='#c7ccd1',lw=1)
        ax.set(xlim=(-.55,5.55),ylim=(.85,1.20),xticks=range(6),
               xticklabels=['L0\nstep 1','L32\nstep 1','L62\nstep 1','L0\nstep 129','L32\nstep 129','L62\nstep 129'])
        ax.set_title(f'Actual tensor batch B{batch}',loc='left',fontsize=15,pad=17,fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.2f}×'))
        ax.set_yticks([.85,.90,.95,1,1.05,1.10,1.15,1.20]);ax.grid(axis='y',color='#e4e6e9',lw=.7)
        ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',length=0,pad=10)
        ax.set_xlabel('Layer and captured decode step',labelpad=14)
    axes[0].set_ylabel('Speed ratio: native / candidate\n>1 means candidate is faster',labelpad=12)
    axes[1].annotate('Group 0: 0.881×\nretained',xy=(.78,.8806996139009446),xytext=(2.15,.917),
        arrowprops={'arrowstyle':'-','color':'#6a7178','lw':1},ha='left',va='center',fontsize=11)
    fig.text(.075,.945,'E048 · Local eager GDN layer timing',fontsize=21,fontweight='bold')
    fig.text(.075,.902,'Qwen3.8-27B NVFP4 · M5 Ultra 256 GB · warm resident microbenchmark · NOT an engine Pareto point',fontsize=12)
    legend=[Line2D([0],[0],marker='o',markerfacecolor='white',markeredgecolor='#68727d',linestyle='None',label='All 5 group ratios (left to right: group 0–4)'),
            Line2D([0],[0],marker='D',color='#252a31',lw=2,label='Median of 5 groups; labels above'),
            Line2D([0],[0],color='#4b535c',ls='--',label='Parity 1×')]
    fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.068,.865),frameon=False,ncol=3,fontsize=11)
    fig.text(.075,.135,'Each group: 10 alternating native/candidate pairs; 5 warmups per arm precede measurement. All 1,200 arm durations retained.',fontsize=11)
    fig.text(.075,.098,'Timing includes forward output + conv/S materialization and synchronization; seed reset is outside timing. Spread is not a confidence interval.',fontsize=11)
    fig.text(.075,.061,'Independent original-byte/timing audit PASS: 48 files, 72 output/state comparisons. No full-model, compiled-route, HTTP or frontier qualification.',fontsize=11)
    fig.text(.075,.027,'Source: gdn-batched-timing-v1-independent.json · SHA-256 22141e12…b8ba1a · 2026-10-19',fontsize=10,color='#68727d')
    for suffix in ('png','svg'):fig.savefig(out/f'gdn-local-timing-ratios.{suffix}',dpi=160,facecolor='white')
    plt.close(fig)
    receipt=dict(status='RENDERED_NOT_YET_VISUALLY_REVIEWED',audit_sha256=AUDIT_SHA,groups_plotted=60,cases_plotted=12,
        renderer_sha256=sha(Path(__file__)),images={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.glob('gdn-local-timing-ratios.*')})
    (out/'render-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.audit is None and a.out is None:print('Inert: --audit frozen.json --out fresh-directory required.')
    else:assert a.audit is not None and a.out is not None;render(a.audit.resolve(),a.out.resolve())
