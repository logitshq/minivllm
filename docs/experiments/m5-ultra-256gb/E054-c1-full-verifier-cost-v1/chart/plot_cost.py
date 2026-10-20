"""Standalone E054 local-cost figure; explicit audited input required to render."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    def pairs(items):
        value={}
        for key,item in items:
            need(key not in value,'duplicate JSON field');value[key]=item
        return value
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def data_from_audit(path,digest):
    need(sha(path)==digest,'explicit closed audit SHA')
    audit=read(path)
    need(audit['schema']=='E054-local-cost-independent-v2' and audit['status']=='PASS'
         and audit['local_cost_audited'] is True and audit['timing_count']==400 and audit['group_count']==20,'complete E054 local-cost audit')
    need(audit['audit_source_sha256']=='483fac0a92befa1d946a9b7dd75a578f0564a167e175f29661da6cf81847afcb','corrected v2 auditor source')
    need(all(audit[k] is False for k in ('http_qualified','performance_qualified','engine_pareto_point')),'retain local-only scope')
    ledger=Path(audit['child'])/'timings.jsonl'
    need(sha(ledger)==audit['source_and_small_evidence'][str(ledger)],'audit-bound raw timing ledger')
    with ledger.open() as stream:rows=[json.loads(line) for line in stream]
    need(len(rows)==400,'all400 original timing rows')
    plotted=[];consumed=0
    for case,(width,baseline) in zip(audit['timings'],((4,'device'),(4,'host'),(8,'device'),(8,'host'))):
        need((case['T'],case['baseline'])==(width,baseline),'four ordered actual timing cases')
        subset=rows[consumed:consumed+100];consumed+=100;groups=[]
        for g in range(5):
            native=[];candidate=[];ratios=[]
            for pair in range(10):
                actual=subset[g*20+pair*2:g*20+pair*2+2]
                order=['native','candidate'] if (g*10+pair)%2==0 else ['candidate','native']
                for ordinal,row in enumerate(actual):
                    need((row['T'],row['baseline'],row['group'],row['pair'],row['ordinal'],row['arm'])
                         ==(width,baseline,g,pair,ordinal,order[ordinal]) and row['order']==order,'chronological paired sample')
                    need(type(row['elapsed_ns']) is int and row['elapsed_ns']>0,'positive integer elapsed duration')
                times={r['arm']:r['elapsed_ns'] for r in actual}
                native.append(times['native']);candidate.append(times['candidate']);ratios.append(times['native']/times['candidate'])
            checked=case['summary']['groups'][g]
            need(checked==dict(group=g,median_paired_ratio=statistics.median(ratios),ratios=ratios),'audited pair ratio recheck')
            groups.append(dict(group=g+1,native_ms=[v/1e6 for v in native],verifier_ms=[v/1e6 for v in candidate],
                native_median_ms=statistics.median(native)/1e6,verifier_median_ms=statistics.median(candidate)/1e6))
        need(case['summary']['median_group_ratio']==statistics.median(g['median_paired_ratio'] for g in case['summary']['groups']),'audited median group ratio recheck')
        for arm in ('native','candidate'):
            values=sorted(r['elapsed_ns'] for r in subset if r['arm']==arm)
            need(case['summary']['totals_ns'][arm]==dict(p50=statistics.median(values),p90=values[math.ceil(.9*len(values))-1]),'audited total median/p90 recheck')
        if baseline=='device':plotted.append(dict(T=width,groups=groups,median_paired_native_over_verifier=case['summary']['median_group_ratio']))
    need(consumed==400 and len(plotted)==2 and len(audit['timings'])==4,'exact T4/T8 device-only selection')
    return dict(schema='E054-plot-data-v1',audit=dict(path=str(path),sha256=digest),
        raw_timings=dict(path=str(ledger),sha256=sha(ledger)),plot_source_sha256=sha(__file__),cases=plotted,
        plotted_timing_calls=200,total_audited_timing_calls=400,group_statistic='median of10 calls per arm;5 groups per T',
        summary_statistic='median of5 group medians; paired ratios separately use audited median-of-group-median pair ratios',
        local_only=True,drafting_measured=False,http_measured=False,pareto_point=False)


def render(data,out):
    out.mkdir(parents=True,exist_ok=False);os.environ.setdefault('MPLCONFIGDIR',str(out/'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#252A31','text.color':'#252A31','xtick.color':'#424953','ytick.color':'#424953',
        'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'})
    colors=('#2166AC','#D88822');jitter=(-.14,-.07,0,.07,.14)
    maximum=max(g[key] for c in data['cases'] for g in c['groups'] for key in ('native_median_ms','verifier_median_ms'))
    fig,axes=plt.subplots(1,2,figsize=(11.8,7),sharey=True)
    fig.subplots_adjust(left=.095,right=.98,bottom=.31,top=.79,wspace=.17)
    for ax,case in zip(axes,data['cases']):
        for offset,g in zip(jitter,case['groups']):
            values=[g['native_median_ms'],g['verifier_median_ms']]
            ax.plot([offset,1+offset],values,color='#C6CBD1',lw=.8,zorder=1)
            for x,y,color in zip((offset,1+offset),values,colors):
                ax.scatter(x,y,s=48,color=color,edgecolor='white',linewidth=.7,zorder=3)
                ax.annotate(str(g['group']),(x,y),xytext=(0,-13),textcoords='offset points',ha='center',fontsize=8,color='#545C66')
        for x,key in enumerate(('native_median_ms','verifier_median_ms')):
            median=statistics.median(g[key] for g in case['groups'])
            ax.hlines(median,x-.22,x+.22,color='#252A31',lw=1.6,zorder=4)
            ax.scatter(x,median,s=66,marker='D',color='#252A31',edgecolor='white',linewidth=.7,zorder=5)
            ax.text(x,median+maximum*.06,f'{median:.2f} ms',ha='center',va='bottom',fontsize=12,fontweight='bold')
        ax.set_title(f'T = {case["T"]} proposed tokens',fontsize=14,pad=16)
        ax.set_xticks([0,1],['Native device\nAR(T)','Full verifier\n+ selected-state commit'])
        ax.set_xlim(-.40,1.40);ax.set_ylim(0,maximum*1.24);ax.grid(axis='y',color='#E4E7EB',lw=.7);ax.set_axisbelow(True)
        ax.text(.5,.97,f'Audited paired native / verifier: {case["median_paired_native_over_verifier"]:.3f}×',
            transform=ax.transAxes,ha='center',va='top',fontsize=10,color='#525B66')
    axes[0].set_ylabel('Time per T-token block (ms; lower is better)',labelpad=11)
    fig.suptitle('C1 local verifier and commit cost',x=.095,ha='left',y=.96,fontsize=21,fontweight='bold')
    fig.text(.095,.903,'Qwen3.8-27B NVFP4 · Apple M5 Ultra · fixed prefix position 1022 · k = T',fontsize=11,color='#525B66')
    fig.legend(handles=[Line2D([],[],marker='o',color='#888F98',linestyle='none',markersize=6,label='Group medians (groups 1–5)'),
        Line2D([],[],marker='D',color='#252A31',lw=1.6,markersize=6,label='Median of five group medians')],
        loc='lower center',bbox_to_anchor=(.53,.205),frameon=False,ncol=2,fontsize=10)
    fig.text(.095,.155,'Each group contains 10 alternating AB/BA pairs. Grey lines connect the same timing group; spread is not a confidence interval.',fontsize=9.5)
    fig.text(.095,.115,'Device baseline only: 200 displayed timed calls of 400 audited calls. Host-sequential baseline is excluded.',fontsize=9.5)
    fig.text(.095,.075,'Local cost, assuming all T tokens are useful. Drafting, acceptance/rejection, scheduler and HTTP costs are absent. No Pareto claim.',fontsize=9.5)
    fig.text(.095,.035,'Source: closed E054 independent audit '+data['audit']['sha256'][:12]+' · raw timing ledger '+data['raw_timings']['sha256'][:12],fontsize=8.5,color='#69737F')
    fig.savefig(out/'local-verifier-cost.png',dpi=180);fig.savefig(out/'local-verifier-cost.svg');plt.close(fig)
    (out/'plot-data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='RENDERED_PENDING_VISUAL_REVIEW',png=str(out/'local-verifier-cost.png'),svg=str(out/'local-verifier-cost.svg'))))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--render',action='store_true');p.add_argument('--audit',type=Path)
    p.add_argument('--audit-sha256');p.add_argument('--out-dir',type=Path);a=p.parse_args(argv)
    if not a.render:
        need(not any((a.audit,a.audit_sha256,a.out_dir)),'explicit --render required')
        print('SOURCE READY ONLY: requires closed audited E054. No result read or image generated.');return
    need(all((a.audit,a.audit_sha256,a.out_dir)),'explicit audited source SHA and fresh output directory required')
    render(data_from_audit(a.audit.resolve(),a.audit_sha256),a.out_dir.resolve())


if __name__=='__main__':main()
