"""Standalone E055 paired C8 figure. Inert unless --render is explicit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

PHASES=('r1-c8-on','r1-c8-off','r2-c8-off','r2-c8-on','r3-c8-on','r3-c8-off')
METRICS=('aggregate_output_tok_s','e2e_p50_s','e2e_p95_s')
AUDITOR_SHA='79a22462ce49584b483cdea53cc5c7255e8721464cd497c3a4cdaa0896ad72c8'


def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(value):return type(value) in (int,float) and math.isfinite(value) and value>0


def prepare(path,digest):
    need(sha(path)==digest,'explicit closed audit SHA')
    a=json.loads(path.read_bytes())
    need(a['schema']=='E055-six-process-metrics-ids-v2' and a['status']=='PASS_METRICS_AND_IDS'
         and a['audit_source_sha256']==AUDITOR_SHA,'actual registered six-process timing audit')
    need(a['timing_valid'] is a['same_schedule_numerical_prerequisite'] is True,'valid timing and E053 prerequisite')
    need(a['performance_qualified'] is a['formal_full_state_replayed'] is False
         and a['diagnostic_throughput_only'] is True,'retain audit qualification limits')
    points=a['points'];c=a['comparison']
    need([p['phase'] for p in points]==list(PHASES) and a['actual_request_count']==432,'all six closed processes /432 requests')
    for p in points:
        need(p['concurrency']==8 and p['requests']==64 and p['arm'] in ('off','on')
             and p['repeat'] in (1,2,3) and all(finite(p[k]) for k in METRICS),'actual finite C8 process points')
    pairs=[]
    for repeat,first,second,checked in zip((1,2,3),points[::2],points[1::2],c['pairs']):
        arms={p['arm']:p for p in (first,second)}
        need(set(arms)=={'off','on'} and first['repeat']==second['repeat']==repeat,'adjacent C8 pair')
        ratios=dict(throughput_ratio=arms['on'][METRICS[0]]/arms['off'][METRICS[0]],
                    p50_ratio=arms['on'][METRICS[1]]/arms['off'][METRICS[1]],
                    p95_ratio=arms['on'][METRICS[2]]/arms['off'][METRICS[2]])
        need(checked['repeat']==repeat and checked['order']==[first['arm'],second['arm']]
             and all(checked[k]==v for k,v in ratios.items()),'recomputed paired metrics')
        pairs.append(dict(repeat=repeat,order=checked['order'],off=arms['off'],on=arms['on'],**ratios,
            throughput_gain_percent=100*(ratios['throughput_ratio']-1),p50_reduction_percent=100*(1-ratios['p50_ratio'])))
    need(len(c['pairs'])==len(pairs)==3,'three pairs')
    for key,value in c['median_pair_ratios'].items():
        need(value==statistics.median(p[key] for p in pairs),'paired median recheck')
    summaries={}
    for arm in ('off','on'):
        summaries[arm]={}
        for key in METRICS:
            values=[p[key] for p in points if p['arm']==arm]
            stats=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
            need(stats==c['process_summary'][arm][key],'per-arm process summary recheck');summaries[arm][key]=stats
    need(type(a['cross_schedule_token_equal']) is bool,'separate cross-schedule ID status')
    return dict(schema='E055-paired-performance-plot-data-v1',audit=dict(path=str(path),sha256=digest),
        plot_source_sha256=sha(__file__),points=points,pairs=pairs,process_summary=summaries,
        speed_gate_passed=c['speed_gate_passed'],off_rate_spread_fraction=c['off_rate_spread_fraction'],
        timing_valid=True,same_schedule_numerical_prerequisite=True,
        cross_schedule_token_equal=a['cross_schedule_token_equal'],cross_schedule_differences=a['cross_schedule_differences'],
        performance_qualified=False,formal_full_state_replayed=False,frontier_point=False,
        scope='C8 only; three adjacent OFF/ON fresh-process pairs. No current comparator runs, TPOT/P90/cost or concurrency frontier.')


def render(data,out):
    out.mkdir(parents=True,exist_ok=False);os.environ.setdefault('MPLCONFIGDIR',str(out/'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'white','svg.fonttype':'none','svg.hashsalt':'E055-C8-pairs-v1'})
    colors={'off':'#576A80','on':'#15806E'};xkey='e2e_p50_s';ykey='aggregate_output_tok_s'
    fig,(ax,ratio)=plt.subplots(1,2,figsize=(13.4,7.5),gridspec_kw={'width_ratios':[1.6,1]})
    fig.subplots_adjust(left=.09,right=.97,bottom=.30,top=.77,wspace=.30)
    for pair in data['pairs']:
        off,on=pair['off'],pair['on'];start=(off[xkey],off[ykey]);end=(on[xkey],on[ykey])
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#8A97A5',lw=1.2),zorder=2)
        for arm,p in (('off',off),('on',on)):
            ax.scatter(p[xkey],p[ykey],marker='o' if arm=='off' else '^',s=48,color=colors[arm],zorder=4)
            ax.annotate('R'+str(pair['repeat']),(p[xkey],p[ykey]),xytext=(-5,8+10*(pair['repeat']-1)) if arm=='off' else (5,-14-10*(pair['repeat']-1)),
                textcoords='offset points',ha='right' if arm=='off' else 'left',fontsize=8,color=colors[arm])
    for arm,summary in data['process_summary'].items():
        x,y=summary[xkey],summary[ykey]
        ax.errorbar(x['median'],y['median'],xerr=[[x['median']-x['minimum']],[x['maximum']-x['median']]],
            yerr=[[y['median']-y['minimum']],[y['maximum']-y['median']]],fmt='D',ms=9,mfc=colors[arm],mec='white',
            color=colors[arm],elinewidth=1.1,capsize=3,zorder=5)
    for values,set_limits in (([p[xkey] for p in data['points']],ax.set_xlim),([p[ykey] for p in data['points']],ax.set_ylim)):
        span=max(max(values)-min(values),max(values)*.015);set_limits(max(0,min(values)-span*.35),max(values)+span*.35)
    ax.set_xlabel('Full HTTP E2E p50 latency (seconds; lower is better)',labelpad=10)
    ax.set_ylabel('Aggregate committed output throughput (tokens/s)',labelpad=10)
    ax.set_title('C8 detail · axes zoomed · better toward upper left',loc='left',fontsize=11,pad=12)
    ratio.axhline(0,color='#677687',lw=.9,ls='--')
    ratio.axhline(3,color='#B84437',lw=1,ls='--',label='3% adoption threshold')
    for key,color,label,offset in (('throughput_gain_percent','#15806E','Throughput gain',-.09),
                                   ('p50_reduction_percent','#745D9C','E2E p50 reduction',.09)):
        values=[p[key] for p in data['pairs']];xs=[1+offset,2+offset,3+offset]
        ratio.plot(xs,values,'o-',ms=6,lw=1.2,color=color,label=label)
        median=statistics.median(values);ratio.axhline(median,color=color,lw=1,ls=':',alpha=.8)
        ratio.annotate(f'Median {median:+.2f}%',(3+offset,median),xytext=(-3,10 if offset<0 else -16),
            textcoords='offset points',ha='right',fontsize=9,color=color)
    ratio.set_xticks((1,2,3),('R1\nON→OFF','R2\nOFF→ON','R3\nON→OFF'))
    ratio.set_xlim(.65,3.4);ratio.set_ylabel('Change versus paired OFF (%) · higher is better')
    ratio.set_title('Three process pairs · actual execution order below',loc='left',fontsize=10,pad=12)
    ratio.legend(loc='best',frameon=False,fontsize=9)
    for panel in (ax,ratio):panel.grid(color='#E3E8EE',lw=.7);panel.set_axisbelow(True)
    fig.suptitle('C8 batched GDN: measured OFF → ON pairs',x=.09,ha='left',y=.96,fontsize=20,fontweight='bold')
    fig.text(.09,.91,'M5 Ultra · Qwen3.8-27B NVFP4 · graph ON in both arms · observers OFF · MB50',fontsize=11,color='#526173')
    fig.legend(handles=[Line2D([],[],marker='o',color=colors['off'],ls='none',label='GDN OFF process'),
        Line2D([],[],marker='^',color=colors['on'],ls='none',label='Batched GDN ON process'),
        Line2D([],[],marker='D',color='#526173',ls='none',label='Per-arm metric median; whiskers = process min–max')],
        loc='upper left',bbox_to_anchor=(.083,.88),frameon=False,ncol=3,fontsize=9)
    token='identical' if data['cross_schedule_token_equal'] else 'different greedy trajectories (not an arithmetic-error finding)'
    gate='PASS' if data['speed_gate_passed'] else 'NOT PASSED'
    lines=[f'6 fresh processes: 3 adjacent pairs, each 8 warm + 64 formal requests. Arrows connect OFF→ON within a pair, not concurrency settings.',
        f'Timing audit valid; preregistered speed gate: {gate}. Cross-schedule returned IDs: {token}.',
        'E053 proves arithmetic for its captured history; these formal schedules were not full-state replayed. Timing and numerical scope stay separate.',
        'One HTTP concurrency (C8), not a frontier. Process spread is not a confidence interval. No current oMLX/Lithos, TPOT, P90 or cost comparison.',
        'Source: closed six-process audit '+data['audit']['sha256'][:12]+' · exact points and qualification flags retained in plot-data.json.']
    for y,line in zip((.19,.15,.11,.07,.03),lines):fig.text(.09,y,line,fontsize=8.5,color='#526173')
    for ext in ('png','svg'):fig.savefig(out/('c8-paired-performance.'+ext),dpi=180,metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig);(out/'plot-data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='RENDERED_PENDING_VISUAL_REVIEW',directory=str(out))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--render',action='store_true');p.add_argument('--audit',type=Path)
    p.add_argument('--audit-sha256');p.add_argument('--out-dir',type=Path);args=p.parse_args()
    if not args.render:
        need(not any((args.audit,args.audit_sha256,args.out_dir)),'explicit --render required')
        print('SOURCE_READY_ONLY: no data read or plotting imports.');return
    need(all((args.audit,args.audit_sha256,args.out_dir)),'closed audit SHA and fresh output directory required')
    render(prepare(args.audit.resolve(),args.audit_sha256),args.out_dir.resolve())


if __name__=='__main__':main()
