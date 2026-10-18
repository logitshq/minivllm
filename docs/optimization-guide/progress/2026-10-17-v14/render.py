"""Redraw v13's audited data with explicit empirical Pareto connectors; no new runs."""
from pathlib import Path
import argparse, hashlib, json, statistics

SOURCE_SHA = 'ff7d06adbe1ecda482173a6747a7e673aa71a73a54713bfd2d8f2f6b52551ef0'
COLORS = {'native-on':'#15806e','omlx':'#a13d73','lithos':'#ba6e20'}
LABELS = {'native-on':'minivllm (Native ON)','omlx':'oMLX','lithos':'Lithos'}
MARKERS = {'native-on':'D','omlx':'s','lithos':'^'}
XKEYS = ('e2e_p50_s','e2e_p95_s')
YKEY = 'aggregate_output_tok_s'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def frontier(rows, x):
    return sorted([r for r in rows if not any(q[x] <= r[x] and q[YKEY] >= r[YKEY]
        and (q[x] < r[x] or q[YKEY] > r[YKEY]) for q in rows)], key=lambda r:r[x])
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--repo',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=args.repo.resolve(); out=args.output.resolve()
    assert not out.exists(), 'Refuse to overwrite an existing render'
    out.mkdir(parents=True)
    src=root/'docs/optimization-guide/progress/2026-10-17-v13/chartdata.json'
    assert sha(src)==SOURCE_SHA
    data=json.loads(src.read_bytes()); assert len(data['points'])==26
    rows=[]
    for summary in data['process_metric_medians']:
        c=summary['configuration']; conc=summary['concurrency']
        actual=[data['points'][d['point_index']] for d in data['display'] if d['configuration']==c and data['points'][d['point_index']]['concurrency']==conc]
        assert len(actual)==summary['fresh_processes']
        row={'configuration':c,'concurrency':conc,'fresh_processes':len(actual),'ranges':{}}
        for key in (YKEY,*XKEYS):
            values=[r[key] for r in actual]; derived=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
            assert derived==summary['metrics'][key], (c,conc,key)
            row[key]=derived['median']; row['ranges'][key]=[derived['minimum'],derived['maximum']]
        rows.append(row)
    active=[r for r in rows if r['configuration'] in COLORS]
    fronts={x:{c:frontier([r for r in active if r['configuration']==c],x) for c in COLORS} for x in XKEYS}
    joint={x:frontier(active,x) for x in XKEYS}
    for x in XKEYS:
        assert [(r['configuration'],r['concurrency']) for r in joint[x]]==[('omlx',1),('native-on',8)]
        assert joint[x]==frontier(rows,x)
    report={'schema':'D512-explicit-frontier-v14','source':str(src),'source_sha256':SOURCE_SHA,
        'performance_points':26,'new_performance_points':0,'summary_rows':rows,'per_engine_frontiers':fronts,
        'joint_frontiers':joint,'normalized_x':'512 / median across processes of each process p50 end-to-end latency',
        'semantics':['Medians and ranges of process metrics, not pooled request percentiles or confidence intervals.',
        'Strict point-estimate nondominance, no significance threshold. Lithos C8 throughput exceeds C1 by only 0.00614 tokens/s.',
        'Segments connect measured summary points only; intermediate configurations are unmeasured and not predicted.',
        'The joint frontier switches engines; connecting endpoints does not make the segment an executable configuration.',
        'E2E reciprocal is not ITL or TPOT reciprocal; all retained requests actually output512 tokens.',
        'Native OFF/history are retained as grey observations and do not alter the joint median frontier.']}
    (out/'chartdata.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none','svg.hashsalt':'D512-frontier-v14'})
    handles=[Line2D([],[],color=COLORS[c],marker=MARKERS[c],lw=2.7,label=LABELS[c]) for c in COLORS]
    handles += [Line2D([],[],color='#1e293b',ls=(0,(5,4)),lw=2.2,label='Joint measured frontier'),
                Line2D([],[],color='#94a3b8',marker='o',ls='none',markersize=5,label='Other native configurations')]
    def draw(ax,x,normalized=False,annotate=True):
        fx=lambda r:512/r[x] if normalized else r[x]
        ax.set_axisbelow(True); ax.grid(color='#e2e8f0',lw=.75)
        for d in data['display']:
            p=data['points'][d['point_index']]; c=d['configuration']
            ax.scatter(fx(p),p[YKEY],s=21,color=COLORS.get(c,'#94a3b8'),alpha=.43,zorder=2,edgecolors='none')
        for c in COLORS:
            f=fronts[x][c]
            ax.plot([fx(r) for r in f],[r[YKEY] for r in f],color=COLORS[c],lw=2.7,zorder=3)
            for r in f:
                lo,hi=r['ranges'][x]; xl,xh=(512/hi,512/lo) if normalized else (lo,hi)
                yl,yh=r['ranges'][YKEY]
                ax.errorbar(fx(r),r[YKEY],xerr=[[fx(r)-xl],[xh-fx(r)]],yerr=[[r[YKEY]-yl],[yh-r[YKEY]]],
                    fmt=MARKERS[c],ms=8,color=COLORS[c],mec='white',mew=.7,elinewidth=1,capsize=3,zorder=6)
        f=joint[x]
        ax.plot([fx(r) for r in f],[r[YKEY] for r in f],color='#1e293b',ls=(0,(5,4)),lw=2.2,zorder=4)
        for r in f:
            ax.scatter(fx(r),r[YKEY],s=175,facecolors='none',edgecolors='#1e293b',lw=1.4,zorder=5)
        ax.set_ylim(0,195)
        ax.set_xlim((0,56) if normalized else (0,100))
        ax.set_ylabel('Aggregate output throughput (tokens/s)')
        if normalized:
            ax.set_xlabel('E2E normalized interactivity: 512 / process-median p50 latency (tokens/s)',labelpad=12)
            offsets={('native-on',1):(-114,-34),('omlx',1):(-70,18),('lithos',1):(-3,-42),
                     ('native-on',8):(12,16),('omlx',8):(13,-27),('lithos',8):(-9,16)}
            for r in active:
                c=r['configuration']; conc=r['concurrency']; off=offsets[(c,conc)]
                ax.annotate(f'{LABELS[c]} · C{conc}',(fx(r),r[YKEY]),xytext=off,textcoords='offset points',fontsize=10.5,
                    color=COLORS[c],arrowprops={'arrowstyle':'-','color':COLORS[c],'lw':.8},zorder=8)
            ax.text(.98,.97,'BETTER  ↗',transform=ax.transAxes,ha='right',va='top',color='#475569',weight='bold')
        else:
            metric='p50' if x==XKEYS[0] else 'p95'
            ax.set_xlabel(f'Full HTTP {metric} latency (seconds)  ← lower',labelpad=10)
            ax.text(.03,.96,'BETTER  ↖',transform=ax.transAxes,va='top',color='#475569',weight='bold')
            for r in active:
                if r['concurrency']==8:
                    c=r['configuration']; dy=14 if c=='native-on' else -24 if c=='omlx' else 14
                    ax.annotate(f'{LABELS[c]} C8',(fx(r),r[YKEY]),xytext=(8 if c!='lithos' else -80,dy),textcoords='offset points',color=COLORS[c],fontsize=9.5)
            inset=ax.inset_axes([.55,.49,.41,.40]); inset.set_facecolor('white'); inset.grid(color='#e2e8f0',lw=.6)
            for r in active:
                if r['concurrency']==1:
                    c=r['configuration']; inset.scatter(fx(r),r[YKEY],marker=MARKERS[c],color=COLORS[c],s=48)
            inset.set_xlim(9.9,11.75); inset.set_ylim(43.7,52.1); inset.tick_params(labelsize=8)
            inset.set_title('C1 zoom · measured medians',fontsize=9)
    def footer(fig):
        lines=['26 retained process observations; colored endpoints = median of 3 processes/configuration/C. Whiskers = observed min–max, not CI.',
               'Lines are guides between measured endpoints. Only C1 and C8 measured; intermediate configurations are NOT measured or predicted.',
               'Black dashed line joins the jointly nondominated endpoints across engines; it is not an executable mixed configuration.',
               'Same v13 data, no new benchmark. Fixed D prompts, normal EOS/cap512; Lithos C8 uses official FIFO. Native ON = E024 graph.']
        for y,t in zip((.139,.108,.077,.046),lines): fig.text(.07,y,t,fontsize=9,color='#475569')
    fig,ax=plt.subplots(figsize=(13.3,8.5));fig.subplots_adjust(left=.08,right=.96,bottom=.24,top=.79)
    fig.text(.08,.955,'M5 Ultra · measured Pareto frontiers',fontsize=23,weight='bold',color='#172536')
    fig.text(.08,.918,'Qwen3.8-27B NVFP4 · D512 · InferenceX-style orientation, using full-request speed',fontsize=12,color='#475569')
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.075,.885),ncol=3,frameon=False,fontsize=10.5)
    draw(ax,XKEYS[0],True);footer(fig)
    files=[]
    for ext in ('png','svg'):
        p=out/f'D512-frontier-interactivity.{ext}';fig.savefig(p,dpi=170,metadata={'Date':None} if ext=='svg' else None);files.append(p)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(16,8.5));fig.subplots_adjust(left=.06,right=.98,bottom=.24,top=.79,wspace=.22)
    fig.text(.06,.955,'M5 Ultra · measured throughput–latency frontiers',fontsize=22,weight='bold',color='#172536')
    fig.text(.06,.918,'Qwen3.8-27B NVFP4 · same audited D512 data · full HTTP request latency',fontsize=12,color='#475569')
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,.885),ncol=3,frameon=False,fontsize=10.5)
    for ax,x in zip(axes,XKEYS):draw(ax,x)
    footer(fig)
    for ext in ('png','svg'):
        p=out/f'D512-frontier-latency.{ext}';fig.savefig(p,dpi=170,metadata={'Date':None} if ext=='svg' else None);files.append(p)
    plt.close(fig)
    for p in files:
        if p.suffix == '.svg':
            p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    verification={'status':'RENDERED','source_sha256':SOURCE_SHA,'renderer_sha256':sha(Path(__file__)),
        'summary_medians_recomputed_from_all_26_processes':True,'joint_frontier_unchanged_with_off_and_history':True,
        'artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in [out/'chartdata.json',*files]},'visual_review':'pending'}
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({'status':'RENDERED','output':str(out),'images':[str(p) for p in files if p.suffix=='.png']}))
if __name__=='__main__':main()
