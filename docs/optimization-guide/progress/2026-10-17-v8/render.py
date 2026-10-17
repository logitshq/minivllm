"""Render only audited D512 observations; no service or benchmark work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

NATIVE_SHA='b669350e95ef07f4da5cac3a5cf6fb6fdaf2aea57e886d3ba60b9afd15e950bd'
LITHOS_AUDITOR_SHA='768dc970dca201a700c94e07e17c5fdd399c903f444272ea150eb6336575c9c2'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)
def nondominated(point,points,x):
    return not any(q[x]<=point[x] and q['aggregate_output_tok_s']>=point['aggregate_output_tok_s']
        and (q[x]<point[x] or q['aggregate_output_tok_s']>point['aggregate_output_tok_s']) for q in points)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native','lithos','out'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();native=json.loads(args.native.read_bytes());lithos=json.loads(args.lithos.read_bytes())
    need(digest(args.native)==NATIVE_SHA and native['schema']=='E023-native-formal-independent-audit-v1','fixed historical native comparator')
    need(lithos['schema']=='E023-lithos-D512-formal-independent-audit-v1'
         and lithos['audit_source_sha256']==LITHOS_AUDITOR_SHA,'registered Lithos auditor')
    for a in (native,lithos):
        need(a['status']=='PASS' and a['qualified'] is True and a['performance_qualified'] is True
             and a['restoration']['status']=='PASS','closed independently audited performance only')
    need(lithos['resources']['independent_resource_verified'] is True and lithos['formal_requests']==384
         and lithos['warm_requests']==48 and len(lithos['points'])==6 and len(native['points'])==2,'complete fixed comparison')
    rows=[]
    for engine,audit,n in (('mini-vLLM',native,1),('Lithos',lithos,3)):
        for c in (1,8):
            selected=[q for q in audit['points'] if q['concurrency']==c]
            need(len(selected)==n and {q['repeat'] for q in selected}==set(range(1,n+1)),'fresh replicate coverage')
            for point in selected:
                need(point['requests']==64 and point['warm_requests_checked']==8
                     and point['aggregate_output_tok_s']==point['committed_output_tokens']/point['common_window_s']
                     and 0<point['e2e_p50_s']<=point['e2e_p95_s'],'point identity/units')
                rows.append(dict(engine=engine,**point))
    for row in rows:
        row['nondominated_p50_among_displayed_observations']=nondominated(row,rows,'e2e_p50_s')
        row['nondominated_p95_among_displayed_observations']=nondominated(row,rows,'e2e_p95_s')
    summaries=[]
    for engine in ('mini-vLLM','Lithos'):
        for c in (1,8):
            selected=[q for q in rows if q['engine']==engine and q['concurrency']==c]
            summaries.append(dict(engine=engine,concurrency=c,fresh_processes=len(selected),
                **{k:statistics.median(q[k] for q in selected) for k in ('aggregate_output_tok_s','e2e_p50_s','e2e_p95_s')},
                scope='Median of process metrics, not a pooled request percentile or CI.'))
    args.out.mkdir(parents=True,exist_ok=False)
    for name,path in (('native-independent-audit.json',args.native),('lithos-independent-audit.json',args.lithos),('render.py',Path(__file__))):
        shutil.copyfile(path,args.out/name)
    chart=dict(schema='D512-native-lithos-observed-pareto-v1',hardware='M5 Ultra 256 GiB',
        model='mlx-community/Qwen3.8-27B-nvfp4',revision='5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6',
        workload='Same eight D coding fixtures, 886–978 prompt tokens, greedy normal EOS, output cap512',
        sources={name:dict(path=str(path.resolve()),sha256=digest(path)) for name,path in (('native',args.native),('lithos',args.lithos))},
        points=rows,process_metric_medians=summaries,omlx_D512='pending',
        scope='Discrete observed p50/p95–throughput tradeoffs only. Native n=1/C is historical; Lithos n=3/C is a new cycle. No paired trial, CI, maximum-concurrency or global-frontier claim. Lithos C8 is HTTP concurrency with official FIFO GPU execution. Not a cross-framework full-state or output-quality test.')
    (args.out/'chartdata.json').write_text(json.dumps(chart,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
    fig.subplots_adjust(left=.085,right=.97,bottom=.31,top=.76,wspace=.19)
    fig.text(.085,.94,'Qwen3.8-27B · D512 throughput–latency',size=22,weight='bold',color='#142238')
    fig.text(.085,.89,'M5 Ultra 256 GiB · same NVFP4 weights and prompts · normal EOS · observer OFF',size=12,color='#475569')
    colors={'mini-vLLM':'#1267b1','Lithos':'#c2671b'};markers={'mini-vLLM':'o','Lithos':'^'}
    ymax=max(q['aggregate_output_tok_s'] for q in rows)*1.23
    for ax,x,label in zip(axes,('e2e_p50_s','e2e_p95_s'),('p50','p95')):
        ax.set_facecolor('white');ax.grid(color='#e3e8ef',linewidth=.8,zorder=0)
        ax.set_xlim(0,max(q[x] for q in rows)*1.17);ax.set_ylim(0,ymax)
        for q in rows:
            frontier=nondominated(q,rows,x)
            ax.scatter(q[x],q['aggregate_output_tok_s'],s=83,color=colors[q['engine']],marker=markers[q['engine']],
                edgecolor='#142238' if frontier else 'white',linewidth=1.2,zorder=4,alpha=.9)
        for q in summaries:
            selected=[r for r in rows if r['engine']==q['engine'] and r['concurrency']==q['concurrency']]
            # Text sits at the observed cluster; no fictitious median point is drawn.
            anchor=min(selected,key=lambda r:abs(r[x]-q[x]))
            offset=(8,-23 if q['engine']=='mini-vLLM' and q['concurrency']==1 else 13)
            ax.annotate(f"{q['engine']} C{q['concurrency']}",(anchor[x],anchor['aggregate_output_tok_s']),
                xytext=offset,textcoords='offset points',size=10,color=colors[q['engine']])
        ax.set_xlabel('Full POST-to-DONE '+label+' latency (s)  ← lower',labelpad=12)
        ax.xaxis.set_major_locator(MaxNLocator(6));ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.text(.025,.96,'Better toward upper left',transform=ax.transAxes,va='top',size=10,color='#64748b')
    axes[0].set_ylabel('Aggregate committed output (tokens/s)  ↑ higher',labelpad=10)
    handles=[Line2D([],[],marker=markers[e],color='none',markerfacecolor=colors[e],markeredgecolor='white',markersize=9,
        label=e+(' · 1 fresh process / C' if e=='mini-vLLM' else ' · 3 fresh processes / C')) for e in colors]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.08,.845),frameon=False,ncol=2)
    for i,q in enumerate(summaries):
        x=.085+i*.225
        fig.text(x,.225,f"{q['engine']} C{q['concurrency']} · n={q['fresh_processes']}",weight='bold',color=colors[q['engine']],size=11)
        fig.text(x,.185,f"{q['aggregate_output_tok_s']:.2f} tok/s",size=14,color='#142238')
        fig.text(x,.151,f"p50 {q['e2e_p50_s']:.2f}s · p95 {q['e2e_p95_s']:.2f}s",size=10,color='#475569')
    fig.text(.085,.095,'Dots are individual processes; table values are process-metric medians. Black edges mark observed nondominated dots.',size=10,color='#475569')
    fig.text(.085,.059,'8 warm + 64 measured requests / process. No paired repeats or CI. oMLX D512 remains pending; C8 is not a measured hardware limit.',size=10,color='#475569')
    for suffix in ('png','svg'):
        target=args.out/('D512-native-lithos-pareto.'+suffix);fig.savefig(target,dpi=180)
        if suffix=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
    lines=['# D512 native and Lithos observed operating points','',chart['workload']+'.', '',chart['scope'],'',
        '| Engine | C | Repeat | Committed tokens | Window (s) | tok/s | p50 (s) | p95 (s) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for q in rows:lines.append(f"| {q['engine']} | {q['concurrency']} | {q['repeat']} | {q['committed_output_tokens']} | {q['common_window_s']:.6f} | {q['aggregate_output_tok_s']:.6f} | {q['e2e_p50_s']:.6f} | {q['e2e_p95_s']:.6f} |")
    lines+=['','Throughput = committed output / (last formal DONE − first formal POST). Latency includes server queueing and excludes waiting for a client slot. EOF/drain/close must pass but is outside the latency interval.',
        '', 'Black outlines identify nondominated points among these eight observed processes only. Repetition spread is not a confidence interval. No interpolated frontier is drawn. Median summaries do not invent new observed coordinates.',
        '', 'Native numerical eligibility covers representative recorded histories. Lithos eligibility binds actual prior full-input IDs, route/count/text/cache checks; formal wire has counts rather than raw IDs. This comparison does not establish identical logits, state or coding quality across implementations.',
        '', '[Data](chartdata.json) · [PNG](D512-native-lithos-pareto.png) · [SVG](D512-native-lithos-pareto.svg) · [Native audit](native-independent-audit.json) · [Lithos audit](lithos-independent-audit.json) · [Renderer](render.py)','']
    (args.out/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'RENDERED','points':len(rows),'out':str(args.out.resolve())}))

if __name__=='__main__':main()
