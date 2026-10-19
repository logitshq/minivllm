"""Plot retained scalar observations; no model/backend import or external state."""
import hashlib, json, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/minivllm-plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).parent
raw = (HERE/'observations.json').read_bytes(); data = json.loads(raw)
rows = data['rows']; assert len(rows) == 34 and rows[-1]['guard_status'] == 'REJECTED'
x = [r['elapsed_end_s'] for r in rows]; gib = 2**30
colors = {'blue': '#286a9c', 'pink': '#a13d73', 'gold': '#ac7628', 'gray': '#475569', 'red': '#9b3348'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'svg.hashsalt': 'E034-memory-v1'})
fig, axs = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
fig.subplots_adjust(left=.085, right=.96, top=.825, bottom=.18, hspace=.5, wspace=.25)
def draw(ax, values, label, color, marker='o'):
    ax.plot(x, values, label=label, color=color, lw=1.8, marker=marker, ms=3)
    ax.scatter(x[-1], values[-1], color=colors['red'], marker='x', s=70, zorder=5)
def vm(key): return [r['vm_stat']['pages'][key]*r['vm_stat']['page_size']/gib for r in rows]
a = axs[0,0]
draw(a,[float('nan') if r['child_footprint_bytes'] is None else r['child_footprint_bytes']/gib for r in rows], 'Child footprint', colors['blue'])
draw(a,[r['system_footprint_bytes']/gib for r in rows], 'System footprint', colors['pink'], 's')
a.set_title('Footprint accounting · separate, never summed', loc='left', fontsize=12); a.legend(frameon=False, fontsize=9)
a.set_ylabel('GiB'); a.set_ylim(0,240)
specs = [(axs[0,1],vm('Pages occupied by compressor'),'Compressor occupied · physical pages','GiB',colors['gold']),
    (axs[1,0],vm('Pages free'),'Free pages','GiB',colors['blue']),
    (axs[1,1],vm('File-backed pages'),'File-backed pages','GiB',colors['gray']),
    (axs[2,0],[r['swap_growth_bytes']/2**20 for r in rows],'Global swap used · change from cold baseline','MiB',colors['pink']),
    (axs[2,1],[r['swapouts_growth_pages'] for r in rows],'Global Swapouts · change from cold baseline','16 KiB pages',colors['gold'])]
for ax, values, title, unit, color in specs:
    draw(ax,values,title,color); ax.set_title(title,loc='left',fontsize=12); ax.set_ylabel(unit)
    ax.set_ylim(bottom=min(0,min(values))); ax.text(.025,.88,f'Last: {values[-1]:,.3f} {unit}' if unit!='16 KiB pages' else f'Last: {values[-1]:,} pages',transform=ax.transAxes,fontsize=10)
for ax in axs.flat:
    ax.grid(color='#e2e8f0',lw=.7); ax.set_axisbelow(True)
    ax.axvline(0,color='#94a3b8',ls=':',lw=1)
    ax.axvline(data['guard_rejection']['elapsed_s'],color=colors['red'],ls='--',lw=1)
    ax.set_xlim(min(x)-.6,max(x)+1.5)
for ax in axs[2]: ax.set_xlabel('Elapsed from official load start (s; cold samples < 0)')
fig.suptitle('GLM load stopped on swap growth — compression grew before the abort',x=.085,ha='left',y=.97,fontsize=19,weight='bold',color='#172536')
fig.text(.085,.928,'M5 Ultra · 256 GiB · GLM‑5.3‑Flash‑4bit · 34 recorded samples · FAILED load diagnostic, 0 requests',fontsize=12,color=colors['gray'])
fig.text(.085,.887,'Dashed red = swap guard rejection; red × = retained final observation. No physical readings are invented after this point.',fontsize=10,color=colors['gray'])
fig.text(.085,.115,'At rejection: child 182.446 GiB / system 203.920 GiB accounting; compressor occupied 108.604 GiB; free pages 0.091 GiB.',fontsize=10,color=colors['gray'])
fig.text(.085,.082,'MLX allocation counters and these VM categories measure different things. Do not add categories or infer headroom from 256 − footprint.',fontsize=10,color=colors['gray'])
fig.text(.085,.049,'Points are sample completion times; commands are sequential. Swap counters are global. This is not a Pareto point or a resident-fit proof.',fontsize=10,color=colors['gray'])
artifacts = {}
for ext in ('png','svg'):
    p = HERE/f'GLM-load-memory.{ext}'; assert not p.exists()
    fig.savefig(p,dpi=170,metadata={'Date':None} if ext=='svg' else None)
    artifacts[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
plt.close(fig)
report={'status':'RENDERED','visual_review':'PENDING','observations_sha256':hashlib.sha256(raw).hexdigest(),
    'render_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'artifacts':artifacts}
(HERE/'render-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
