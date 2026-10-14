"""InferenceX 式的 Pareto 曲线：x = 每用户 tok/s（交互速度，decode），y = 整机 tok/s（吞吐）。
读 bench/results/pareto-ours.txt、pareto-mlxlm.txt、spec_check_v3.txt，写 docs/pareto.html（自包含，带 hover 和表格）。"""
import json, re
from pathlib import Path

R = Path("bench/results")
ours = []
for line in (R / "pareto-ours.txt").read_text().splitlines():
    m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
    if m:
        B, total, out_tps, dec, ttft, peak = m.groups(); B = int(B); dec = float(dec)
        ours.append({"label": f"{B}", "B": B, "x": dec / B, "y": dec, "note": f"峰值 {float(peak):.1f} GB，TTFT {float(ttft):.0f} s"})
mlx = []; mlx_fail = None
for line in (R / "pareto-mlxlm.txt").read_text().splitlines():
    m = re.match(r"b=(\d+) .*generation_tps=([\d.]+).*peak_memory=([\d.]+)", line)
    if m:
        B, g, peak = int(m.group(1)), float(m.group(2)), float(m.group(3))
        mlx.append({"label": f"{B}", "B": B, "x": g / B, "y": g, "note": f"峰值 {peak:.1f} GB"})
    elif line.startswith("b="):
        mlx_fail = int(line.split()[0][2:])
spec = []
names = {"math": "数学", "code": "代码", "kv": "短中文"}
txt = (R / "spec_check_v3.txt").read_text()
blk5 = txt.split("--- block 5")[-1]
for m in re.finditer(r"dflash (\w+)\s*:\s*([\d.]+) tok/s decode-only \(([\d.]+)x\)", blk5):
    k, v, sp = m.group(1), float(m.group(2)), m.group(3)
    spec.append({"label": names.get(k, k), "B": 1, "x": v, "y": v, "note": f"{sp}×，块 5，无损"})
# mlx-lm 真正的 server（continuous batching）在并发 8 的 decode 阶段：TPOT 165 ms → 每用户 6.1，整机 48.5（tutorial 05）。
mlx_server = [{"label": "它的 server，并发 8", "B": 8, "x": 1000 / 165, "y": 8 * 1000 / 165, "note": "mlx_lm.server，decode 阶段；并发 32 OOM", "hollow": True}]
# dflash-mlx 在同样三个 prompt 上（块 5，128 token，27B-4bit，它默认开 thinking）：bench/results/dflash-mlx-3prompts.txt
dm = {}
for line in (R / "dflash-mlx-3prompts.txt").read_text().splitlines():
    if line.startswith("=== "): cur = line[4:].strip()
    m = re.search(r"dflash generation_tps median\s*:\s*([\d.]+)", line) or re.search(r"dflash ([\d.]+) speedup", line)
    if m: dm[cur] = float(m.group(1))
dflash_mlx = [{"label": names.get(k, k), "B": 1, "x": v, "y": v, "note": "dflash-mlx，块 5，4bit affine" + ("；它开了 thinking，答案长得多" if k == "kv" else ""), "diamond": True, "hollow": True} for k, v in dm.items()]
for p in spec: p["diamond"] = True
data = {"series": [
    {"name": "minivllm（引擎，并发 1 到 32）", "slot": 1, "points": ours, "connect": True},
    {"name": "minivllm + DFlash2（单用户，无损）", "slot": 1, "points": spec, "connect": False},
    {"name": "mlx-lm（静态 batch；空心 = 它的 server）", "slot": 2, "points": mlx + mlx_server, "connect": True, "fail": mlx_fail},
    {"name": "dflash-mlx（单用户）", "slot": 2, "points": dflash_mlx, "connect": False},
]}
html = r'''<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pareto 曲线</title>
<style>
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f0efec;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e6e5e1;
  --series-1:#2a78d6;--series-2:#8a8984;--series-3:#1baf7a;font-family:-apple-system,"PingFang SC","Helvetica Neue",Arial,sans-serif;
  background:var(--surface-1);color:var(--text-primary);padding:16px;max-width:900px;margin:0 auto}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#383835;
  --text-primary:#fff;--text-secondary:#c3c2b7;--grid:#2e2e2c;--series-1:#3987e5;--series-2:#9c9b95;--series-3:#199e70}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#383835;--text-primary:#fff;--text-secondary:#c3c2b7;--grid:#2e2e2c;
  --series-1:#3987e5;--series-2:#9c9b95;--series-3:#199e70}
body{margin:0;background:#fcfcfb}@media (prefers-color-scheme:dark){body{background:#1a1a19}}
h1{font-size:18px;margin:0 0 4px;font-weight:600}.sub{font-size:13px;color:var(--text-secondary);margin:0 0 12px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:0 0 6px}.legend span{display:inline-flex;align-items:center;gap:6px}
.key{width:18px;height:2px;background:currentColor;display:inline-block;position:relative}.key::after{content:"";position:absolute;left:5px;top:-3px;width:8px;height:8px;border-radius:50%;background:currentColor}
.key.diamond{background:none}.key.diamond::after{border-radius:0;transform:rotate(45deg);left:5px;top:-4px}
svg{width:100%;height:auto;display:block}
.grid line{stroke:var(--grid);stroke-width:1}.axis text{fill:var(--text-secondary);font-size:12px}.axis line,.axis path{stroke:var(--text-secondary);stroke-width:1}
.axis .title{font-size:13px;fill:var(--text-primary)}
.pt-label{fill:var(--text-secondary);font-size:11px}.ring{fill:var(--surface-1)}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--grid);border-radius:6px;padding:8px 10px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;max-width:260px}
.tip b{font-size:14px}.tip .k{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px}
details{margin-top:14px;font-size:13px}table{border-collapse:collapse;margin-top:8px}td,th{padding:4px 10px;border-bottom:1px solid var(--grid);text-align:right}th:first-child,td:first-child{text-align:left}
.foot{font-size:12px;color:var(--text-secondary);margin-top:10px}
</style></head><body><div class="viz-root">
<h1>单用户开 DFlash2 快 2.2 倍，和 dflash-mlx 打平；并发 16 以内和 mlx-lm 的 kernel 打平，24 以上它快 20%，但它的 server 跑不到那里</h1>
<p class="sub">Qwen3.8-27B nvfp4，M4 Max 48 GB。每个点是一个并发数：x = 每用户 decode tok/s（交互速度），y = 整机 decode tok/s（吞吐），右上越好。1024 token 进、128 出，decode 阶段持续值。画法参考 InferenceX。</p>
<div class="legend" id="legend"></div>
<div id="chart"></div><div class="tip" id="tip"></div>
<details><summary>数据表</summary><div id="table"></div></details>
<p class="foot">蓝 = 这个引擎，灰 = 基线；圆点 = 合批扫描（随机 token 的 prompt），菱形 = 单用户投机解码（真实 prompt，块 5）。mlx-lm 用 <code>mlx_lm.benchmark</code> 静态 batch，它的 server 在并发 32 时 Metal OOM；dflash-mlx 跑的是同样三个 prompt，但它默认开 thinking，短中文那条答案长得多，不可直接比。</p>
</div>
<script>
const DATA = __DATA__;
const W=860,H=520,ML=64,MR=24,MT=16,MB=52;
const pts=DATA.series.flatMap(s=>s.points);
const xmax=Math.ceil(Math.max(...pts.map(p=>p.x))/10)*10+5, ymax=Math.ceil(Math.max(...pts.map(p=>p.y))/20)*20+10;
const sx=x=>ML+(x/xmax)*(W-ML-MR), sy=y=>MT+(1-y/ymax)*(H-MT-MB);
const ns="http://www.w3.org/2000/svg", el=(t,a={})=>{const e=document.createElementNS(ns,t);for(const k in a)e.setAttribute(k,a[k]);return e};
const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"Pareto 曲线"});
const grid=el("g",{class:"grid"}),axis=el("g",{class:"axis"});
const ystep=ymax>200?50:20, xstep=xmax>80?20:10;
for(let y=0;y<=ymax;y+=ystep){grid.appendChild(el("line",{x1:ML,x2:W-MR,y1:sy(y),y2:sy(y)}));const t=el("text",{x:ML-8,y:sy(y)+4,"text-anchor":"end"});t.textContent=y;axis.appendChild(t)}
for(let x=0;x<=xmax;x+=xstep){const t=el("text",{x:sx(x),y:H-MB+18,"text-anchor":"middle"});t.textContent=x;axis.appendChild(t)}
axis.appendChild(el("line",{x1:ML,x2:W-MR,y1:sy(0),y2:sy(0)}));axis.appendChild(el("line",{x1:ML,x2:ML,y1:MT,y2:sy(0)}));
const xt=el("text",{x:(ML+W-MR)/2,y:H-10,"text-anchor":"middle",class:"title"});xt.textContent="每用户 tok/s（交互速度）";axis.appendChild(xt);
const yt=el("text",{x:14,y:(MT+sy(0))/2,"text-anchor":"middle",class:"title",transform:`rotate(-90 14 ${(MT+sy(0))/2})`});yt.textContent="整机 tok/s（吞吐）";axis.appendChild(yt);
svg.appendChild(grid);svg.appendChild(axis);
const tip=document.getElementById("tip");
DATA.series.forEach(s=>{
  const col=`var(--series-${s.slot})`;const g=el("g");
  const ps=[...s.points].sort((a,b)=>a.B-b.B);
  const solid=ps.filter(p=>!p.hollow);
  if(s.connect&&solid.length>1){g.appendChild(el("path",{d:solid.map((p,i)=>(i?"L":"M")+sx(p.x)+" "+sy(p.y)).join(" "),fill:"none",stroke:col,"stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"}))}
  ps.forEach((p,i)=>{
    g.appendChild(el("circle",{cx:sx(p.x),cy:sy(p.y),r:6.5,class:"ring"}));
    let dot;
    if(p.diamond){dot=el("rect",{x:sx(p.x)-5,y:sy(p.y)-5,width:10,height:10,transform:`rotate(45 ${sx(p.x)} ${sy(p.y)})`,fill:p.hollow?"var(--surface-1)":col,stroke:col,"stroke-width":2})}
    else{dot=p.hollow?el("circle",{cx:sx(p.x),cy:sy(p.y),r:4.5,fill:"var(--surface-1)",stroke:col,"stroke-width":2}):el("circle",{cx:sx(p.x),cy:sy(p.y),r:4.5,fill:col})}
    g.appendChild(dot);
    const want=s.connect?(s.slot===1?["1","2","4","8","16","32"].includes(p.label):(p.hollow||p.label==="32")):true;
    if(want){const dy=s.connect?((p.label==="1"&&!p.hollow)||p.hollow?16:-8):(p.hollow?16:-8);let dx=(s.connect&&["16","24","32"].includes(p.label))?20:9;
      const left=!s.connect&&p.hollow&&p.y<55; if(left)dx=-9;
      const lab=el("text",{x:sx(p.x)+dx,y:sy(p.y)+dy,class:"pt-label","text-anchor":left?"end":"start"});
      lab.textContent=(s.connect&&!p.hollow?"并发 ":"")+p.label;g.appendChild(lab)}
    const hit=el("circle",{cx:sx(p.x),cy:sy(p.y),r:14,fill:"transparent",tabindex:0});
    const show=ev=>{tip.style.display="block";tip.innerHTML="";const b=document.createElement("b");b.textContent=`${p.y.toFixed(1)} tok/s 整机 · ${p.x.toFixed(1)} tok/s 每用户`;
      const d=document.createElement("div");const k=document.createElement("span");k.className="k";k.style.background=col;d.appendChild(k);d.appendChild(document.createTextNode(`${s.name.split("（")[0]} · ${s.connect&&!p.hollow?"并发 "+p.B:p.label}`));
      const n=document.createElement("div");n.textContent=p.note||"";tip.appendChild(b);tip.appendChild(d);tip.appendChild(n);
      const r=hit.getBoundingClientRect();tip.style.left=Math.min(window.innerWidth-280,r.right+8)+"px";tip.style.top=Math.max(8,r.top-10)+"px";dot.setAttribute("r",6)};
    const hide=()=>{tip.style.display="none";dot.setAttribute("r",4.5)};
    hit.addEventListener("pointerenter",show);hit.addEventListener("focus",show);hit.addEventListener("pointerleave",hide);hit.addEventListener("blur",hide);g.appendChild(hit);
  });
  if(s.fail&&ps.length){const last=ps[ps.length-1];const t=el("text",{x:sx(last.x)-9,y:sy(last.y)+28,class:"pt-label","text-anchor":"end"});t.textContent=`并发 ${s.fail}：Metal OOM`;g.appendChild(t)}
  svg.appendChild(g);
  const li=document.createElement("span");li.style.color=col;const key=document.createElement("i");key.className=s.connect?"key":"key diamond";li.appendChild(key);
  const nm=document.createElement("span");nm.style.color="var(--text-primary)";nm.textContent=s.name;li.appendChild(nm);document.getElementById("legend").appendChild(li);
});
document.getElementById("chart").appendChild(svg);
const tbl=document.createElement("table");const hr=tbl.insertRow();["系列","并发 / 场景","每用户 tok/s","整机 tok/s","备注"].forEach(h=>{const th=document.createElement("th");th.textContent=h;hr.appendChild(th)});
DATA.series.forEach(s=>[...s.points].sort((a,b)=>a.B-b.B).forEach(p=>{const r=tbl.insertRow();[s.name.split("（")[0],s.connect&&!p.hollow?p.B:p.label,p.x.toFixed(1),p.y.toFixed(1),p.note||""].forEach(v=>{r.insertCell().textContent=v})}));
document.getElementById("table").appendChild(tbl);
</script></body></html>'''
Path("docs/pareto.html").write_text(html.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
print("docs/pareto.html written;", {s["name"]: len(s["points"]) for s in data["series"]}, "mlx fail at", mlx_fail)
