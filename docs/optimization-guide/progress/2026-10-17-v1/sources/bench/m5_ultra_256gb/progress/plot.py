#!/usr/bin/env python3
"""Rebuild the dated M5 progress figures from retained JSON, with no MLX import.

All experiments remain separate; no sum/product of optimization speedups.
Use --snapshot to rebuild from the small source copies beside the figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-10-17"
EXP = "docs/experiments/m5-ultra-256gb"
RAW = ".artifacts/m5-ultra-256gb"
OUT = ROOT / "docs/optimization-guide/progress/2026-10-17-v1"
ARM_COLORS = {"reference": "#356F9F", "candidate": "#C67424"}
PROCESS_COLORS = ["#356F9F", "#C67424", "#856189"]
STATUS_COLORS = {"PASS": "#356F9F", "REJECT": "#AD526C", "PENDING": "#71777C"}
MARKERS = ["o", "s", "^"]
HISTORY = [32, 2048, 8192, 32768, 131072]
BLOCKS = [3, 8]


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sources:
    def __init__(self, root, output):
        self.root, self.output, self.manifest = root, output, {}

    def read(self, relative):
        src = self.root / relative
        dst = self.output / "sources" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        self.manifest[relative] = {"sha256": digest(dst), "bytes": dst.stat().st_size}
        return json.loads(dst.read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def collect_window(s, folder, label, stage, cache_gib):
    base = f"{RAW}/{folder}"
    timing = s.read(f"{base}/timing.json")
    plan = s.read(f"{base}/plan.json")
    complete = s.read(f"{base}/complete.json")
    assert len(timing) == 10
    metric = "ms" if stage == "v1" else "owned_lifecycle_ms"
    points = []
    for index, row in enumerate(timing):
        t = row["full_draft"]
        formal = t["samples"]
        assert len(formal) == 12
        assert [a["variant"] for a in formal] == ["reference", "candidate", "candidate", "reference"] * 3
        med = {v: statistics.median(a[metric] for a in formal if a["variant"] == v)
               for v in ARM_COLORS}
        stored = t["median_ms"] if stage == "v1" else t["median_ms"][metric]
        for v in ARM_COLORS:
            close(med[v], stored[v])
        speedup = med["reference"] / med["candidate"]
        close(speedup, t["speedup"] if stage == "v1" else t["speedup"][metric])
        points.append({"history_rows": row["history_rows"], "block": row["block"],
                       "median_ms": med, "speedup": speedup,
                       "raw_samples": formal, "source": f"{base}/timing.json",
                       "source_pointer": f"/{index}/full_draft"})
    assert {(p["history_rows"], p["block"]) for p in points} == {(n, b) for n in HISTORY for b in BLOCKS}
    short = min(p["speedup"] for p in points if p["history_rows"] in (32, 2048))
    long = min(p["speedup"] for p in points if p["history_rows"] in (32768, 131072))
    gate = short >= .97 and long >= 1.10
    close(short, complete.get("short_min", complete.get("short_min_speedup")))
    close(long, complete.get("long_min", complete.get("long_min_speedup")))
    assert gate == complete.get("gate", complete.get("advance"))
    return {"label": label, "stage": stage, "cache_gib": cache_gib,
            "pid": plan.get("pid"), "metric": metric, "points": points,
            "source_plan": f"{base}/plan.json", "complete": complete,
            "short_min_speedup": short, "long_min_speedup": long, "gate": gate}


def collect(s):
    summaries = {name: s.read(f"{EXP}/{folder}/summary.json") for name, folder in {
        "E010": "prefill-nax-v2", "E011": "E011-large-graph-v1",
        "E012": "E012-draft-selector-integrated-v1", "E014-v1": "E014-draft-window-v1",
        "E014-v2": "E014-draft-window-v2", "E015": "E015-official-mtp-v1",
        "E016": "E016-draft-graph-v1"}.items()}
    prototype = [collect_window(s, folder, label, stage, pool) for folder, label, stage, pool in [
        ("draft-window-v1", "v1 / 2 GiB", "v1", 2),
        ("draft-window-v2-cache2", "v2 / 2 GiB", "v2", 2),
        ("draft-window-v2-cache16", "v2 / 16 GiB", "v2", 16)]]
    integrated = [collect_window(s, f"draft-window-integrated-v1-run{i}", f"run {i}",
                                 "integrated", 16 if i <= 3 else 2) for i in range(1, 5)]
    assert [p["gate"] for p in integrated] == [True, False, True, True]
    assert sum(p["complete"]["numeric_checks"] for p in integrated) == 2944
    # Recompute all HTTP paired ratios and their geometric means from process rows.
    http = summaries["E010"]
    for c in (1, 8, 32, 64):
        for metric in ("x", "y", "TTFT_s"):
            ratios = []
            for pair in (1, 2, 3):
                rs = {r["variant"]: r for r in http["rows"] if r["C"] == c and r["pair"] == pair}
                ratios.append(rs["candidate"][metric] / rs["reference"][metric])
            for a, b in zip(ratios, http["paired_ratios"][str(c)][metric]):
                close(a, b)
            close(statistics.geometric_mean(ratios), http["paired_geomeans"][str(c)][metric])
    for r in summaries["E012"]["rows"]:
        close(r["full_draft_speedup"], r["full_draft_median_ms"]["reference"] / r["full_draft_median_ms"]["candidate"])
    for r in summaries["E016"]["rows"]:
        close(r["speedup"], r["median_ms"]["reference"] / r["median_ms"]["candidate"])
    status = [
        ["E010", "HTTP", "Grouped prefill", "PASS", "Adopted within HTTP contract", "C8/32/64 throughput +14.68 / +12.30 / +14.54%", "24 fresh processes; 19 input / 32 output; target only"],
        ["E011", "Module", "Large decode graph", "REJECT", "Screen rejected", "B16: 1.023×; below 1.10× gate", "One process; no HTTP conclusion"],
        ["E012", "Module", "Device selector", "PASS", "Qualified opt-in; default OFF", "Full-draft process GM: 1.185 / 1.185 / 1.191×", "3 fresh processes; 3 prompts × blocks 3/5/8"],
        ["E014 v1", "Module", "Trim KV on each append", "REJECT", "Prototype speed gate failed", "Shortest failing point: N2048 / B3 = 0.910×", "1 process; synthetic KV replay; old timing contract"],
        ["E014 v2", "Module", "Amortized bounded KV", "PASS", "Prototype gates passed", "16 GiB: short minimum 0.983×; long minimum 1.189×", "2 fresh processes, one per pool; 368 strict cases"],
        ["E014 core", "Module", "Public-loader bounded KV", "REJECT", "Adoption gate failed; default OFF", "16 GiB gates: PASS / FAIL / PASS; run 2 short 0.946×", "4 fresh processes; 2,944 strict cases; synthetic KV replay"],
        ["E015", "Engine API", "Official oMLX MTP", "PASS", "Numeric + actual-route qualification", "192 raw IDs equal; no speed result", "Instrumented; 1 prompt; 3 requests per arm; not HTTP"],
        ["E016", "Module", "Draft backbone graph", "REJECT", "Screen rejected", "Full-draft geometric mean 1.040×; gate 1.10×", "1 process; 9 timing cases; 312 arrays checked"],
        ["Next gate", "HTTP", "Speculative generation", "PENDING", "Generation / rollback / HTTP pending", "No combined speedup or global-fastest claim", "E012 / E014 module gains are not HTTP gains"],
    ]
    return {"date": DATE, "device": "Apple M5 Ultra, 256 GiB, MLX 0.32.3",
            "target_revision": "5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6",
            "draft_revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
            "prototype": prototype, "integrated": integrated, "summaries": summaries,
            "statuses": status, "integrated_adoption_pass": all(p["gate"] for p in integrated[:3]),
            "interval_contract": "No confidence intervals. Raw timing dots and independent process results remain visible.",
            "scope": "Separate HTTP, engine qualification, and draft module experiments; do not add or multiply gains."}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#858B91", "text.color": "#242A30",
        "axes.labelcolor": "#242A30", "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "none", "svg.hashsalt": "minivllm-progress-2026-10-17-v1"})
    return plt


def axes_history(ax, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_xlim(22, 185000)
    ax.set_xticks(HISTORY, ["32", "2,048", "8,192", "32,768", "131,072"])
    ax.set_xlabel("Logical history rows (log₂ axis)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E5E7EA", linewidth=.7)
    ax.set_axisbelow(True)


def save(fig, output, name, footer):
    fig.text(.055, .018, footer, fontsize=10, color="#50575E", va="bottom", linespacing=1.5)
    fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(output / f"{name}.svg", metadata={"Date": None}, facecolor="white")


def prototype_figures(data, out, plt):
    from matplotlib.lines import Line2D
    for ratio in (False, True):
        fig, axs = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True)
        fig.subplots_adjust(left=.055, right=.98, top=.81, bottom=.20, hspace=.43, wspace=.20)
        fig.suptitle("E014 prototype: " + ("KV OFF / KV ON speedup" if ratio else "KV OFF and KV ON measured latency"),
                     x=.055, y=.965, ha="left", fontsize=21, weight="bold")
        fig.text(.055, .918, "Separate contracts: v1 = eager trim + legacy timing; v2 = amortized trim + owned lifecycle. One fresh process per panel column.", fontsize=11)
        for col, proc in enumerate(data["prototype"]):
            for row, block in enumerate(BLOCKS):
                ax = axs[row, col]
                points = [p for p in proc["points"] if p["block"] == block]
                x = [p["history_rows"] for p in points]
                ax.set_title(f"{proc['label']} · block {block}")
                if ratio:
                    y = [p["speedup"] for p in points]
                    ax.plot(x, y, color=ARM_COLORS["candidate"], marker="D", lw=1.4, ms=6)
                    for n, val in zip(x, y):
                        ax.annotate(f"{val:.3f}×", (n, val), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
                    ax.axhline(1, color="#42474D", lw=1)
                    ax.plot([32, 2048], [.97, .97], color="#70757A", ls=":", lw=1.4)
                    ax.plot([32768, 131072], [1.10, 1.10], color="#70757A", ls="--", lw=1.4)
                    ax.set_ylim(.83, 3.95)
                else:
                    for variant, color in ARM_COLORS.items():
                        for p in points:
                            ax.scatter([p["history_rows"]] * 6, [s[proc["metric"]] for s in p["raw_samples"] if s["variant"] == variant],
                                       color=color, s=13, alpha=.28, edgecolors="none")
                        ax.plot(x, [p["median_ms"][variant] for p in points], color=color,
                                marker="o" if variant == "reference" else "s", ms=5.5, lw=1.6)
                    ax.set_ylim(0, 38)
                axes_history(ax, "Speedup (×; higher is faster)" if ratio else "Measured wall time (ms)")
                ax.tick_params(labelbottom=True)
        handles = ([Line2D([], [], color=ARM_COLORS["candidate"], marker="D", label="Ratio of six-sample medians; no CI"),
                    Line2D([], [], color="#70757A", ls=":", label="Short gate ≥0.97×"),
                    Line2D([], [], color="#70757A", ls="--", label="Long gate ≥1.10×")] if ratio else
                   [Line2D([], [], color=c, marker="o" if v == "reference" else "s", label="KV OFF (reference)" if v == "reference" else "KV ON (candidate)") for v, c in ARM_COLORS.items()])
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .89), ncol=3, frameon=False)
        save(fig, out, "e014-prototype-speedup" if ratio else "e014-prototype-latency",
             "Actual N values; synthetic repeated real-KV state, not long generation. 3 ABBA rounds → six formal observations per arm and point.\n"
             + ("v1 / v2 gates: FAIL / PASS / PASS. Lines guide the eye; no values inferred between measurements.\n" if ratio else
                "Large markers = process medians; faint dots = every formal timing (observed spread, not a confidence interval).\n")
             + "v1 times restore + draft; v2 times draft + own teardown, with prepare outside. Policy, timing, and pool changed; cross-version ms differences have no single-variable attribution.")
        plt.close(fig)


def integrated_figures(data, out, plt):
    from matplotlib.lines import Line2D
    for ratio in (False, True):
        fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
        fig.subplots_adjust(left=.065, right=.98, top=.79, bottom=.19, hspace=.43, wspace=.19)
        fig.suptitle("E014 public-loader integration: " + ("all four process speedups" if ratio else "all four process latencies"),
                     x=.065, y=.965, ha="left", fontsize=21, weight="bold")
        fig.text(.065, .915, "Adoption gate FAILED: 16 GiB run 2, N32 / block 3 = 0.946× < 0.97×. All 2,944 strict checks passed; switch remains OFF by default.", fontsize=11)
        for col, pool in enumerate([16, 2]):
            procs = [p for p in data["integrated"] if p["cache_gib"] == pool]
            for row, block in enumerate(BLOCKS):
                ax = axs[row, col]
                ax.set_title(f"{pool} GiB pool · block {block} · {len(procs)} independent process(es)")
                for i, proc in enumerate(procs):
                    points = [p for p in proc["points"] if p["block"] == block]
                    x = [p["history_rows"] for p in points]
                    if ratio:
                        ax.plot(x, [p["speedup"] for p in points], marker=MARKERS[i], color=PROCESS_COLORS[i],
                                ls=["-", "--", ":"][i], lw=1.5, ms=6, label=proc["label"])
                    else:
                        for variant, color in ARM_COLORS.items():
                            ax.plot(x, [p["median_ms"][variant] for p in points], marker=MARKERS[i], color=color,
                                    ls=["-", "--", ":"][i], lw=1.4, ms=5.5)
                            for p in points:
                                ax.scatter([p["history_rows"]] * 6,
                                    [s["owned_lifecycle_ms"] for s in p["raw_samples"] if s["variant"] == variant],
                                    color=color, marker=MARKERS[i], s=11, alpha=.18, edgecolors="none")
                if ratio:
                    ax.axhline(1, color="#42474D", lw=1)
                    ax.plot([32, 2048], [.97, .97], color="#70757A", ls=":", lw=1.3)
                    ax.plot([32768, 131072], [1.10, 1.10], color="#70757A", ls="--", lw=1.3)
                    ax.set_ylim(.82, 3.5)
                    ax.legend(loc="upper left", frameon=False)
                    if pool == 16 and block == 3:
                        ax.annotate("run 2: 0.946× (FAIL)", (32, .9462723305), xytext=(70, 1.60),
                                    arrowprops={"arrowstyle": "->", "color": "#AD526C"}, color="#AD526C", fontsize=11)
                else:
                    ax.set_ylim(0, 38)
                axes_history(ax, "KV OFF / ON (×; higher is faster)" if ratio else "Draft + own teardown (ms)")
                ax.tick_params(labelbottom=True)
        handles = ([Line2D([], [], color="#70757A", ls=":", label="Short gate ≥0.97× at N32 and N2048"),
                    Line2D([], [], color="#70757A", ls="--", label="Long gate ≥1.10× at N32768 and N131072")] if ratio else
                   [Line2D([], [], color=c, lw=1.5, label="KV OFF" if v == "reference" else "KV ON") for v,c in ARM_COLORS.items()] +
                   [Line2D([], [], color="#686E75", marker=m, ls=["-", "--", ":"][i], label=f"16 GiB run {i+1}" if i else "16 GiB run 1 / 2 GiB run 4") for i,m in enumerate(MARKERS)])
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06,.885), ncol=3, frameon=False)
        save(fig, out, "e014-integrated-speedup" if ratio else "e014-integrated-latency",
             "Fixed process order: 16 / 16 / 16 / 2 GiB. Each point uses six formal observations per arm; all process results retained, no pooled median or CI.\n"
             "Complete draft + own teardown; pre-restore outside timing. Four draft weight copies resident, qualification caches reset. Faint latency dots show formal observations.\n"
             "Synthetic repeated real-KV state, selector ON in both timed arms. Python reference release does not prove allocator reclamation. No target verify, rollback, long-generation or HTTP claim.")
        plt.close(fig)


def gate_figure(data, out, plt):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(left=.08, right=.97, top=.76, bottom=.26, wspace=.30)
    fig.suptitle("E014 integration gate: every fresh 16 GiB run must pass", x=.07, y=.95, ha="left", fontsize=19, weight="bold")
    for ax, metric, gate, title in zip(axs, ["short_min_speedup", "long_min_speedup"], [.97, 1.10],
                                      ["Minimum across four short points", "Minimum across four long points"]):
        vals = [p[metric] for p in data["integrated"]]
        ax.axhline(gate, color="#51565C", ls="--", lw=1.4)
        for i, v in enumerate(vals):
            ax.scatter(i, v, marker="o" if i < 3 else "s", s=80, color="#AD526C" if v < gate else "#356F9F")
            ax.annotate(f"{v:.4f}×", (i,v), xytext=(0,10), textcoords="offset points", ha="center", fontsize=11)
        ax.set_xticks(range(4), ["run 1\n16 GiB", "run 2\n16 GiB", "run 3\n16 GiB", "run 4\n2 GiB"])
        ax.set_xlim(-.35, 3.35)
        ax.set_title(title)
        ax.set_ylabel("KV OFF / ON (×)")
        ax.set_ylim((.93, 1.02) if metric.startswith("short") else (1.08, 1.25))
        ax.grid(axis="y", color="#E5E7EA")
    save(fig, out, "e014-integrated-gates",
         "Points are process minima, not confidence bounds. Short = N32 / N2048 × blocks 3 / 8; long = N32768 / N131072 × blocks 3 / 8.\n"
         "Dashed thresholds: short ≥0.97×, long ≥1.10×. Focused y-ranges differ by panel. 2 GiB is a separate compatibility check.\n"
         "Three 16 GiB gates = PASS / FAIL / PASS → overall adoption FAIL. No post-hoc rerun selection; default OFF.")
    plt.close(fig)


def progress_figure(data, out, plt):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(left=.07, right=.97, top=.83, bottom=.15, hspace=.55, wspace=.22)
    fig.suptitle("Measured progress: HTTP and isolated modules stay separate", x=.07, y=.965, ha="left", fontsize=20, weight="bold")
    fig.text(.07, .916, "Independent processes/pairs are shown individually. These are different baselines and workloads; speedups must not be added or multiplied.", fontsize=11)
    h = data["summaries"]["E010"]
    for col, metric, title in [(0,"y","E010 · HTTP aggregate throughput"), (1,"TTFT_s","E010 · HTTP time to first text")]:
        ax = axs[0,col]
        for pair in range(3):
            ax.plot([1,8,32,64], [h["paired_ratios"][str(c)][metric][pair] for c in (1,8,32,64)],
                    color=PROCESS_COLORS[pair], marker=MARKERS[pair], lw=1.1, label=f"fresh-process pair {pair+1}")
        ax.set_xscale("log",base=2); ax.set_xticks([1,8,32,64],["1","8","32","64"])
        ax.axhline(1, color="#50565D",lw=1)
        ax.set_title(title);ax.set_xlabel("Concurrency (log₂ axis)")
        ax.set_ylabel("Candidate / reference (×)")
        ax.set_ylim((.95,1.25) if col==0 else (.58,1.04))
        ax.legend(frameon=False, fontsize=9)
        ax.text(.02,.02,"Higher is better" if col==0 else "Lower is better",transform=ax.transAxes,fontsize=10)
    sel=data["summaries"]["E012"]
    ax=axs[1,0]
    for p in range(1,4):
        rs=[r for r in sel["rows"] if r["repetition"]==p]
        ax.scatter([r["block"] for r in rs],[r["full_draft_speedup"] for r in rs],
                   color=PROCESS_COLORS[p-1],marker=MARKERS[p-1],s=35,alpha=.8,
                   label=f"process {p} · GM {sel['geomean_by_process'][p-1]:.3f}×")
    ax.set_title("E012 · full draft with device selector");ax.legend(frameon=False,fontsize=9,loc="upper left")
    ax.set_ylabel("Reference / candidate (×)")
    graph=data["summaries"]["E016"]
    ax=axs[1,1]
    for i,prompt in enumerate(["math","code","chinese"]):
        rs=[r for r in graph["rows"] if r["prompt"]==prompt]
        ax.scatter([r["block"] for r in rs],[r["speedup"] for r in rs],color=PROCESS_COLORS[i],marker=MARKERS[i],s=40,label=prompt)
    ax.axhline(1.10,color="#70757A",ls="--",lw=1.3)
    ax.set_title("E016 · full draft with backbone graph");ax.legend(frameon=False,fontsize=9,loc="upper right")
    ax.set_ylabel("Reference / candidate (×)")
    for ax in axs[1,:]:
        ax.axhline(1,color="#50565D",lw=1)
        ax.set_ylim(.98,1.32);ax.set_xticks([3,5,8]);ax.set_xlim(2.5,8.5);ax.set_xlabel("Draft block size")
    for ax in axs.flat: ax.grid(axis="y",color="#E5E7EA");ax.set_axisbelow(True)
    save(fig,out,"measured-progress",
         "Top: 24 fresh HTTP processes; 19 input → 32 output tokens, target only, C1/8/32/64. Each pair ratio uses that pair's process summary.\n"
         "Bottom left: 3 fresh processes × 3 prompts × 3 blocks, all 27 points; each process GM uses its 9 cases. Bottom right: 1 process, all 9 prompt/block cases.\n"
         "Module ratios use full-draft median latencies. Observation spread is not a confidence interval. E016 GM = 1.040×, below the 1.10× screen gate.")
    plt.close(fig)


def status_figure(data,out,plt):
    fig,ax=plt.subplots(figsize=(18,10.4));ax.axis("off")
    fig.subplots_adjust(left=.025,right=.98,top=.83,bottom=.08)
    fig.suptitle("M5 Ultra experiment progress · 2026-10-17",x=.035,y=.965,ha="left",fontsize=23,weight="bold")
    fig.text(.035,.91,"Accepted, rejected, and unfinished steps are preserved. Numeric qualification and performance adoption are separate decisions.",fontsize=12)
    columns=["Experiment / scope","Module","Decision","Measured evidence / limits"]
    rows=[]
    for eid,scope,module,state,decision,evidence,limit in data["statuses"]:
        rows.append([f"{eid}\n{scope}",module,f"{state}\n{decision}",f"{evidence}\n{limit}"])
    table=ax.table(cellText=rows,colLabels=columns,colWidths=[.105,.185,.25,.46],cellLoc="left",loc="center")
    table.auto_set_font_size(False);table.set_fontsize(10.6)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor("#E1E5E8");cell.set_linewidth(.6)
        cell.set_height(.108 if row else .06)
        cell.PAD=.03
        cell.set_facecolor("#F2F4F6" if row==0 else ("#FBFCFD" if row%2 else "white"))
        if row==0:cell.set_text_props(weight="bold",color="#30363C")
        elif col==2:cell.set_text_props(color=STATUS_COLORS[data["statuses"][row-1][3]])
    fig.text(.035,.025,"Pinned source JSON + SHA-256 + chart-ready rows accompany every figure. No global-fastest claim, combined multiplier, or inference from unmeasured workloads.",fontsize=11,color="#50575E")
    fig.savefig(out/"progress-status.png",dpi=180);fig.savefig(out/"progress-status.svg",metadata={"Date":None})
    plt.close(fig)


def readme(data,out):
    rows="\n".join("| "+" | ".join(r[:3]+[r[4],r[5],r[6]])+" |" for r in data["statuses"])
    (out/"README.md").write_text(f"""# M5 Ultra progress snapshot · {DATE}

E014 原型 v2 的两个池都通过，但实际生产 loader 接入的三个 16 GiB fresh processes 为 **PASS / FAIL / PASS**。第二进程 N32/B3 为 **0.946272×**，低于 0.97×；因此整体采用门未过，模块保持默认 OFF。四进程全部 **2,944** 个严格数值检查通过。2 GiB 兼容性进程通过。所有进程均保留，未合成一个掩盖失败的中位数。

![Progress status](progress-status.png)

## 可复现科学图

- [E014 prototype latency](e014-prototype-latency.png) · [SVG](e014-prototype-latency.svg)
- [E014 prototype speedup](e014-prototype-speedup.png) · [SVG](e014-prototype-speedup.svg)
- [E014 integrated latency](e014-integrated-latency.png) · [SVG](e014-integrated-latency.svg)
- [E014 integrated speedup](e014-integrated-speedup.png) · [SVG](e014-integrated-speedup.svg)
- [E014 independent-process gates](e014-integrated-gates.png) · [SVG](e014-integrated-gates.svg)
- [HTTP / module measured progress](measured-progress.png) · [SVG](measured-progress.svg)
- [Status table](progress-status.png) · [SVG](progress-status.svg)

图中的 history 是实际数值坐标（log₂），长 history 由短真实 KV 重复构建，是 **synthetic cache-state replay，不是实际长生成**。Block 3/8 分面。KV OFF 蓝色、KV ON 橙色；进程使用不同形状及线型。原型每个条件只有一个 fresh process；core 接入保留全部三个 16 GiB 及一个 2 GiB 进程。小点是全部正式 timing，粗点是每进程六样本的中位数，不是独立进程或置信区间。连接线只引导阅读，不推算未测点。

v1 每次追加都裁 KV，legacy timing 包含 restore + draft；v2 改为 physical > 4096 才裁到 2048，并测量完整 draft + 本臂 teardown，pre-restore 不计时。策略、计时生命周期及池条件均改变，**不能把 v1→v2 的绝对毫秒差归因于一个因素**。core 接入又有四份独立 drafter 权重常驻，是另一个阶段。应用层 Python refs 清理不等于 allocator 立即回收。原型 16 GiB 最后资源区间 Swapins +3，不能声称完全无 paging。

HTTP E010 是 19→32 target-only 请求；E012/E014/E016 是局部 draft 模块；E015 是有观察器的官方 EnginePool 数值/真实路由资格，**没有 speed 结果**。这些比值不相乘、不相加，也不构成全球最快或 speculative HTTP 的结论。

## 逐步状态

| ID | Scope | Module | Decision | Evidence | Limits |
|---|---|---|---|---|---|
{rows}

## 重画与证据

从仓库根运行（Python + Matplotlib 3.9.4；不导入 MLX，不使用 GPU）。本机使用独立 `.artifacts/m5-ultra-256gb/progress-render-venv`，完整版本见 [requirements.txt](requirements.txt)：

```sh
python3 bench/m5_ultra_256gb/progress/plot.py
```

只从本快照已固定的小 JSON 重画到新目录，无需模型、原始 .artifacts 或网络：

```sh
python3 bench/m5_ultra_256gb/progress/plot.py --snapshot docs/optimization-guide/progress/2026-10-17-v1/sources --output /tmp/minivllm-progress-rebuild
```

[chartdata.json](chartdata.json) 保存每个图点及其原始 JSON path/pointer、正式样本和独立进程身份；[source-manifest.json](source-manifest.json) 保存每个输入的 SHA-256 / bytes；`sources/` 保存输入副本；[summary.json](summary.json) 保存复核与采用结论。生成器在画图前重算全部 E014 中位数、速度比和门，以及 E010 逐进程配对比值、E012/E016 模块比值。正式实验的完整数组、资源与进程清理审计属于对应实验归档，本图不会代替它们。

源实验：[E010](../../../experiments/m5-ultra-256gb/prefill-nax-v2/README.md)、[E012](../../../experiments/m5-ultra-256gb/E012-draft-selector-integrated-v1/README.md)、[E014 v2](../../../experiments/m5-ultra-256gb/E014-draft-window-v2/README.md)、[E014 integrated](../../../experiments/m5-ultra-256gb/E014-draft-window-integrated-v1/README.md)、[E015](../../../experiments/m5-ultra-256gb/E015-official-mtp-v1/README.md)、[E016](../../../experiments/m5-ultra-256gb/E016-draft-graph-v1/README.md)。

固定背景白色、细灰网格、DejaVu Sans；PNG 为 180 dpi，SVG 保留文字。工程仓库图使用本项目身份，无第三方品牌装饰。
""")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot",type=Path,help="Read the retained sources/ tree instead of the working repository")
    ap.add_argument("--output",type=Path,default=OUT)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.snapshot:
        pinned=json.loads((args.snapshot.parent/"source-manifest.json").read_text())
        for rel, meta in pinned.items():
            assert digest(args.snapshot/rel)==meta["sha256"], rel
    s=Sources(args.snapshot.resolve() if args.snapshot else ROOT,args.output)
    data=collect(s);dump(args.output/"chartdata.json",data)
    source=Path(__file__).resolve(); codecopy=args.output/"sources/bench/m5_ultra_256gb/progress/plot.py"
    codecopy.parent.mkdir(parents=True,exist_ok=True)
    if source != codecopy.resolve():shutil.copyfile(source,codecopy)
    s.manifest["bench/m5_ultra_256gb/progress/plot.py"]={"sha256":digest(codecopy),"bytes":codecopy.stat().st_size}
    dump(args.output/"source-manifest.json",s.manifest)
    plt=style();prototype_figures(data,args.output,plt);integrated_figures(data,args.output,plt)
    gate_figure(data,args.output,plt);progress_figure(data,args.output,plt);status_figure(data,args.output,plt)
    readme(data,args.output)
    import matplotlib
    summary={"date":DATE,"status":"NUMERIC_CHART_RECONCILIATION_PASS","sources":len(s.manifest),
             "figures":7,"formats":["PNG","SVG"],"matplotlib":matplotlib.__version__,
             "e014_prototype_gates":[p["gate"] for p in data["prototype"]],
             "e014_integrated_process_gates":[p["gate"] for p in data["integrated"]],
             "e014_integrated_adoption_pass":data["integrated_adoption_pass"],
             "e014_formal_samples_retained":sum(len(p["raw_samples"]) for group in (data["prototype"],data["integrated"]) for proc in group for p in proc["points"]),
             "visual_inspection":"Recorded separately in visual-review.json after rendered inspection",
             "claims":"No CI, combined multiplier, cross-stage causal effect, or HTTP extrapolation"}
    dump(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
