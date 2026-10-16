"""CPU replay: rank actual GPU command-buffer envelopes; never apportion mixed groups."""
import argparse
import collections
import json
from pathlib import Path

def classify(dispatch,shapes):
    name=dispatch["kernel"].lower()
    if any(x in name for x in ("qmv","qmm","gemv","gemm","fp_quant")):
        ints={v for b in dispatch.get("bindings",{}).values() for v in b.get("int32_values",[])}
        possible={s["component"] for s in shapes if s["K"] in ints and s["N"] in ints}
        return next(iter(possible))+"_quantized_projection" if len(possible)==1 else "quantized_unattributed"
    if any(x in name for x in ("gated_delta","gdn","conv_silu")):return "GDN_recurrence_or_conv"
    if any(x in name for x in ("sdpa","attention","rope")):return "attention"
    return "elementwise_reduction_or_misc"

def summarize(directory):
    receipt=json.loads((directory/"receipt.json").read_text());native=json.loads((directory/"native.json").read_text())
    groups=[];kernels=collections.Counter();host=collections.Counter();intervals=[];categories=collections.Counter()
    for row in native["rows"]:
        ds=row["dispatches"];names=collections.Counter(d["kernel"] for d in ds)
        family=collections.Counter(classify(d,receipt["linear_shapes"]) for d in ds)
        valid=row.get("completed") and row.get("status")==4 and row.get("gpu_end_s",0)>row.get("gpu_start_s",0)>0
        duration=round((row["gpu_end_s"]-row["gpu_start_s"])*1e9) if valid else None
        category=next(iter(family)) if len(family)==1 else "mixed" if family else "empty_or_unobserved"
        if duration is not None:
            intervals.append((row["gpu_start_s"],row["gpu_end_s"]))
            categories[category]+=duration
        groups.append({"id":row["id"],"step":row["step"],"GPU_envelope_ns":duration,"category":category,"members":dict(names),"dispatch_count":len(ds),"member_categories":dict(family),"host_encoder_span_ns":row.get("host_encoder_span_ns"),"host_commit_api_ns":row.get("host_commit_api_ns"),"error":row.get("error")})
        kernels.update(names)
        for d in ds:host[d["kernel"]]+=d["host_dispatch_api_ns"]
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return {"scope":"instrumented diagnostic, not an E2E/Pareto point","dispatch_boundary_supported":False,"per_dispatch_GPU_time_available":False,"group_GPU_sum_ns":sum(x["GPU_envelope_ns"] or 0 for x in groups),"group_GPU_envelope_union_ns":round(sum(b-a for a,b in merged)*1e9),"group_GPU_ns_by_category_sum":dict(categories),"mixed_never_apportioned":True,"projection_attribution":"derived unique source Linear K/N component among actual captured small-int bindings; ambiguous stays unattributed","top_GPU_buffers":sorted(groups,key=lambda x:x["GPU_envelope_ns"] or -1,reverse=True)[:30],"kernels_by_dispatch_count":[{"name":n,"count":c,"host_dispatch_API_ns":host[n]} for n,c in kernels.most_common()],"invalid_or_incomplete_buffers":sum(x["GPU_envelope_ns"] is None for x in groups),"unhooked_indirect_dispatch_and_blit_coverage":"unknown; buffer GPU envelope may include noncompute work","host_timing_scope":"model graph-call wall; encoder lifetime spans include MLX work/instrumentation; dispatch/commit API wall are separately observed; eval wall also waits for GPU"}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("directory",type=Path);args=ap.parse_args();value=summarize(args.directory);(args.directory/"summary.json").write_text(json.dumps(value,indent=2)+"\n");print(json.dumps(value,indent=2))
