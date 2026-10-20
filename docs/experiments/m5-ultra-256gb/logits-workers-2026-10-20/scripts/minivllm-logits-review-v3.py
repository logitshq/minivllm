from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.request

R = Path("/Users/ryanli/code/minivllm")
A = R/".artifacts/m5-ultra-256gb"
OUT = A/"logits-research-2026-10-20-v3"
OUT.mkdir(exist_ok=False)

def run(name):
    folder=OUT/name
    folder.mkdir()
    request=json.loads((A/"logits-research-2026-10-20-v2"/name/"request.json").read_text())
    request.update(reasoning_effort="none", max_tokens=8192)
    if name=="optimization-research":
        rel="minivllm/model/qwen3_5.py"
        source=(R/rel).read_text().splitlines()
        request["messages"][-1]["content"]+=(
            "\nIMPORTANT correction to your second review: You proposed capturing "
            "post_attention_layernorm h or final normalized. Check the actual model "
            "implementation below. Identify the exact expression to capture and "
            "retract the wrong proposed location. Limit this response to 300 words.\nFILE "
            +rel+"\n"+"\n".join(f"{i+1}: {line}" for i,line in enumerate(source)))
    (folder/"request.json").write_text(json.dumps(request,indent=2))
    key=Path("/Users/ryanli/.config/logits-codex/api-key").read_text().strip()
    start=time.monotonic()
    result={"model":request["model"],"reasoning_effort":"none"}
    try:
        req=urllib.request.Request("https://api.logits.com/v1/chat/completions",
            data=json.dumps(request).encode(),
            headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=120) as response:
            value=json.loads(response.read())
            result["http_status"]=response.status
        (folder/"response.json").write_text(json.dumps(value,indent=2))
        choice=value["choices"][0]
        review=choice["message"].get("content") or ""
        (folder/"review.md").write_text(review)
        result.update(returned_model=value.get("model"),finish_reason=choice.get("finish_reason"),
                      usage=value.get("usage"),review_chars=len(review),
                      transport_pass=bool(review) and choice.get("finish_reason")=="stop")
    except Exception as error:
        result.update(transport_pass=False,error_type=type(error).__name__,
                      error=str(error).replace(key,"[REDACTED]"))
    result["elapsed_s"]=round(time.monotonic()-start,3)
    (folder/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result

with ThreadPoolExecutor(max_workers=2) as pool:
    results=list(pool.map(run,["e055-review","optimization-research"]))
(OUT/"summary.json").write_text(json.dumps(results,indent=2))
