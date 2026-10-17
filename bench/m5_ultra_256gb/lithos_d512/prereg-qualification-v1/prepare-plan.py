"""Freeze one C1/C8 qualification plan, without process/service/model actions."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

P=Path(__file__).resolve().parent
HERE=P.parent
ROOT=HERE.parents[2]
BASE=ROOT/'.artifacts/m5-ultra-256gb'
J=BASE/'lithos-d512-qualification-v1-cold-journal'
O=BASE/'lithos-d512-qualification-v1-owner'
OWNER_SHA='a6796b153d558954899a46a96cd8ead828928e8627aea1ba8a8b6453ab74bddd'
CONTROLLER_SHA='becfd425ba67cd42acbd172cc24ff15e9ba6f38bc2ae7009e68da8936bc89b92'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE/'owner.py')==OWNER_SHA and sha(HERE/'controller.py')==CONTROLLER_SHA
assert not (P/'owner-plan.json').exists() and not J.exists() and not O.exists()
assert sha(P/'verification.json')=='be3a6b683cc341914032c76d183c9a6b1d2342a5c27ddd3c49b4b6d7fe346a79'
spec=importlib.util.spec_from_file_location('frozen_lithos_owner',HERE/'owner.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prior=ROOT/'docs/experiments/m5-ultra-256gb/E024-route-timing-v1'
ref=lambda p:dict(path=str(p),sha256=sha(p))
predecessor=dict(**ref(prior/'restored-journal.json'),audit=ref(prior/'restoration-audit.json'))
key='minivllm-lithos-d512-local-qualification-v1'
# Public experiment-only key on localhost, unrelated to any account credential.
environment={k:os.environ[k] for k in ('PATH','HOME','TMPDIR','LANG','LC_ALL','USER','LOGNAME') if k in os.environ}
environment.update(MLX_MAX_OPS_PER_BUFFER='50',MLX_MAX_MB_PER_BUFFER='50',MLX_BFS_MAX_WIDTH='20',MLX_METAL_FAST_SYNCH='0',
    HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MONOLITH_ACCELERATOR='on',LITHOS_METAL_API_KEY=key)
deadlines=dict(ready=180,request=600,group=900,overall=1200,close=10,term=15,kill=15,resource_interval=5)
plan=m.plan_template(
    qualification=ref(HERE.parent/'pareto_realistic/prereg-qualification-v1/plan.json'),
    cold_restart=dict(journal_path=str(J),controller=ref(HERE/'controller.py'),approved_contract='cold-restart-persisted-visible-v1',
        proposal_plan_sha256='e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8',predecessor=predecessor),
    http=dict(port=18881,api_key=key,deadlines=deadlines,environment=environment),
    verification=json.loads((P/'verification.json').read_bytes()))
(P/'owner-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True,allow_nan=False)+'\n')
args=SimpleNamespace(plan=P/'owner-plan.json',plan_sha256=sha(P/'owner-plan.json'),out=O)
_,_,_,_,specs=m.validate_plan(args)
(P/'preparation.json').write_text(json.dumps(dict(schema='Lithos-D512-prereg-prepare-v1',status='CPU_PREPARED',
    owner_plan=ref(P/'owner-plan.json'),prepare_source=ref(Path(__file__)),owner_directory=str(O),journal_directory=str(J),
    trial_specs=specs,deadline_seconds=plan['deadline_seconds'],runtime_weights_pack_verified=False,
    service_action=False,scope='Metadata and source validation only; actual payload checks occur in owner before/after qualification.'),indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status='CPU_PREPARED',owner_plan_sha256=args.plan_sha256,deadline_seconds=plan['deadline_seconds'],
    request_counts=[len(s['jobs']) for s in specs],port=18881)))
