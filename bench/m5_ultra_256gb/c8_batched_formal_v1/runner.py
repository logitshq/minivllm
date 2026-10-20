"""E055 reuses the settled E045 HTTP owner; retain its receipt and add exact scope."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'concurrency_config_v2/runner.py'
BASE_SHA='dfb0b6f07e775249babab6e76c62dbf1ab8012027e3cb46f6c694b8b2b730577'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA
spec=importlib.util.spec_from_file_location('_e055_owned_resources',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
sha,write,process,verify_pins,physical=base.sha,base.write,base.process,base.verify_pins,base.physical

async def run(plan,out):
    fn=base.run;scope=dict(vars(base),HERE=HERE)
    code=await FunctionType(fn.__code__,scope,'run',fn.__defaults__,fn.__closure__)(plan,out)
    receipt=Path(out)/'result.json'
    if receipt.exists():
        inherited=json.loads(receipt.read_text())
        write(Path(out)/'scope.json',dict(schema='E055-paired-formal-owner-scope-v1',
            inherited_result=dict(path=str(receipt),sha256=sha(receipt),status=inherited['status']),
            expected_trials=6,warm_per_trial=8,formal_per_trial=64,source_mutation=False,
            performance_qualified=False,engine_pareto_point=False,
            scope='Original owner receipt retained verbatim, including its legacy8+8 description. Actual E055 specs run8 warm+64 formal per fresh process; all performance/output claims await independent audit.'))
    return code

def main():
    fn=base.main;scope=dict(vars(base),HERE=HERE,run=run)
    return FunctionType(fn.__code__,scope,'main',fn.__defaults__,fn.__closure__)()

if __name__=='__main__':raise SystemExit(main())
