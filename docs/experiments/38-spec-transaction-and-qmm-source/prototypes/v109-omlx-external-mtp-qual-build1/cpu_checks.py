"""Source/API boundary and receipt-counter tests only, never numeric simulation."""
import ast
import copy
import json
from pathlib import Path
import sys
import textwrap
from checks import inspect_observations, equal_streams
D=Path(__file__).resolve().parent
P=D.parent
OLD=P/'m4-final-native15-official-attempt2/client.py'

def functions(path):
    return {n.name:n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef)}

def main():
    count=0
    for path in D.glob('*.py'):
        compile(path.read_text(),str(path),'exec'); count+=1
    old=functions(OLD);new=functions(D/'fullqual.py')
    for name in ('request','group'):
        assert ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False);count+=1
    S=Path(json.loads((D/'config.json').read_text())['runtime_vendor_root'])
    wanted={'omlx/scheduler.py':['Scheduler','_route_to_vlm_mtp','_log_vlm_mtp_stats','_process_batch_responses'],
            'omlx/engine/vlm.py':['VLMBatchedEngine','set_vlm_mtp_drafter'],
            'omlx/speculative/vlm_mtp.py':['load_vlm_mtp_drafter'],
            'mlx_vlm/speculative/mtp.py':['_mtp_verify_target']}
    for path,names in wanted.items():
        tree=ast.parse((S/path).read_text());found={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        assert set(names)<=found;count+=1
    # Receipt tampering checks exercise the actual fail-closed validator only.
    cfg={'eos_token_ids':[999],'prompt_ids':list(range(19)),'draft_model_dir':'/draft','draft_config_block_size':3}
    requests=[{'visible_roundtrip_ids':list(range(32))} for _ in range(3)]
    rows=[{'kind':'observer_installed'}]
    for i in range(3):
        rows.append({'kind':'terminal','request_id':str(i),'finish_reason':'length','error':None,'prompt_tokens':19,'completion_tokens':32,'cached_tokens':0,'raw_output_ids':list(range(32))})
    ordinary=inspect_observations(rows,'none',requests,cfg);count+=1
    assert equal_streams(ordinary,copy.deepcopy(ordinary))==list(range(32));count+=1
    def reject(fn):
        try:fn()
        except AssertionError:return
        raise AssertionError('tampered receipt was accepted')
    for key,value in [('cached_tokens',1),('finish_reason','stop'),('completion_tokens',31),('raw_output_ids',[0]*32)]:
        bad=copy.deepcopy(rows);bad[1][key]=value
        reject(lambda:inspect_observations(bad,'none',requests,cfg));count+=1
    reject(lambda:inspect_observations(rows,'mtp',requests,cfg));count+=1
    altered=copy.deepcopy(ordinary);altered[1]['raw_output_ids'][1]=987
    reject(lambda:equal_streams(ordinary,altered));count+=1
    observer=(D/'observer.py').read_text()
    assert 'mx.eval(' not in observer and 'mx.async_eval(' not in observer
    assert 'result = original(' in observer and "shape[1] - 1" in observer
    count+=1
    assert not any(x=='mlx' or x.startswith(('mlx.','numpy')) for x in sys.modules)
    result={'status':'PASS_CPU_SOURCE_API_AST_AND_RECEIPT_REJECTION_ONLY','checks':count,'GPU_used':False,'MLX_imported':False,'model_loaded':False,'inference_claim':False}
    print(json.dumps(result))
    return result
if __name__=='__main__':main()
