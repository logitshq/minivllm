"""One actual C2/C4 pair; frozen numerical primitives and original hard budget.

Only called by an explicitly run replay adapter under an external cold owner.
No backend import or actions occur on import. The registered C1/C8 plan is kept
byte-for-byte as provenance; this runner records its real C2/C4 loader identity.
"""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import traceback
HERE=Path(__file__).resolve().parent
BASE_SHA='26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def local(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
domain=local(HERE/'domain.py','_c24_single_pair_domain')


def run(base, qualification, numeric, fixtures, concurrency, out):
    domain.configuration(numeric,concurrency)
    domain.need(sha(base.__file__)==BASE_SHA,'frozen numerical primitives changed')
    raw=Path(qualification['path']).read_bytes()
    domain.need(hashlib.sha256(raw).hexdigest()==qualification['sha256']==domain.BASE_PLAN_SHA
                and json.loads(raw)==numeric,'unchanged original base plan')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'plan.json').write_bytes(raw)
    started=time.monotonic();budget=numeric['deadline_s']
    def deadline(*unused):
        if unused or time.monotonic()-started>=budget:raise TimeoutError('absolute numerical qualification deadline')
    domain.need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'outer Python alarm already active')
    previous_handler=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,budget)
    engines=[];observers=[];mx=None;ledger=None
    result=dict(schema='GDN-graph-combined-c24-pair-result-v1',status='INCOMPLETE',cases=[],
        numeric_only_not_http=True,plan_sha256=qualification['sha256'],qualifier_sha256=sha(__file__),
        math_source_sha256=BASE_SHA,concurrency=concurrency,configuration=domain.configuration(numeric,concurrency))
    def phase(name,**fields):
        if ledger is not None:
            ledger.write(json.dumps(dict(event=name,elapsed_wall_s=time.monotonic()-started,**fields))+'\n');ledger.flush()
    def release_pair():
        phase('release-pair-start',model_instances=len(engines))
        first=None
        for observed in observers:
            try:observed.restore()
            except BaseException as error:
                if first is None:first=error
            finally:
                observed.calls.clear();observed.previous_future=observed.postrun=None
        observers.clear();engines.clear();gc.collect()
        if mx is not None:mx.synchronize();mx.clear_cache()
        if first is not None:raise first
        phase('release-pair-complete',qualification_only=True)
    try:
        source=base.pinned_json(numeric['source_manifest'])['files']
        model=base.pinned_json(numeric['model_manifest'])
        domain.need(model.get('schema')=='E023-model-v1' and
            Path(model['model']).resolve()==Path(numeric['model']).resolve() and
            model['revision']==Path(numeric['model']).name,'exact original model manifest/path/revision')
        fixed_fixtures=base.pinned_json(numeric['fixtures'])
        domain.need(Path(fixed_fixtures['model']).resolve()==Path(numeric['model']).resolve()
            and fixed_fixtures['fixtures']==fixtures,'exact original model fixtures')
        root=Path(numeric['package_root']).resolve()/'tutorial_code/miniengine'
        domain.need(set(source)=={str(p.relative_to(root)) for p in root.rglob('*.py')},'full native Python source manifest')
        required={'config.json','generation_config.json'}|{str(p.relative_to(numeric['model'])) for p in Path(numeric['model']).rglob('*.safetensors')}
        domain.need(required<=set(model['files']) and any(p.endswith('.safetensors') for p in required),'all model shards covered')
        base.verify_files(root,source,deadline);base.verify_files(numeric['model'],model['files'],deadline)
        domain.need(len(fixtures)==8 and all(f['normal_eos'] is True and f['max_output_tokens']==512
            and len(f['prompt_ids'])>512 and len(f['prompt_ids'])+512<=2048 for f in fixtures),'original D8 normalEOS512')
        sys.path.insert(0,str(Path(numeric['package_root']).resolve()))
        import mlx.core as mx
        import tutorial_code.miniengine as package
        from tutorial_code.miniengine.engine.llm_engine import LLMEngine
        from tutorial_code.miniengine.engine import model_runner as runner_module
        from tutorial_code.miniengine.sampling_params import SamplingParams
        from mlx.utils import tree_flatten
        domain.need(Path(package.__file__).resolve().parent==root,'actual loaded native root')
        cmp=base.Comparator(mx,out);ledger=(out/'steps.jsonl').open('x')
        domain.load_pair_into(engines,observers,numeric=numeric,fixtures=fixtures,value=concurrency,
            engine_type=LLMEngine,mx=mx,tree_flatten=tree_flatten,observer_type=base.Observer,
            comparator=cmp,deadline=deadline,phase=phase)
        cases=base.workloads(fixtures)
        domain.need(cases==[dict(name=f'D-C{concurrency}',concurrency=concurrency)],'one actual C2/C4 recorded case')
        result['cases'].append(base.run_case(cases[0],engines,observers,SamplingParams,runner_module,cmp,ledger,deadline))
        result['peak_bytes']=mx.get_peak_memory();release_pair()
        base.verify_files(root,source,deadline);base.verify_files(numeric['model'],model['files'],deadline)
        result.update(status='PASS',total_requests=sum(len(c['completed']) for c in result['cases']),
            source_model_pins_rechecked=True,candidate=numeric['candidate'],
            reference='normal engine eager; gates/grouping OFF; previous ahead discarded before each step',
            scope='One C2/C4 captured history; original mathematical comparisons, no performance qualification.')
    except BaseException as error:
        result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        try:
            if engines or observers:release_pair()
        except BaseException as error:
            result.update(status='FAIL',cleanup_error=dict(type=type(error).__name__,message=str(error)))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_handler)
            if ledger is not None:ledger.close()
        result['elapsed_wall_s']=time.monotonic()-started;base.write(out/'complete.json',result)
    return 0 if result['status']=='PASS' else 1
