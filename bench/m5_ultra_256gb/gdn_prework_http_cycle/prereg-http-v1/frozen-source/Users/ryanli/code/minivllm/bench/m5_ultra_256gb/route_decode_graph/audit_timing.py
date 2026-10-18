"""Small offline summary of the fixed forward diagnostic, never a Pareto point.

No success tensors were saved: numerical evidence is pinned producer bit checks
and full-byte hashes. Timing is forward+all products eval+sync, excluding copies,
verification and compilation. Teardown is separate, not request latency.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
TIMING_SHA = 'd87457b559a876187c43b7096dfb9bb09c97f8d79655563e0b459011fc543190'
SUPPORT_SHA = 'afb403c89c6af2d290b599af5fe932db848af5fdcfc07b0699245b8ea02b8490'
MATRIX_PROOF_SHA = '12ae408150d99873fea27375c0f1489e50067f5fdb2b1012e56eac00fbf58f4b'
DEFAULT_PROOF = HERE.parents[2]/'.artifacts/m5-ultra-256gb/route-decode-matrix-v1-analysis/independent-numerical.json'
POINTS = ((1,1023),(8,1023),(1,1024),(8,1024))


def need(ok, message):
    if not ok: raise ValueError(message)
def read(path): return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def support():
    path=HERE/'audit_matrix.py';need(sha(path)==SUPPORT_SHA,'fixed matrix audit helper')
    spec=importlib.util.spec_from_file_location('_timing_audit_support',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def positive(x,zero=False):return type(x) in (int,float) and math.isfinite(x) and (x>=0 if zero else x>0)

def metrics(rows,batch):
    need(len(rows)==16,'four warm and twelve formal samples')
    for warm,part in ((True,rows[:4]),(False,rows[4:])):
        need([r['arm'] for r in part]==list('ACCA')*(1 if warm else 3),'actual ACCA sequence')
        for index,row in enumerate(part):
            need(row['index']==index and type(row['index']) is int and row['warm'] is warm and row['status']=='PASS','sample status/index/warm')
            need(all(positive(row[k],k=='teardown_ms') for k in ('forward_ms','teardown_ms','owned_lifecycle_ms')),'finite sample intervals')
            need(row['owned_lifecycle_ms']==row['forward_ms']+row['teardown_ms']
                 and row['diagnostic_tokens_per_second']==batch*1000/row['forward_ms'],'derived sample arithmetic')
    formal=rows[4:];summary={}
    for arm in ('A','C'):
        selected=[r for r in formal if r['arm']==arm]
        summary[arm]={key:statistics.median(r[key] for r in selected) for key in ('forward_ms','teardown_ms','owned_lifecycle_ms','diagnostic_tokens_per_second')}
    pairs=[]
    for i in range(3):
        block=formal[4*i:4*i+4]
        amean=(block[0]['forward_ms']+block[3]['forward_ms'])/2
        cmean=(block[1]['forward_ms']+block[2]['forward_ms'])/2
        pairs.append(dict(block=i,arms='ACCA',forward_ms=[r['forward_ms'] for r in block],A_mean_ms=amean,C_mean_ms=cmean,A_over_C=amean/cmean))
    return dict(formal_samples=12,warm_samples=4,medians=summary,median_forward_speedup=summary['A']['forward_ms']/summary['C']['forward_ms'],
                ACCA_blocks=pairs,raw_samples=rows)


def identities(m,child,c,plan_sha,proof):
    need(sha(proof)==MATRIX_PROOF_SHA,'accepted independent matrix receipt bytes')
    matrix=read(proof);need(matrix['status']=='PASS' and matrix['matrix_qualified'] is True and matrix['point_count']==52,'completed prerequisite matrix')
    p=read(child/'plan.json');need(sha(child/'plan.json')==plan_sha==c['plan_sha256'],'timing plan SHA')
    need(set(p)=={'schema','qualifier_sha256','candidate','e024_plan','deadline_s'} and p['schema']=='E024-route-timing-plan-v1'
         and p['qualifier_sha256']==TIMING_SHA and positive(p['deadline_s']) and p['deadline_s']<=1800,'fixed timing plan')
    need(p['candidate']==dict(path=str(HERE/'candidate.py'),sha256=m.CANDIDATE_SHA),'fixed candidate')
    oldref=p['e024_plan'];need(set(oldref)=={'path','sha256'} and oldref['sha256']==m.E024_SHA==sha(oldref['path']),'old E024 plan')
    old=read(oldref['path']);need(m.same(read(child/'inherited-e024-plan.json'),old),'saved E024 plan')
    inherited=old['e023_plan'];need(inherited['sha256']==m.E023_SHA==sha(inherited['path']),'inherited E023 plan')
    inherited=read(inherited['path']);need(m.same(read(child/'inherited-e023-plan.json'),inherited),'saved E023 plan')
    # Nine real files, independently pinned: no producer import or fabricated plan.
    expected={str(HERE/'qualifier_timing.py'):TIMING_SHA,str(HERE/'qualifier_matrix.py'):m.QUALIFIER_SHA,
        str(HERE/'candidate.py'):m.CANDIDATE_SHA,str(HERE/'qualify.py'):'fe8e1c044e97885ec7c97fddce04a6e34f5b9067be4d982c5f6ad1071c59c801',
        str(HERE.parent/'long_decode_isolation/qualify.py'):'9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e',
        str(HERE.parent/'long_decode_graph/qualify.py'):'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
        str(HERE.parent/'long_decode_graph/candidate.py'):'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
        str(HERE.parent/'pareto_realistic/qualify_native.py'):'26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
        str(HERE.parent/'long_decode_graph/model-source-pins.json'):m.PINS_SHA}
    need(c['sources']==expected and all(sha(path)==value for path,value in expected.items()),'all nine actual source pins')
    pinned={}
    for key in ('source_manifest','model_manifest','fixtures'):
        entry=inherited[key];need(sha(entry['path'])==entry['sha256'],'manifest pin');pinned[key]=read(entry['path'])
    root=Path(inherited['package_root'])/'tutorial_code/miniengine';files=pinned['source_manifest']['files']
    need(c['model_source_receipt']==dict(root=str(root.resolve()),manifest_sha256=m.PINS_SHA,files=files)
         and files==read(HERE.parent/'long_decode_graph/model-source-pins.json')['files']
         and set(files)=={str(x.relative_to(root)) for x in root.rglob('*.py')}
         and all(sha(root/path)==value for path,value in files.items()),'loaded core source tree')
    need(pinned['model_manifest']['model']==pinned['fixtures']['model']==inherited['model']
         and pinned['model_manifest']['revision']==Path(inherited['model']).name
         and c['mlx_environment']==inherited['execution']['environment'],'model/runtime identity receipts')
    diagnostic=[]
    for f in pinned['fixtures']['fixtures']:
        ids=f['prompt_ids'];diagnostic.append({**f,'fixture_id':'diagnostic-repeat-'+f['fixture_id'],
            'source_fixture_id':f['fixture_id'],'prompt_ids':(ids*((1023+len(ids)-1)//len(ids)))[:1023],
            'diagnostic_only':True,'normal_eos':True,'max_output_tokens':4})
    need(len(diagnostic)==8 and m.same(read(child/'diagnostic-inputs.json'),diagnostic),'actual repeated diagnostic input')
    return p


def scan(m,child,c,parameters):
    path=child/'events.jsonl';stat=path.stat();digest=hashlib.sha256();elapsed=-1.;count=0
    configs=[];samples=[];result=[]
    def take(event,**fields):
        nonlocal elapsed,count
        line=stream.readline();need(line.endswith(b'\n') and len(line)<4<<20,'closed bounded ledger line');digest.update(line);count+=1
        row=json.loads(line);need(row['event']==event and all(m.same(row[k],v) for k,v in fields.items()),'ordered event '+event)
        when=row['elapsed_wall_s'];need(positive(when,True) and when>=elapsed,'monotonic event time');elapsed=when;return row
    def compare(rows,label,kind,b):
        m.checks(rows,[(name,label,shape,m.DTYPES[dtype]) for name,(shape,dtype) in m.layout(b,kind).items()])
    def memory(phase):
        row=take('memory',phase=phase);need(all(type(row[k]) is int and row[k]>=0 for k in ('active','cache','peak')),'memory receipt')
    with path.open('rb') as stream:
        memory('before-load-reference');memory('before-load-candidate')
        row=take('loaded-weights');m.checks(row['checks'],[('root/'+name,'independent-loaded-weights',v['shape'],v['dtype']) for name,v in parameters.items()])
        previous=0
        for used in (257,1023,1024):
            end=used-1
            for i in range(8):
                row=take('prefill',row=i,begin=previous,end=end)
                # This producer has only matrix prefill; helper's stage is an explicit adapter argument.
                m.prefill_checks(dict(row,stage='matrix'),'matrix',i,previous,end)
            previous=end
            if used==257:continue
            for b in (1,8):
                canonical=take('canonical',B=b,used=used)['hashes'];m.hashes(canonical,b,'input')
                for key,values in (('positions',[used-1]*b),('lens',[used]*b)):
                    need(canonical['arrays']['root/'+key]['sha256']==m.int_hash(values),'actual canonical context')
                oracle=None
                for enabled in (False,True):
                    name=f'B{b}-used{used}-gates'+('ON' if enabled else 'OFF');memory('before-'+name);outputs={}
                    for arm in ('A','C'):
                        row=take('qualification',name=name,arm=arm,status='PASS')
                        for key in ('input_before','input_after'):m.hashes(row[key],b,'input');need(m.same(row[key],canonical),'qualification original inputs unchanged')
                        m.hashes(row['output'],b,'output');compare(row['checks'][:131],name+'/'+arm+'/canonical','input',b)
                        if oracle is None:need(len(row['checks'])==131,'OFF eager canonical checks only');oracle=row['output']
                        else:
                            compare(row['checks'][131:],name+'/'+arm+'/OFF-eager-oracle','output',b)
                            need(m.same(row['output'],oracle),'gates ON/C exact against OFF eager original-byte hashes')
                        expected_route=dict(kind='eager',bound=used) if arm=='A' else dict(invocations=1,key=repr((1023 if used<1024 else 1536,((enabled,False),)*48)),bound=1023 if used<1024 else 1536)
                        need(m.same(row['route'],expected_route),'actual eager/compiled gates route witness');outputs[arm]=row['output']
                    config_samples=[]
                    for warm,order in ((True,'ACCA'),(False,'ACCA'*3)):
                        for index,arm in enumerate(order):
                            measured=take('sample-measured',name=name,arm=arm,index=index,warm=warm,status='RUNNING')
                            row=take('sample',name=name,arm=arm,index=index,warm=warm,status='PASS')
                            need(all(m.same(row[k],v) for k,v in measured.items() if k not in ('event','status','elapsed_wall_s')),'measured interval binds verified/released sample')
                            need(not row.get('error') and not row.get('cleanup_error'),'sample cleanup/validation failure')
                            compare(row['checks'][:129],name+'/sample-output','output',b)
                            compare(row['checks'][129:],name+'/sample-input-unchanged','input',b)
                            recorded={k:v for k,v in row.items() if k not in ('checks','elapsed_wall_s')};samples.append(recorded);config_samples.append(recorded)
                    summary=metrics(config_samples,b);result.append(dict(name=name,B=b,used=used,gates=enabled,**summary))
                    configs.append(dict(name=name,status='PASS',B=b,used=used,gates=enabled,qualification_output_hashes=outputs));memory('after-'+name)
        need(not stream.read(1),'no extra ledger event')
    need((path.stat().st_size,path.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns),'stable ledger')
    need(m.same(c['configs'],configs) and m.same(c['samples'],samples),'terminal configs/all128 sample binding')
    need(elapsed<=c['elapsed_wall_s'],'events precede terminal')
    return dict(configs=result,ledger_sha256=digest.hexdigest(),ledger_lines=count,qualification_output_checks=12*129,qualification_input_checks=16*131,
        prefill_checks=24*129,sample_checks=128*260,warm_samples=32,formal_samples=96)


def audit(child,plan_sha256,matrix_audit=DEFAULT_PROOF):
    child=Path(child).resolve();result=dict(schema='E024-route-timing-audit-v1',status='INCOMPLETE',diagnostic_complete=False,
        full_model_qualified=False,performance_qualified=False,http_qualified=False,pareto_eligible=False,
        numerical_scope='Source-pinned device original-bit checks/hash receipts; no success raw tensor snapshots independently recomputed.',
        timing_scope='Forward+eval all129 products+sync only. Copies, qualification, verification and teardown excluded. A_over_C is a diagnostic speedup, not HTTP throughput or latency.',
        owner_resources_audited=False,audit_source_sha256=sha(__file__))
    try:
        m=support();complete_path=child/'complete.json';before=sha(complete_path);c=read(complete_path)
        result['producer_status']=c['status'];result['complete_sha256']=before
        if c['status']!='PASS':
            result.update(status='FAIL' if c['status']=='FAIL' else 'INCOMPLETE',reason='Incomplete/rejected producer: no timing acceptance.');return result
        need(c['schema']=='E024-route-timing-result-v1' and c['diagnostic_complete'] is True and c['source_model_pins_rechecked'] is True
             and all(c[k] is False for k in ('full_model_qualified','performance_qualified','http_qualified','pareto_eligible'))
             and c['device_run'] is True and c['device']==dict(name='Apple M5 Ultra',architecture='applegpu_g17d',version='0.32.3')
             and not c.get('error') and not c.get('cleanup_error'),'strict bounded diagnostic completion')
        p=identities(m,child,c,plan_sha256,matrix_audit)
        need(positive(c['elapsed_wall_s']) and c['elapsed_wall_s']<=p['deadline_s'],'child deadline')
        contract=c['contract'];expected=dict(schema='E024-route-timing-contract-v1',points=[dict(B=b,used=u) for b,u in POINTS],gate_order=['off','on'],
            warm_arms=list('ACCA'),formal_arms=list('ACCA')*3,expected_configs=8,expected_warm_samples=32,expected_formal_samples=96,
            output_arrays=129,input_arrays=131,KV_backing=1536,prefill_checkpoints=[256,1022,1023],cache_limit_gib=2)
        expected.update({k:True for k in ('original_dtype','warm_excluded','compile_excluded','observer_off_timing','no_speed_acceptance_threshold')})
        expected.update({k:False for k in ('full_model_qualified','performance_qualified','http_qualified','pareto_eligible')})
        need(all(m.same(contract[k],v) for k,v in expected.items()),'fixed eight-config timing contract')
        parameters=m.parameters(m.raw_reader(),child,c);summary=scan(m,child,c,parameters)
        need(sha(complete_path)==before and sha(child/'plan.json')==plan_sha256,'stable terminal and plan')
        result.update(status='PASS',diagnostic_complete=True,plan_sha256=plan_sha256,matrix_audit_sha256=MATRIX_PROOF_SHA,
            parameter_count=len(parameters),model_payload_rehashed=False,**summary)
    except FileNotFoundError as error:result['reason']=str(error)
    except Exception as error:result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--matrix-audit',type=Path,default=DEFAULT_PROOF);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=audit(args.child,args.plan_sha256,args.matrix_audit)
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
