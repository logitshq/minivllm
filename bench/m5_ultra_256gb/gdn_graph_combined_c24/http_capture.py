"""C2/C4 capture specifications; exact frozen transport and reap coroutine."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType,SimpleNamespace
HERE=Path(__file__).resolve().parent
DONOR=HERE.parent/'gdn_graph_combined/http_capture.py'
DONOR_SHA='9b98abc3d8f383dba9bd25eafade73cea62e198abde0d734a1e0d2b8ed3a58f2'
SERVER=HERE/'http_server.py'
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def local(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
need(sha(DONOR)==DONOR_SHA,'frozen combined capture changed')
donor=local(DONOR,'_c24_combined_capture')
lifecycle=donor.donor
native=lifecycle.donor
LIFECYCLE_PATH,LIFECYCLE_SHA=donor.DONOR,donor.DONOR_SHA
q=native.load('http_qualification')
domain=local(HERE/'domain.py','_c24_domain')
System,no_process,owned_listener,write=donor.System,donor.no_process,donor.owned_listener,donor.write
ROUTE_SCHEMA=donor.ROUTE_SCHEMA

def source_pins():
    server=local(SERVER,'_c24_server_contract')
    return {**server.source_pins(),str(DONOR):DONOR_SHA,str(LIFECYCLE_PATH):LIFECYCLE_SHA,str(HERE/'domain.py'):sha(HERE/'domain.py'),
        str(Path(__file__).resolve()):sha(__file__)}

def expected_identity(numeric, fixtures, source, model, concurrency, transport):
    cfg = domain.configuration(numeric, concurrency); flags = numeric['candidate']
    eos = sorted(fixtures[0]['eos_token_ids'])
    need(all(sorted(f['eos_token_ids']) == eos for f in fixtures), 'consistent fixture EOS sets required')
    configuration = {k: cfg[k] for k in ('max_len', 'max_batched_tokens', 'cache_limit_gib')}
    configuration['max_seqs'] = cfg['max_seqs']
    configuration.update({k: 'on' if v else 'off' for k, v in flags.items()})
    configuration.update(tcp_nodelay='on' if transport['tcp_nodelay'] else 'off',
                         io_timeout_seconds=float(transport['io_timeout_seconds']))
    path = Path(numeric['model']).resolve()
    return {'configuration': configuration, 'source_files': source['files'],
        'source_manifest_sha256': numeric['source_manifest']['sha256'], 'manifest_all_sources_match': True,
        'package_root_loaded': str(Path(numeric['package_root']).resolve() / 'tutorial_code/miniengine'),
        'model_snapshot_path': str(path), 'model_snapshot_name': path.name,
        'model_config_sha256': model['files']['config.json'],
        'server_source_sha256': q.FIXED[q.HERE / 'native_server.py'],
        'step_observer': True, 'per_token_observer': False, 'qualification_trace_enabled': True,
        'drafter': None, 'normal_eos': True, 'eos_ids': eos, 'prefix_cache': False,
        'completion_usage': 'committed target IDs excluding terminal EOS; raw IDs include EOS',
        'text_decode': 'skip_special_tokens=False; terminal EOS excluded',
        'cancellation': 'pending skipped; active cancellation permanently fails this benchmark server',
        'weights_identity_scope': 'external controller must verify model weight pins; this server checks source/config identity'}

def prepare(base_plan_path,base_plan_sha,concurrency,*,python,environment,transport,deadlines):
    domain.concurrency(concurrency)
    server=local(SERVER,'_c24_capture_server')
    facade=SimpleNamespace(**{**vars(q),'jobs_for':domain.jobs_for,'expected_identity':expected_identity})
    def load(name):
        return facade if name=='http_qualification' else native.load(name)
    create=FunctionType(native.prepare.__code__,dict(native.prepare.__globals__,load=load),
        native.prepare.__name__,native.prepare.__defaults__)
    original=create(base_plan_path,base_plan_sha,concurrency,python=python,environment=environment,
        transport=transport,deadlines=deadlines)
    pins=server.source_pins();sources={**original['sources'],**source_pins()}
    command=list(original['command'])
    need(command[3]==str(native.HERE/'native_server.py'),'real original server command')
    command[3]=str(SERVER)
    command+=['--phase','qualification','--gdn-prework','on','--route-decode-graph','on']
    args=SimpleNamespace(phase='qualification',gdn_prework='on',route_decode_graph='on',max_seqs=concurrency)
    identity=dict(original['identity']);identity.update(server.identity_fields(args,pins,installed=True,route_observer=True))
    return {**original,'schema':'GDN-graph-combined-c24-capture-spec-v1','sources':sources,
        'command':command,'identity':identity,'route_events_schema':ROUTE_SCHEMA,
        'qualification_scope':'C2/C4 observed capture only; independent full-state replay/resource/restoration required.'}

def validate(spec):
    need(spec.get('schema')=='GDN-graph-combined-c24-capture-spec-v1','actual new C24 spec')
    expected=prepare(spec['base_plan']['path'],spec['base_plan']['sha256'],spec['concurrency'],
        python=spec['python'],environment=spec['environment'],transport=spec['transport'],deadlines=spec['deadlines'])
    need(donor.same(spec,expected),'C24 capture spec changed')
    need(all(sha(p)==h for p,h in spec['sources'].items()),'C24 source changed')

route_startup=donor.route_startup

async def capture(spec,out,*,checkpoint,backend=None):
    namespace=dict(lifecycle.capture.__globals__,validate=validate,route_startup=route_startup)
    run=FunctionType(lifecycle.capture.__code__,namespace,lifecycle.capture.__name__,lifecycle.capture.__defaults__)
    run.__kwdefaults__=dict(lifecycle.capture.__kwdefaults__ or {})
    result=await run(spec,out,checkpoint=checkpoint,backend=backend)
    result.update(schema='GDN-graph-combined-c24-capture-result-v1',
        lifecycle_source=dict(path=str(LIFECYCLE_PATH),sha256=LIFECYCLE_SHA),
        capture_adapter_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        scope='New C2/C4 capture; not numerical or performance qualification.')
    write(Path(out)/'complete.json',result)
    return result
