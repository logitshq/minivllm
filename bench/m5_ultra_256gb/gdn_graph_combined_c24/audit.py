"""Independent C2/C4 recorded-history audit; source-only until producer seal.

Reuse frozen wire/count/full-array/state/reap mathematics. Explicit new domain
changes are eight requests, actual C2/C4 slot capacity, and independently
reconstructed full-batch plus continuing B1 tail coverage. No backend import,
model, process, service or evidence-writing action occurs on import.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent/'gdn_graph_combined'
DONOR = ORIGINAL/'http_audit.py'
DONOR_SHA = 'e41abc8e5073ca812ffdcfc398471a028a6bbd0e12c03a39ec75b0c0e4498717'

def _module(path, digest, name):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('frozen C24 audit dependency changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

_original = _module(DONOR,DONOR_SHA,'_c24_original_combined_audit')
for _name in ('NATIVE','ROUTE','HELPER','HELPER_SHA','HTTP_AUDIT','HTTP_AUDIT_SHA',
              'BASE','BASE_SHA','SERVING','SERVING_SHA','BASE_PLAN_SHA','helper',
              'need','same','parse','read','sha','pinned','finite','file_identity',
              'Incomplete','http_intervals','tokenizer_and_manifests','route_case_type'):
    globals()[_name] = getattr(_original,_name)
PRODUCER = HERE/'qualify.py'
PRODUCER_SHA = 'UNSEALED'
SUPPORT = {PRODUCER:PRODUCER_SHA,HTTP_AUDIT:HTTP_AUDIT_SHA,BASE:BASE_SHA,
    SERVING:SERVING_SHA,HELPER:HELPER_SHA,DONOR:DONOR_SHA,
    ORIGINAL/'audit.py':'7e7718ddf1e1b6d744ec374e19b5513248f0374306503aeb8908391edb403270'}
# These unsealed adjacent files must also match the future immutable plan's
# complete source map; the independent producer seal is mandatory before PASS.
_spec = importlib.util.spec_from_file_location('_c24_audit_domain',HERE/'domain.py')
domain = importlib.util.module_from_spec(_spec);_spec.loader.exec_module(domain)
_http = _original.load(HTTP_AUDIT,HTTP_AUDIT_SHA)


def case_type(base):
    original = _http.case_type(base)
    class Case(original):
        def __init__(self,recording,fixtures,config):
            C = domain.concurrency(recording['concurrency'])
            super().__init__(recording,fixtures,config)
            # The inherited primitive otherwise initializes eight slots for
            # any non-C1 label; HTTP replaces its temporary jobs separately.
            self.C = C
            self.free = list(range(C-1,-1,-1))
            self._previous_decode = []
            self._continuing_tail = False
        def accept(self,row):
            super().accept(row)
            current = list(row['schedule']['decode'])
            if len(self._previous_decode)>1 and len(current)==1 and current[0] in self._previous_decode:
                self._continuing_tail = True
            self._previous_decode = current
        def finish(self,claimed):
            need(max(self.hist,default=0)==self.C and self._continuing_tail,
                 'actual full B=C and continuing multirow-to-B1 tail required')
            coverage = dict(full_batch=self.C,continuing_tail_B1=self._continuing_tail)
            same(claimed.get('domain_coverage'),coverage,'independently reconstructed C24 coverage')
            core = dict(claimed);core.pop('domain_coverage')
            result = super().finish(core)
            result['domain_coverage'] = coverage
            return result
    return Case


def load(path,digest):
    module = _original.load(path,digest)
    if Path(path)==HTTP_AUDIT:
        original = module.stream_ledger
        stream = FunctionType(original.__code__,dict(original.__globals__,case_type=case_type),
                              original.__name__,original.__defaults__,original.__closure__)
        stream.__kwdefaults__ = original.__kwdefaults__
        return SimpleNamespace(**{**vars(module),'stream_ledger':stream})
    return module


def _variant(name,replacements):
    source = inspect.getsource(getattr(_original,name))
    for old,new,count in replacements:
        need(source.count(old)==count,'fixed C24 audit source anchor changed: '+name)
        source = source.replace(old,new)
    tree = ast.parse(source)
    need(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef),'one audit function')
    exec(compile(tree,__file__,'exec'),globals())
    return source


_cursor_source = _variant('cursor_type',[("('HTTP-C1','HTTP-C8')","('HTTP-C2','HTTP-C4')",2)])
_interval_source = _variant('http_intervals',[
    ("len(group['requests']) == (8 if C == 1 else 16)",
     "len(group['requests']) == domain.request_count(C)",1),
    ('complete8/16 HTTP queue','complete new-domain eight HTTP queue',1)])

class DualRoutes(_original.DualRoutes):
    __init__ = FunctionType(_original.DualRoutes.__init__.__code__,globals(),'__init__',
                            _original.DualRoutes.__init__.__defaults__)


_capture_source = _variant('capture_binding',[
    ('capture.donor.ready_identity','capture.native.ready_identity',1)])
_binding_source = _variant('binding',[
    ('count = 8 if C == 1 else 16','count = domain.request_count(C)',1),
    ("'GDN-graph-combined-recorded-http-result-v1'","'GDN-graph-combined-c24-recorded-http-result-v1'",1),
    ("claimed['qualifier_sha256'] == producer.FIXED[producer.BASE]", "claimed['qualifier_sha256'] == sha(HERE/'pair.py')\n         and claimed['schema'] == 'GDN-graph-combined-c24-pair-result-v1'\n         and claimed['math_source_sha256'] == producer.FIXED[producer.BASE]\n         and claimed['concurrency'] == C\n         and claimed['configuration'] == domain.configuration(numeric,C)",1),
    ("producer.load(NATIVE/'http_qualification.py', '_route_audit_jobs').jobs_for(fixtures, C)",
     'domain.jobs_for(fixtures, C)',1),
    ('exact HTTP8/16 original body jobs','exact new-domain eight original body jobs',1)])
_audit_source = _variant('audit',[
    ("'GDN-graph-combined-recorded-http-independent-v1'","'GDN-graph-combined-c24-recorded-http-independent-v1'",1),
    ("ledger['requests'] == (8 if C == 1 else 16)","ledger['requests'] == domain.request_count(C)",1)])
main = FunctionType(_original.main.__code__,globals(),'main',_original.main.__defaults__)

if __name__=='__main__':raise SystemExit(main())
