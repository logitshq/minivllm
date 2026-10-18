"""New C2/C4 domain entry; frozen combined worker/handler/candidate are reused.
No backend import or service operation happens on import. The two candidates
remain default OFF; this entry itself grants no qualification.
"""
import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
from types import FunctionType
HERE=Path(__file__).resolve().parent
DONOR=HERE.parent/'gdn_graph_combined/http_server.py'
DONOR_SHA='287f34561c83437823923cf45b67f0aa6811f7eba911ee375f62519ba508ef64'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def module(path,digest,name):
    if sha(path)!=digest:raise ValueError('frozen C24 dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
donor=module(DONOR,DONOR_SHA,'_c24_combined_server')
ROUTE_SCHEMA=donor.ROUTE_SCHEMA

def source_pins():
    return {**donor.source_pins(),str(Path(__file__).resolve()):sha(__file__)}

def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--package-root', required=True, type=Path)
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--source-manifest-sha256', required=True)
    for name in ('max-len', 'max-seqs', 'max-batched-tokens', 'cache-limit-gib'):
        parser.add_argument('--' + name, required=True, type=int)
    for name in ('compile-decode', 'fuse-decode', 'gates-compile', 'norm-reuse', 'tcp-nodelay'):
        parser.add_argument('--' + name, required=True, choices=('off', 'on'))
    parser.add_argument('--io-timeout-seconds', required=True, type=float)
    parser.add_argument('--qualification-trace', type=Path)
    parser.add_argument('--route-events', type=Path)
    parser.add_argument('--phase', choices=('formal', 'qualification'), default='formal')
    parser.add_argument('--route-decode-graph', choices=('off', 'on'), default='off')
    parser.add_argument('--gdn-prework', choices=('off', 'on'), default='off')
    args = parser.parse_args(argv)
    if not (args.max_len > 0 and args.max_seqs > 0 and args.max_batched_tokens >= args.max_seqs
            and args.cache_limit_gib >= 0 and math.isfinite(args.io_timeout_seconds)
            and args.io_timeout_seconds > 0 and 0 < args.port < 65536):
        parser.error('invalid capacity, allocator, timeout or port')
    if args.fuse_decode == 'on' and args.compile_decode != 'on':
        parser.error('fuse-decode requires compile-decode')
    if (args.phase == 'qualification') != (args.qualification_trace is not None):
        parser.error('qualification requires its step trace; formal forbids observers')
    if args.phase == 'qualification' and not (args.gdn_prework == args.route_decode_graph == 'on'):
        parser.error('this qualification entry requires both candidates ON')
    wants_route_events = args.phase == 'qualification'
    if wants_route_events != (args.route_events is not None):
        parser.error('route-events required exactly for qualification with route ON')
    if args.route_events is not None and args.route_events.resolve() == args.qualification_trace.resolve():
        parser.error('route-events and qualification-trace must be separate files')
    if (args.route_decode_graph == 'on' or args.gdn_prework == 'on') and not (
            args.max_len == 2048 and args.max_seqs in (2, 4)
            and args.compile_decode == args.fuse_decode == args.gates_compile == 'on'
            and args.norm_reuse == 'off'):
        parser.error('route ON requires new C2/C4 context2048, short+fuse+gates ON, norm OFF')
    if args.max_seqs not in (2,4):
        parser.error('new entry is only C2/C4; no relabeled C1/C8')
    return args

def identity_fields(args,pins,*,installed,route_observer):
    result=donor.identity_fields(args,pins,installed=installed,route_observer=route_observer)
    result.update(server_source_path=str(Path(__file__).resolve()),server_source_sha256=pins[str(Path(__file__).resolve())],
        combined_entry_donor=dict(path=str(DONOR),sha256=DONOR_SHA),
        qualification_domain=dict(schema='GDN-graph-combined-c24-domain-v1',max_seqs=args.max_seqs,
            normal_eos=True,max_output_tokens=512,requests=8))
    return result

def main(argv=None):
    # Original main and nested worker run under explicit dependencies. Donor
    # module globals, __file__, HERE and actual source provenance stay unchanged.
    namespace=dict(donor.main.__globals__,arguments=arguments,source_pins=source_pins,identity_fields=identity_fields)
    run=FunctionType(donor.main.__code__,namespace,donor.main.__name__,donor.main.__defaults__)
    return run(argv)

if __name__=='__main__':main()
