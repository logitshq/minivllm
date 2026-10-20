"""Default-OFF batched GDN+graph HTTP entry; frozen worker/SSE/loader/cleanup.

Qualification retains separate schedule, route and per-layer Python trace
receipts. Formal mode forbids observers. Neither mode grants performance or
HTTP-history qualification; the external owner supplies those prerequisites.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent; B = HERE.parent
COMBINED = B/'gdn_graph_combined/http_server.py'
COMBINED_SHA = '287f34561c83437823923cf45b67f0aa6811f7eba911ee375f62519ba508ef64'
BATCHED = B/'gdn_batched_graph/integration.py'
BATCHED_SHA = '6e2e4d4350c36476292f2d0ff2bc7e095561f76a2c233b1a99575027378d50b7'
ROUTE_SCHEMA = 'GDN-graph-combined-http-events-v1'
MLX = {'MLX_MAX_OPS_PER_BUFFER':'50', 'MLX_MAX_MB_PER_BUFFER':'50',
       'MLX_BFS_MAX_WIDTH':'20', 'MLX_METAL_FAST_SYNCH':'0'}
TRACE_SCOPE = 'Actual per-layer Python branch deltas including tracing; not GPU calls or completed arithmetic.'


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path, digest, name, *, register=False):
    need(sha(path) == digest, 'frozen HTTP support changed: '+str(path))
    if register: need(name not in sys.modules, 'private HTTP module already loaded')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if register: sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except BaseException:
        if register: sys.modules.pop(name, None)
        raise
    return module

def original(): return load(COMBINED, COMBINED_SHA, '_batched_http_original_contract')

def source_pins():
    old = original(); pins = old.source_pins()
    pins.update(load(BATCHED, BATCHED_SHA, '_batched_http_install_contract').source_pins())
    pins.update({str(COMBINED):COMBINED_SHA, str(Path(__file__).resolve()):sha(__file__)})
    return pins


def arguments(argv=None):
    args = original().arguments(argv)
    need(args.max_seqs == 8 and args.max_len == 2048 and args.max_batched_tokens == 2048
         and args.cache_limit_gib == 2, 'fixed C8/context2048/budget2048/cache2GiB')
    # Original parser already requires qualification both ON and separate traces,
    # and forbids either observer in formal mode. Retain formal switch ablations.
    return args


def identity_fields(args, pins, *, installed, route_observer):
    old = original()
    identity = old.identity_fields(args, pins, installed=installed, route_observer=route_observer)
    identity.update(server_source_path=str(Path(__file__).resolve()), server_source_sha256=sha(__file__),
                    numerical_qualified=False, full_state_qualified=False, performance_qualified=False)
    identity['combined_candidates'].update(schema='GDN-batched-graph-http-adapter-v1',
        gdn_candidate_domain='B1..8', integration_path=str(BATCHED), integration_sha256=BATCHED_SHA,
        trace_policy_source_path=str(Path(__file__).resolve()), trace_policy_source_sha256=sha(__file__),
        trace_counter_scope=TRACE_SCOPE,
        qualification_scope='New actual candidate HTTP capture requires its own complete replay and independent audit')
    return identity


def batched_delta(before, after, batch):
    need(type(batch) is int and 1 <= batch <= 8, 'registered batched GDN domain')
    need(len(before) == len(after) == 48, 'all48 installed GDN layers')
    rows = []
    for left, right in zip(before, after):
        need(set(left) == set(right) == {'candidate','fallback'}
             and all(type(v) is int and v >= 0 for row in (left,right) for v in row.values()), 'integer layer counters')
        rows.append((right['candidate']-left['candidate'], right['fallback']-left['fallback']))
    need(all(row == rows[0] for row in rows) and min(rows[0]) >= 0, 'mixed or stale layer trace deltas')
    need(rows[0][1] == 0, 'batched candidate must not take native fallback')
    return rows[0]  # Zero on compiled reuse is expected, not a failed device call.


def trace_type(combined):
    namespace = dict(vars(combined), delta=batched_delta)
    class TraceGraph(combined.TraceGraph):
        __call__ = FunctionType(combined.TraceGraph.__call__.__code__, namespace, '__call__',
                                combined.TraceGraph.__call__.__defaults__)
    return TraceGraph


def observer_type(entry, combined):
    return original().observer_type(entry, SimpleNamespace(TraceGraph=trace_type(combined)))


def execution_main(donor):
    def load_fixed(path, name):
        digest = BATCHED_SHA if path == BATCHED else donor.FIXED[path]
        return load(path, digest, name, register=True)
    namespace = dict(vars(donor), INTEGRATION=BATCHED, arguments=arguments, source_pins=source_pins,
                     identity_fields=identity_fields, observer_type=observer_type, load_fixed=load_fixed)
    return FunctionType(donor.main.__code__, namespace, 'main', donor.main.__defaults__)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(json.dumps(dict(default_action='describe only', enabled=False,
            phases=['qualification','formal'], numerical_qualified=False, performance_qualified=False)))
        return 0
    need(argv[0] == '--run' and argv.count('--run') == 1, 'explicit --run required before server arguments')
    need({k:v for k,v in os.environ.items() if k.startswith('MLX_')} == MLX, 'fixed MB50/ops50/BFS20/sync0 environment')
    return execution_main(original())(argv[1:])


if __name__ == '__main__': raise SystemExit(main())
