"""E047 observation-only C8 graph route through the unchanged combined loader.

The old combined qualification entry only admits GDN ON. This private adapter
uses its unchanged main and graph-only installation, with the existing route
QualificationObserver over the actual graph handle. No numerical claim follows.
"""
import importlib.util
import hashlib
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
COMBINED = HERE.parent / 'gdn_graph_combined/http_server.py'
ROUTE = HERE.parent / 'route_decode_graph/http_server_v2.py'
FIXED = {COMBINED: '287f34561c83437823923cf45b67f0aa6811f7eba911ee375f62519ba508ef64',
         ROUTE: '8ed952b4870465718a33e0af7e4ecddbf3fbd6265571e7efd3467b904c5621b7'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok, message):
    if not ok: raise ValueError(message)


def module(path):
    need(path in FIXED and sha(path) == FIXED[path], 'fixed server support changed')
    spec = importlib.util.spec_from_file_location('_e047_' + path.parent.name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def source_pins():
    pins = module(COMBINED).source_pins()
    pins.update({str(path): digest for path, digest in FIXED.items()})
    pins[str(Path(__file__).resolve())] = sha(__file__)
    return pins


def arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    need(argv.count('--gdn-prework') == 1, 'explicit single GDN OFF flag required')
    i = argv.index('--gdn-prework')
    need(argv[i + 1:i + 2] == ['off'], 'E047 must retain GDN OFF')
    del argv[i:i + 2]
    args = module(ROUTE).arguments(argv)
    need(args.phase == 'qualification' and args.max_seqs == 8
         and args.max_batched_tokens == 2048 and args.route_decode_graph == 'on',
         'E047 requires observed C8 graph ON, budget 2048')
    args.gdn_prework = 'off'
    return args


def identity_fields(args, pins, *, installed, route_observer):
    need(args.phase == 'qualification' and args.gdn_prework == 'off'
         and args.route_decode_graph == 'on' and installed is True and route_observer is True,
         'actual graph-only diagnostic installation and observer required')
    base = module(COMBINED).identity_fields(SimpleNamespace(phase='formal',
        gdn_prework='off', route_decode_graph='on'), pins, installed=True, route_observer=False)
    base.update(server_source_path=str(Path(__file__).resolve()),
        server_source_sha256=pins[str(Path(__file__).resolve())], benchmark_phase='qualification',
        diagnostic_only=True, numerical_qualified=False, performance_qualified=False)
    base['combined_candidates'].update(route_observer=True,
        route_events_schema=module(ROUTE).ROUTE_SCHEMA,
        qualification_scope='E047 observed graph-only schedule; independent replay outstanding')
    return base


def observer_type(entry, unused_combined):
    class Observer(entry.QualificationObserver):
        def __init__(self, engine, installation, *args, **kwargs):
            need(installation is not None and installation.graph_handle is not None
                 and not installation.gdn_handles, 'actual graph-only installation required')
            super().__init__(engine, installation.graph_handle, *args, **kwargs)
    return Observer


def main(argv=None):
    original = module(COMBINED).main
    scope = dict(original.__globals__, arguments=arguments, source_pins=source_pins,
                 identity_fields=identity_fields, observer_type=observer_type)
    call = FunctionType(original.__code__, scope, original.__name__, original.__defaults__)
    return call(argv)


if __name__ == '__main__': main()
