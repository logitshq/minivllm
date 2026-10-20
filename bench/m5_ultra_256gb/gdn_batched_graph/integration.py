"""Default-OFF batched GDN + existing graph composition, not device-qualified.

The only mathematical substitution is the original E046 B1–8 adapter. Reuse
E033's fresh-instance ownership and teardown verbatim in an isolated namespace.
Install before tracing, restore after quiescence; this is not live switching.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import FunctionType
import weakref

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'gdn_graph_combined/integration.py'
ORIGINAL_SHA='24f89e06f7e6d01041e6daaead40d1bacd927032a68218f8aa5a495bc092b851'
GDN=HERE.parent/'gdn_prework_batched/candidate.py'
GDN_SHA='acd2a23802296b9e420108203cc215c3170cd1eccd9a859e1a787432f98a5353'


def _load(path,name,digest):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('frozen composition dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


original=_load(ORIGINAL,'_batched_graph_original_composition',ORIGINAL_SHA)
need,Installation=original.need,original.Installation
_installed=weakref.WeakSet()


def source_pins():
    need(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()==ORIGINAL_SHA,'original composition source changed')
    need(hashlib.sha256(GDN.read_bytes()).hexdigest()==GDN_SHA,'batched GDN source changed')
    return {**original.source_pins(),str(GDN):GDN_SHA,
            str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _dependencies():
    source_pins()
    graph,_=original._dependencies()
    return graph,_load(GDN,'_batched_graph_gdn',GDN_SHA)


# These inherited functions reference only the public helpers above, sys and
# this module's own WeakSet. Original module globals/methods remain untouched.
_fresh=FunctionType(original._fresh.__code__,globals(),'_fresh',original._fresh.__defaults__)
install=FunctionType(original.install.__code__,globals(),'install',original.install.__defaults__)
install.__kwdefaults__=dict(original.install.__kwdefaults__)
