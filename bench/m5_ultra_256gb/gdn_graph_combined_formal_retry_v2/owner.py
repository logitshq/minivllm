"""Versioned retry namespace; original twelve-trial algorithm and factory.

The v1 driver passed an unsupported --run and started no owner. This owner keeps
the actual three-argument CLI. Frozen run/full_prepare code objects are rebound
to this real module, never patched in their original module. No device action
occurs on import. Source/plan preparation remains explicitly caller initiated.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

ORIGINAL = Path(__file__).resolve().parent.parent / 'gdn_graph_combined_formal'
ORIGINAL_OWNER_SHA = '511b48fbe722ea031a7b4586439eaa67f25c9c9514af12b614de74fb8c53c103'


def _load_original():
    path = ORIGINAL / 'owner.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_OWNER_SHA:
        raise ValueError('frozen combined formal owner changed')
    spec = importlib.util.spec_from_file_location('_combined_formal_v2_original_owner', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


_original = _load_original()
for _name, _value in vars(_original).items():
    if not _name.startswith('__'): globals()[_name] = _value
HERE = Path(__file__).resolve().parent
PLAN_SCHEMA = 'GDN-graph-combined-http-formal-owner-plan-v2'
JOURNAL_SCHEMA = 'GDN-graph-combined-http-formal-cold-journal-v2'
SUPPORT = dict(_original.SUPPORT)
SUPPORT.update({
    '../gdn_graph_combined_formal/owner.py': ORIGINAL_OWNER_SHA,
    '../gdn_graph_combined_formal/controller.py': '3b8b3033269fcd326e0a52e20470ee27faf65d313be3e3bf0da8881f26f3ac4c',
    '../gdn_graph_combined_formal/audit_restore.py': '157d6470a9cb87223c419a3f4edd82e8556b3714a7157ea1ad505270c8ed7493',
    '../gdn_graph_combined_formal/audit_formal.py': 'b1956036eb7c6df3939e68e2ba5b69b965bfb80951c694107325adb6feed3e5f',
})
for _name, _value in vars(_original).items():
    if isinstance(_value, FunctionType) and _value.__globals__ is vars(_original):
        _fn = FunctionType(_value.__code__, globals(), _name, _value.__defaults__, _value.__closure__)
        _fn.__kwdefaults__ = _value.__kwdefaults__
        globals()[_name] = _fn


def load(name):
    if name == 'formal':
        return module(ORIGINAL / 'formal.py', FORMAL_SHA, 'combined_formal_v2_original_factory')
    return donor().load(name)


if __name__ == '__main__': raise SystemExit(main())
