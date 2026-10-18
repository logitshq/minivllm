"""Independent retry restoration binding; frozen raw/no-experiment rules.

The original auditor functions use this real module's identity and controller.
No original globals or evidence are rewritten; experiment PASS is not required
for independent restoration PASS. No live/backend/service operations.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

ORIGINAL = Path(__file__).resolve().parent.parent / 'gdn_graph_combined_formal'
ORIGINAL_RESTORE_SHA = '157d6470a9cb87223c419a3f4edd82e8556b3714a7157ea1ad505270c8ed7493'
_path = ORIGINAL / 'audit_restore.py'
if hashlib.sha256(_path.read_bytes()).hexdigest() != ORIGINAL_RESTORE_SHA:
    raise ValueError('frozen combined formal restoration auditor changed')
_spec = importlib.util.spec_from_file_location('_combined_formal_v2_original_restore', _path)
_original = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_original)
for _name, _value in vars(_original).items():
    if not _name.startswith('__'): globals()[_name] = _value
HERE = Path(__file__).resolve().parent
OWNER_SHA = 'f7f44001d11533aa6106d3e5dc5c2635dfb736433f10e7ea680a5a83f2863d9e'
CONTROLLER_SHA = 'ca441479a0d2049bcb4daa5a8009b26824b7557a31be8c5c1c2ca57e01ce33cd'
for _name, _value in vars(_original).items():
    if isinstance(_value, FunctionType) and _value.__globals__ is vars(_original):
        _fn = FunctionType(_value.__code__, globals(), _name, _value.__defaults__, _value.__closure__)
        _fn.__kwdefaults__ = _value.__kwdefaults__
        globals()[_name] = _fn


if __name__ == '__main__': raise SystemExit(main())
