"""E052 v2 reuses E045's settled resource checkpoints and exact server cleanup."""
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'concurrency_config_v2/runner.py'
BASE_SHA = 'dfb0b6f07e775249babab6e76c62dbf1ab8012027e3cb46f6c694b8b2b730577'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
spec = importlib.util.spec_from_file_location('_e052v2_owned_resources', BASE)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
sha, write, process, verify_pins, physical = base.sha, base.write, base.process, base.verify_pins, base.physical


def namespace():
    scope = dict(vars(base), HERE=HERE)
    for name in ('run', 'main'):
        fn = getattr(base, name)
        scope[name] = FunctionType(fn.__code__, scope, name, fn.__defaults__, fn.__closure__)
    return scope


async def run(plan, out): return await namespace()['run'](plan, out)
def main(): return namespace()['main']()


if __name__ == '__main__': raise SystemExit(main())
