"""Install the pinned, precompiled oMLX baseline into an isolated environment."""
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    if sys.platform != "darwin" or sys.version_info[:2] != (3, 13):
        raise SystemExit("This baseline lock uses the official macOS Python 3.13 wheel.")
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("Install uv before setting up the oMLX baseline.")
    root = Path(__file__).resolve().parents[2]
    runtime = root / ".artifacts/omlx-baseline/.venv"
    python = runtime / "bin/python"
    if not python.exists():
        subprocess.run([uv, "venv", "--python", sys.executable, str(runtime)], check=True)
    lock = Path(__file__).with_name("omlx-requirements.lock")
    subprocess.run([uv, "pip", "sync", "--python", str(python), str(lock)], check=True)
    # Availability is recorded separately from whether a given shape uses a kernel.
    subprocess.run([str(python), "-c",
                    "import json; from omlx.custom_kernels import native_kernel_status; "
                    "s=native_kernel_status(); print(json.dumps(s,indent=2)); "
                    "assert s['qwen35_prefill']['available'] and s['decode_fast']['available']"], check=True)


if __name__ == "__main__":
    main()
