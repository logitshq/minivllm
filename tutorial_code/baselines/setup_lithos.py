"""Build a pinned official Lithos revision using an isolated, bundled dependency set."""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

VERSION = "0.1.2"
ARCHIVE_URL = "https://github.com/lithos-ai/lithos-metal/releases/download/v0.1.2/lithos-metal-0.1.2-macos-arm64.tar.gz"
ARCHIVE_SHA256 = "a51b821923b21ed4134979779f0ed77c98cbc9b227fed7627ec620704169a3e3"
ADAPTER_REVISION = "5485ad6bac0a5649d66a347f3e01a655ea29d937"
ADAPTER_SHA256 = "bb11ae3a74de32fb8c20222e17372c70bc067ad5a27dcd21453b43f5df30cae5"
SOURCE_URL = f"https://github.com/lithos-ai/lithos-metal/archive/{ADAPTER_REVISION}.tar.gz"
SOURCE_SHA256 = "2b33f72e22d3b392df72cda547f27ffbc03ea0597e17acde243fb53bb5081c6f"


def checked_download(url, path, expected):
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Official binary requires Apple Silicon macOS 26+")
    runtime = args.workspace.resolve() / ".artifacts/lithos-baseline"
    runtime.mkdir(parents=True, exist_ok=True)
    archive = runtime / "lithos-metal-0.1.2-macos-arm64.tar.gz"
    checked_download(ARCHIVE_URL, archive, ARCHIVE_SHA256)
    with tarfile.open(archive) as bundle:
        bundle.extractall(runtime, filter="data")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to prepare the isolated Python 3.12.13 environment")
    venv = runtime / ".venv"
    if not (venv / "bin/python").exists():
        subprocess.run([uv, "venv", "--python", "3.12.13", "--managed-python", str(venv)], check=True)
    python = venv / "bin/python"
    wheels = runtime / "lithos-metal-0.1.2/wheels"
    subprocess.run([uv, "pip", "install", "--python", str(python), "--no-index",
                    "--find-links", str(wheels), "lithos-metal[serve]==0.1.2"], check=True)
    # The release wheel lacks M4 Max registration and an untied MLX head-name adapter.
    # Build the complete official revision containing both fixes; do not splice kernels.
    source_archive = runtime / f"lithos-source-{ADAPTER_REVISION}.tar.gz"
    checked_download(SOURCE_URL, source_archive, SOURCE_SHA256)
    with tarfile.open(source_archive) as source:
        source.extractall(runtime, filter="data")
    source_dir = runtime / f"lithos-metal-{ADAPTER_REVISION}"
    subprocess.run([uv, "pip", "install", "--python", str(python), "--no-deps", str(source_dir)], check=True)
    lock = subprocess.check_output([uv, "pip", "freeze", "--python", str(python)], text=True)
    (runtime / "installed-requirements.lock").write_text(lock)
    manifest = {"distribution_version": VERSION, "archive_url": ARCHIVE_URL,
                "archive_sha256": ARCHIVE_SHA256, "python": "3.12.13",
                "source_revision": ADAPTER_REVISION,
                "source_archive_url": SOURCE_URL, "source_archive_sha256": SOURCE_SHA256,
                "checkpoint_adapter_sha256": ADAPTER_SHA256,
                "native_runtime": "compiled from the complete official pinned source, no local patches"}
    (runtime / "installation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
