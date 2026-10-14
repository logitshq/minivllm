#!/bin/zsh
# 基线环境（和引擎的 .venv 分开）：mlx-lm 的 server/benchmark，dflash-mlx 的 DFlash2。
# dflash-mlx 要装 GitHub main（PyPI 0.1.8 不认 DFlash2 的 config）。
set -e
cd "$(dirname "$0")/.."
uv venv .venv-mlx --python 3.13
uv pip install --python .venv-mlx/bin/python mlx mlx-lm "dflash-mlx[bench] @ git+https://github.com/bstnxbt/dflash-mlx"
.venv-mlx/bin/python -c "import mlx.core as mx, mlx_lm, dflash_mlx; print('mlx', mx.__version__, 'mlx_lm', mlx_lm.__version__)"
