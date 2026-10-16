#!/bin/zsh
set -eu
probe_dir=${0:A:h}
mlx_root=${1:-/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx}
probe_sdk=$(xcrun --sdk macosx --show-sdk-path)
/Library/Developer/CommandLineTools/usr/bin/clang++ -std=c++20 -O2 -arch arm64 -isysroot "$probe_sdk" \
  -I"$mlx_root/include" -I"$mlx_root/include/metal_cpp" \
  "$probe_dir/probe.cpp" -L"$mlx_root/lib" -lmlx \
  -Wl,-rpath,"$mlx_root/lib" -framework Metal -framework Foundation \
  -o "$probe_dir/probe-native-reduction"
