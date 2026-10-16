"""Pure stdlib permutation/address contract; no backend import or arithmetic claim."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from rowblock import BASE, metadata, packed_offset


def main():
    checked = 0
    for n, k in [(8, 512), (16, 1024), (16, 5120), (8, 17408)]:
        for bpl in (1, 8):
            size = n * (k // 512) * 32 * bpl
            source = bytes((i * 29 + i // 251) % 256 for i in range(size))
            packed = bytearray(size)
            used = bytearray(size)
            for row in range(n):
                for step in range(k // 512):
                    for lane in range(32):
                        for part in range(bpl):
                            old = ((row * (k // 512) + step) * 32 + lane) * bpl + part
                            new = packed_offset(row, step, lane, part, k=k, bytes_per_lane=bpl)
                            assert not used[new]
                            used[new] = 1
                            packed[new] = source[old]
            assert all(used)
            for row in range(n):
                for step in range(k // 512):
                    for lane in range(32):
                        old = ((row * (k // 512) + step) * 32 + lane) * bpl
                        new = packed_offset(row, step, lane, 0, k=k, bytes_per_lane=bpl)
                        assert packed[new:new+bpl] == source[old:old+bpl]
                        checked += bpl
    for f in BASE.glob("*.py"):
        compile(ast.parse(f.read_text()), str(f), "exec")
    assert not any(n == "mlx" or n.startswith("mlx.") for n in sys.modules)
    print(json.dumps({"status": "PASS_CPU_LAYOUT_CONTRACT", "checked_bytes": checked,
                      "metadata": metadata(), "MLX_imported": False,
                      "GPU_executed": False, "Metal_compile": "UNKNOWN",
                      "numerical_exactness": "UNKNOWN", "performance": "UNKNOWN"}))


if __name__ == "__main__":
    main()
