"""Preserve E2E build1 startup failure; repair runtime audit in a new domain."""
from pathlib import Path
import shutil
repo=Path(__file__).resolve().parents[2];dest=repo/'.artifacts/m5-ultra-256gb/e2e-build2-harness'
dest.mkdir()
for p in (repo/'bench/m5_ultra_256gb/e2e').glob('*.py'):
 s=p.read_text()
 if p.name=='child.py':s=s.replace('  import mlx.core as mx','  import numpy as np\n  import mlx.core as mx')
 if p.name=='run.py':
  s=s.replace('REPO=D.parents[2]', 'REPO=Path('+repr(str(repo))+')').replace('e2e-inputs-build1','e2e-inputs-build2').replace("/'e2e-build1'", "/'e2e-build2'")
  # Output names are a separate attempt, not a fresh path bypass of the old latch.
  s=s.replace("m5-ultra-256gb/e2e-build1", "m5-ultra-256gb/e2e-build2")
 if p.name in ('test_warm_cpu.py','test_protocol.py'):
  s=s.replace('D.parents[2]', 'Path('+repr(str(repo))+')')
 (dest/p.name).write_text(s)
print(dest)
