"""Actual prepared three-mode cycle: stop, direct owner, finally recovery. Default inert."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bench.m5_ultra_256gb.c1_profile_cycle_v1 import _reuse as reuse
_original=reuse.install(globals(),'glm_flash_cycle/execute.py')
REGISTRAR_SHA=reuse.sha(HERE/'preregister.py')
PREFLIGHT_SOURCE=reuse.variant(globals(),_original.preflight,[
    ("'GLM-cold-load-prepared-receipt-v1'","'C1-profile-prepared-receipt-v1'"),
    ("'GLM-cold-load-journal-v1'","'C1-profile-cold-journal-v1'"),
    ("data['trials']==['glm-official-load']","data['trials']==['off','runner','full']")])
EXECUTE_SOURCE=reuse.variant(globals(),_original.execute,[
    ("'glm-owner'","'diagnostic-owner'"),
    ("scope='One direct official GLM load probe, original caps8/32/32/32 and explicit physical policy; no performance or full-model qualification.'",
     "scope='Three direct C1 diagnostic servers, each eight warm plus eight diagnostic requests; no Pareto qualification.'")])
if __name__=='__main__':raise SystemExit(main())
