"""Original raw restoration audit, actual diagnostic owner terminal schema."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bench.m5_ultra_256gb.c1_profile_cycle_v1 import _reuse as reuse
_original=reuse.install(globals(),'gdn_graph_combined_formal/audit_restore.py')
OWNER_SHA=reuse.sha(HERE/'owner.py')
CONTROLLER_SHA=reuse.sha(HERE/'controller.py')
ADAPTED_REAP=reuse.variant(globals(),_original.owner_reap,[("complete['schema']=='GDN-graph-combined-http-formal-owner-result-v1'","complete['schema']=='C1-profile-owner-result-v1'")])
_original_main=main

def main(argv=None):
    if argv==[] or (argv is None and len(sys.argv)==1):print('DESCRIPTION_ONLY: offline diagnostic restoration audit.');return 0
    return _original_main(argv)
if __name__=='__main__':raise SystemExit(main())
