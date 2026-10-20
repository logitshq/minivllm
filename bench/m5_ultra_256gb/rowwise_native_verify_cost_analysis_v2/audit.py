"""Default-OFF E054 audit v2: fix only canonical receipt path expectation.

The frozen producer snapshots already-flat mx.load dictionaries: their tensor
keys already start root/, so snapshot() gives root/root/. Output snapshots are
nested {logits,state} and remain root/logits, root/state. Raw bytes, qualification,
400-timing statistics and complete lifecycle/source closure are unchanged v1 code.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'rowwise_native_verify_cost_analysis_v1/audit.py'
PREVIOUS_SHA='65b83525aa5e51c2de1a562894706a98df230463a4acd152d5c1ea3b90fd36bf'
if hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()!=PREVIOUS_SHA:
    raise ValueError('frozen v1 audit source changed')
_spec=importlib.util.spec_from_file_location('_e054_cost_audit_frozen_v1',PREVIOUS)
base=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(base)


def producer_checks(complete):
    def expected(label,spec):
        return [dict(check=label,name=name,shape=shape,dtype=base.DTYPES[dtype],reference_finite=True,candidate_finite=True,bit_equal=True)
                for name,(shape,dtype) in spec.items()]
    # Add one prefix only to the EXPECTED canonical map. Never rewrite observed
    # receipts, saved tensor names, or output comparison keys.
    canonical={'root/'+key:value for key,value in base.layout(1,True).items()}
    wanted=[]
    for label in base.expected_labels():
        width=1 if label.endswith('/continuation') else int(label[1])
        wanted.extend(expected(label,base.layout(width)))
    for arm in ('canonical','oracle','host','device','candidate'):
        wanted.extend(expected('canonical-preserved/'+arm,canonical))
    order=lambda row:(row['check'],row['name'])
    base.same(sorted(complete['checks'],key=order),sorted(wanted,key=order),'complete source-bound initial comparisons')
    wanted=[]
    for arm in ('canonical','oracle','host','device','candidate'):
        wanted.extend(expected('post-timing-canonical/'+arm,canonical))
    base.same(sorted(complete['post_checks'],key=order),sorted(wanted,key=order),'complete source-bound post-timing comparisons')


def contract():
    return dict(base.contract(),schema='E054-independent-cost-contract-v2',
                revision='Only canonical producer receipt names use expected root/root/; raw files/timings/closure unchanged.')


_scope=dict(vars(base),producer_checks=producer_checks,contract=contract,__file__=__file__)
_frozen_audit=FunctionType(base.audit.__code__,_scope,'audit',base.audit.__defaults__,base.audit.__closure__)


def audit(child,parent_path,parent_sha):
    result=_frozen_audit(child,parent_path,parent_sha)
    result.update(schema='E054-local-cost-independent-v2',previous_auditor=dict(path=str(PREVIOUS),sha256=PREVIOUS_SHA),
                  correction='Canonical producer receipt name prefix only; no actual receipt or numerical expectation rewritten.')
    return result


_main_scope=dict(vars(base),audit=audit,contract=contract)
main=FunctionType(base.main.__code__,_main_scope,'main',base.main.__defaults__,base.main.__closure__)

if __name__=='__main__': raise SystemExit(main())
