"""E053 independent route receipt extension: B1..8 candidate, zero fallback.

Keep frozen complete-route/checker logic. The only branch-policy changes are
external_decode and the per-layer trace-delta gate. Zero delta is compiled reuse
only after a recorded trace for that same installation/bound/B. These counters
are Python branches during tracing/eager calls, never GPU execution counts.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

B=Path(__file__).resolve().parents[1]
ORIGINAL=B/'gdn_graph_combined/http_audit.py'
ORIGINAL_SHA='e41abc8e5073ca812ffdcfc398471a028a6bbd0e12c03a39ec75b0c0e4498717'
if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()!=ORIGINAL_SHA:raise ValueError('frozen combined auditor changed')
spec=importlib.util.spec_from_file_location('_e053_frozen_combined_audit',ORIGINAL)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
need,same=original.need,original.same


def cursor_type(serving):
    parent=original.cursor_type(serving)
    class Cursor(parent):
        def external_decode(self,phase,B):
            need(phase=='HTTP-C8' and type(B) is int and 1<=B<=8,'registered external batched decode')
            for row in self.counters[0]:row['candidate']+=1

        def route(self,phase,step,label,B,used,L):
            need(phase=='HTTP-C8' and type(B) is int and 1<=B<=8,'actual batched HTTP phase/domain')
            slot=0;graph=1<=B<=8 and L>=1536
            bound=1023 if used<1024 else 1536;key=(slot,bound,B)
            row=self.take('combined-graph-call')
            need(set(row)=={'event','status','installation','phase','step','direct_label','B','used','before','after',
                'actual_invocations','kind','bound','per_layer_candidate_delta','per_layer_fallback_delta'},'exact combined event fields')
            same(row['before'],self.counters[slot],'continuous independently tracked48 counters')
            after=row['after'];need(type(after) is list and len(after)==48,'all48 after counters')
            deltas=[]
            for left,right in zip(self.counters[slot],after):
                need(type(right) is dict and set(right)=={'candidate','fallback'}
                     and all(type(x) is int and x>=0 for x in right.values()),'strict integer after counters')
                deltas.append((right['candidate']-left['candidate'],right['fallback']-left['fallback']))
            need(all(x==deltas[0] for x in deltas) and min(deltas[0])>=0,'uniform nonnegative layer trace')
            candidate,fallback=deltas[0]
            need(fallback==0,'B1..8 candidate must never trace native fallback')
            if graph:
                if candidate:
                    kind='compiled-tracing';self.seen.add(key)
                    self.combined['traced_batches'].append(dict(installation=slot,bound=bound,B=B,python_passes=candidate))
                else:
                    need(key in self.seen,'reuse needs prior actual same-instance/bound/B trace')
                    kind='compiled-reuse'
            else:
                need(candidate==1,'eager batched decode executes once per layer');kind='eager-fallback'
            expected=dict(event='combined-graph-call',status='SUBMITTED',installation=slot,phase=phase,step=step,
                direct_label=label,B=B,used=used,before=self.counters[slot],after=after,actual_invocations=int(graph),
                kind=kind,bound=bound,per_layer_candidate_delta=candidate,per_layer_fallback_delta=fallback)
            same(row,expected,'combined event independent call/trace/batched-branch binding')
            self.counters[slot]=[dict(x) for x in after];self.combined[kind]+=1
            # Call the original numerical route witness checker directly; the
            # old combined parent.route would reconsume this row with B1 policy.
            serving.RouteEvents.route(self,phase,step,label,B,used,L)
    return Cursor


class DualRoutes(original.DualRoutes):
    # Reuse exact original lifecycle/finish, changing only its cursor factory.
    __init__=FunctionType(original.DualRoutes.__init__.__code__,
        dict(vars(original),cursor_type=cursor_type),'__init__',original.DualRoutes.__init__.__defaults__)


route_case_type=original.route_case_type
