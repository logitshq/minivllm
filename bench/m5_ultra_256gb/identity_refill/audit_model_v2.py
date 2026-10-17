"""E028 audit v2: preserve independently reconstructed slots across pair reuse.

The immutable v1 auditor supplies every source, 37-request, numerical, array,
refill and completion check. Only its private CaseAudit constructor is extended:
lifecycle reuses the drained D-C8 scheduler's free-slot stack, just as the
producer reuses that engine pair. Fresh D-C1/D-C8 pairs keep the original reset.
No expected state is seeded from a lifecycle evidence row. Stdlib/offline only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1_PATH = HERE / 'audit_model.py'
V1_SHA = 'd46c60266e70b7998e2ebeb17d1c126ef4eb896226857419ba6146ae1b642c67'


def private_v1():
    if hashlib.sha256(V1_PATH.read_bytes()).hexdigest() != V1_SHA:
        raise ValueError('frozen E028 v1 numerical auditor changed')
    spec = importlib.util.spec_from_file_location('_e028_private_v1_audit', V1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_factory = module.extended_case

    def extended_case(base, refills, c8_steps):
        original = original_factory(base, refills, c8_steps)
        previous = None

        class ReusedPairCase(original):
            def __init__(self, *args, **kwargs):
                nonlocal previous
                super().__init__(*args, **kwargs)
                if self.name == 'lifecycle':
                    module.need(previous is not None and previous.name == 'D-C8'
                                and previous.C == self.C == 8, 'lifecycle requires preceding reused C8 pair')
                    module.need(previous.next == len(previous.jobs) and not previous.requests
                                and not previous.waiting and not previous.running and not previous.staging
                                and previous.previous_ahead is None and not previous.previous_arrays
                                and previous.previous_batch == 0 and not previous.old_ids
                                and not previous.old_lens and previous.capacity == 0,
                                'reused C8 predecessor must be fully drained')
                    module.need(all(type(slot) is int for slot in previous.free)
                                and sorted(previous.free) == list(range(self.C)),
                                'reused C8 predecessor must have each free slot exactly once')
                    base.same(self.finished, previous.finished, 'reused C8 pending retire IDs must carry unchanged')
                    # Copy: lifecycle pop/append must not mutate the preceding
                    # case, which stream_ledger checks again in its final pass.
                    self.free = list(previous.free)
                previous = self

        return ReusedPairCase

    module.extended_case = extended_case
    return module


def audit(child_dir):
    result = dict(status='FAIL', qualified=False, ledger_opened=False, performance_qualified=False)
    try:
        result = private_v1().audit(child_dir)
    except Exception as exc:
        result['first_failure'] = dict(type=type(exc).__name__, message=str(exc))
    result.update(schema='E028-model-independent-audit-v2',
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  frozen_v1_source_sha256=V1_SHA,
                  correction='Copy independently reconstructed drained C8 free-slot stack into lifecycle; no evidence-derived seed.',
                  scope='Recorded full-model same-schedule numerical checks only; no HTTP/performance or owner/service qualification.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.out.parent.is_dir() or args.out.exists():
        parser.error('new output path with existing parent required')
    result = audit(args.child)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], qualified=result['qualified'])))
    return int(not result['qualified'])


if __name__ == '__main__':
    raise SystemExit(main())
