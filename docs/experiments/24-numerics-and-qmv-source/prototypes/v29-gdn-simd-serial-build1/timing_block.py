"""V21 stationary block, unchanged timing operations; metadata now1group/2segments."""
import time
from datetime import datetime, timezone
PRIME = 2048
REPEATS = 2048
SEGMENTS = 2
def utc():
    return datetime.now(timezone.utc).isoformat()

def block(mx, invoke, variant):
    """The same prime and synchronizations precede either timed variant."""
    prime_utc, prime_begin = (utc(), time.perf_counter_ns())
    for j in range(PRIME):
        result = invoke(variant, j)
    mx.synchronize()
    prime_end = time.perf_counter_ns()
    start_utc = utc()
    begin = time.perf_counter_ns()
    for j in range(REPEATS):
        result = invoke(variant, j)
    mx.synchronize()
    end = time.perf_counter_ns()
    return ({'variant': variant, 'prime_start_utc': prime_utc, 'prime_start_ns': prime_begin, 'prime_end_ns': prime_end, 'prime_repeats': PRIME, 'start_utc': start_utc, 'end_utc': utc(), 'start_ns': begin, 'end_ns': end, 'total_ns': end - begin, 'ns_per_call': (end - begin) / REPEATS, 'repeats': REPEATS, 'groups_per_call': 1, 'segments_per_call': SEGMENTS}, result)
