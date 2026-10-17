"""Static startup policy only. No inference imports, live request inspection or migration."""

def choose(concurrency=1, *, enable_full_curve=False):
    if type(concurrency) is not int or concurrency not in (1,2,4):
        raise ValueError('only declared C1/C2/C4 points')
    if type(enable_full_curve) is not bool:
        raise TypeError('strict boolean opt-in required')
    if concurrency!=1 and not enable_full_curve:
        raise ValueError('full-curve startup profile is default OFF')
    return 'direct_c1' if concurrency==1 else 'legacy_batched'
