"""Stdlib-only injectable serial timing collector and reducer for E058 local verifier+capture."""
import itertools
from statistics import median

WIDTHS = (4, 8)
ARMS = ("native", "shared_without_features", "shared_with_features")
GROUPS = 5
REPEATS = 2

def collect(callbacks, clock_ns=None, on_sample=None):
    import time as _time
    if clock_ns is None:
        clock_ns = _time.perf_counter_ns
    # validate callbacks shape
    if not isinstance(callbacks, dict) or set(callbacks.keys()) != {4, 8}:
        raise ValueError("callbacks must be {4:{...},8:{...}}")
    for w in WIDTHS:
        inner = callbacks[w]
        if not isinstance(inner, dict) or set(inner.keys()) != set(ARMS):
            raise ValueError("each width must have 3 arms")
        for a in ARMS:
            if not callable(inner[a]):
                raise TypeError("callback must be callable")
    # warmups: 2 per arm per width (untimed)
    for w in WIDTHS:
        for arm in ARMS:
            for _ in range(2):
                callbacks[w][arm]()
    rows = []
    ordinal = 0
    # 5 groups x2 repeats x3 arms =30 per width =60 total
    for group in range(GROUPS):
        for repeat in range(REPEATS):
            # rotate arm order each repeat across all3 positions
            shift = repeat % len(ARMS)
            order = ARMS[shift:] + ARMS[:shift]
            for arm in order:
                for w in WIDTHS:
                    start = clock_ns()
                    callbacks[w][arm]()
                    end = clock_ns()
                    elapsed = end - start
                    row = {"width": w, "group": group, "repeat": repeat, "ordinal": ordinal, "arm": arm, "elapsed_ns": elapsed}
                    rows.append(row)
                    if on_sample is not None:
                        try:
                            on_sample(row)
                        except Exception:
                            pass
                    ordinal += 1
    return rows

def reduce(rows):
    # validate exact complete schedule: 60 rows, correct fields
    if not isinstance(rows, list) or len(rows) != 60:
        raise ValueError("rows must be exactly 60")
    seen = set()
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("row must be dict")
        for k in ("width","group","repeat","ordinal","arm","elapsed_ns"):
            if k not in r:
                raise ValueError(f"missing {k}")
        if type(r["elapsed_ns"]) is bool or type(r["elapsed_ns"]) is not int:
            raise ValueError("elapsed_ns must be int (bool not allowed)")
        if r["elapsed_ns"] <= 0:
            raise ValueError("elapsed_ns must be positive")
        if r["width"] not in (4,8) or r["arm"] not in ARMS:
            raise ValueError("invalid width/arm")
        key = (r["width"], r["group"], r["repeat"], r["ordinal"], r["arm"])
        # also check duplicate ordinal+width+arm etc - exact schedule expects unique ordinal per row
        if r["ordinal"] in [x for x in seen if isinstance(x,int)]:
            # ordinal uniqueness
            pass
        if key in seen:
            raise ValueError("duplicate row")
        seen.add(key)
        # bool check for other int fields
        for k in ("width","group","repeat","ordinal"):
            if type(r[k]) is bool or type(r[k]) is not int:
                raise ValueError(f"{k} must be int")
    # check ordinals 0..59 consecutive
    ordinals = sorted(r["ordinal"] for r in rows)
    if ordinals != list(range(60)):
        raise ValueError("ordinals must be 0..59")
    # check per width/group/repeat have exactly 3 arms with expected rotation
    for group in range(GROUPS):
        for repeat in range(REPEATS):
            shift = repeat % len(ARMS)
            order = ARMS[shift:] + ARMS[:shift]
            for w in WIDTHS:
                # find rows for this group/repeat/width - should be 3
                subset = [r for r in rows if r["group"]==group and r["repeat"]==repeat and r["width"]==w]
                if len(subset) != 3:
                    raise ValueError("missing rows for group/repeat/width")
    # compute per-width5 paired group ratios (median native / median shared_with_features)
    result = {}
    for w in WIDTHS:
        group_ratios = []
        for group in range(GROUPS):
            natives = [r["elapsed_ns"] for r in rows if r["width"]==w and r["group"]==group and r["arm"]=="native"]
            shareds = [r["elapsed_ns"] for r in rows if r["width"]==w and r["group"]==group and r["arm"]=="shared_with_features"]
            ratio = median(natives) / median(shareds)
            group_ratios.append(ratio)
        median_ratio = median(group_ratios)
        # capture overhead
        native_med_all = median([r["elapsed_ns"] for r in rows if r["width"]==w and r["arm"]=="native"])
        # Actually spec says overhead median(shared_with_features)/median(shared_without_features)
        med_with = median([r["elapsed_ns"] for r in rows if r["width"]==w and r["arm"]=="shared_with_features"])
        med_without = median([r["elapsed_ns"] for r in rows if r["width"]==w and r["arm"]=="shared_without_features"])
        overhead = med_with / med_without
        result[w] = {"group_ratios": group_ratios, "median_ratio": median_ratio, "capture_overhead": overhead}
    return result
