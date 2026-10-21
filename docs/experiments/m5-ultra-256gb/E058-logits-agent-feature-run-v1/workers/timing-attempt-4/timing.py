"""Stdlib-only injectable serial timing collector and reducer for E058 local verifier+capture."""
from statistics import median

WIDTHS = (4, 8)
ARMS = ("native", "shared_without_features", "shared_with_features")
GROUPS = 5
REPEATS = 2

def schedule():
    rows = []
    ordinal = 0
    for group in range(GROUPS):
        for repeat in range(REPEATS):
            shift = (group * REPEATS + repeat) % len(ARMS)
            order = ARMS[shift:] + ARMS[:shift]
            for arm in order:
                for w in WIDTHS:
                    rows.append((w, group, repeat, ordinal, arm))
                    ordinal += 1
    return rows

SCHEDULE = schedule()

def collect(callbacks, clock_ns=None, on_sample=None):
    import time as _time
    if clock_ns is None:
        clock_ns = _time.perf_counter_ns
    if not isinstance(callbacks, dict) or set(callbacks.keys()) != {4, 8}:
        raise ValueError("callbacks must be {4:{...},8:{...}}")
    for w in WIDTHS:
        inner = callbacks[w]
        if not isinstance(inner, dict) or set(inner.keys()) != set(ARMS):
            raise ValueError("each width must have 3 arms")
        for a in ARMS:
            if not callable(inner[a]):
                raise TypeError("callback must be callable")
    for w in WIDTHS:
        for arm in ARMS:
            for _ in range(2):
                callbacks[w][arm]()
    rows = []
    ordinal = 0
    for group in range(GROUPS):
        for repeat in range(REPEATS):
            shift = (group * REPEATS + repeat) % len(ARMS)
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
                        on_sample(row)
                    ordinal += 1
    return rows

def reduce(rows):
    if not isinstance(rows, list) or len(rows) != 60:
        raise ValueError("rows must be exactly 60")
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
        if r["width"] not in (4, 8) or r["arm"] not in ARMS:
            raise ValueError("invalid width/arm")
        for k in ("width","group","repeat","ordinal"):
            if type(r[k]) is bool or type(r[k]) is not int:
                raise ValueError(f"{k} must be int")
    actual = [(r["width"], r["group"], r["repeat"], r["ordinal"], r["arm"]) for r in rows]
    if actual != SCHEDULE:
        raise ValueError("rows do not match exact planned schedule")
    result = {}
    for w in WIDTHS:
        group_ratios = []
        for group in range(GROUPS):
            natives = [r["elapsed_ns"] for r in rows if r["width"] == w and r["group"] == group and r["arm"] == "native"]
            shareds = [r["elapsed_ns"] for r in rows if r["width"] == w and r["group"] == group and r["arm"] == "shared_with_features"]
            ratio = median(natives) / median(shareds)
            group_ratios.append(ratio)
        median_ratio = median(group_ratios)
        med_with = median([r["elapsed_ns"] for r in rows if r["width"] == w and r["arm"] == "shared_with_features"])
        med_without = median([r["elapsed_ns"] for r in rows if r["width"] == w and r["arm"] == "shared_without_features"])
        overhead = med_with / med_without
        result[w] = {"group_ratios": group_ratios, "median_ratio": median_ratio, "capture_overhead": overhead}
    return result
