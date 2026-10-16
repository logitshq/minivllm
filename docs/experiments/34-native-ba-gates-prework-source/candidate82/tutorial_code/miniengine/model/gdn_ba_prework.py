"""Prepared original-pointer V76 production adapter; no runtime layout probing.

Only qualified original input-layernorm B1 -> unchanged conv23 producer is in
scope. Caller observes real X/activated/parameter strides outside capture first.
"""
from ..kernels.v76_gates_prework import gdn_gates_prework_metal as core


class PreparedBA:
    __slots__ = ("operands", "operand_strides", "x_strides", "activated_strides", "parameter_ranges")

    def __init__(self, operands, operand_strides, x_strides, activated_strides, parameter_ranges):
        self.operands = tuple(operands)
        self.operand_strides = tuple(tuple(v) for v in operand_strides)
        self.x_strides = tuple(x_strides)
        self.activated_strides = tuple(activated_strides)
        self.parameter_ranges = tuple(tuple(v) for v in parameter_ranges)

    def prework(self, activated, x):
        return core.prepare(activated, x, *self.operands,
            observed_strides=(self.activated_strides, self.x_strides, *self.operand_strides))

    def state(self, prepared, S):
        return core.state(prepared, S)


def prepare(mx, operands, *, observed_parameter_strides, observed_x, observed_x_strides,
            observed_activated, observed_activated_strides, observed_parameter_ranges,
            runtime_version, device_name, architecture, input_layout_contract):
    if (runtime_version != "0.32.3" or device_name != "Apple M4 Max" or architecture != "applegpu_g16s"
            or input_layout_contract != "qualified-original-input-layernorm-B1-and-conv23"):
        raise ValueError("exact qualified runtime and unchanged producer contract required")
    operands = tuple(operands)
    if len(operands) != 6 or len({id(v) for v in operands}) != 6 or len(observed_parameter_ranges) != 6:
        raise ValueError("six distinct original parameter objects/ranges required")
    core._validate((observed_activated, observed_x, *operands),
        (tuple(observed_activated_strides), tuple(observed_x_strides), *observed_parameter_strides), mx)
    expected_bytes = (122880, 15360, 122880, 15360, 192, 192)
    ranges = []
    for observed, nbytes in zip(observed_parameter_ranges, expected_bytes):
        if (len(observed) != 2 or any(type(x) is not int for x in observed)
                or observed[0] <= 0 or observed[1] != nbytes):
            raise ValueError("actual complete original parameter ranges required")
        ranges.append((observed[0], observed[0] + nbytes))
    if any(a < d and c < b for i, (a,b) in enumerate(ranges) for c,d in ranges[i+1:]):
        raise ValueError("aliased original BA/gate parameters are unsupported")
    return PreparedBA(operands, observed_parameter_strides, observed_x_strides,
                      observed_activated_strides, observed_parameter_ranges)
