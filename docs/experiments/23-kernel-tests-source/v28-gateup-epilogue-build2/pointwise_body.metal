const uint i = thread_position_in_grid.x;
T gate_value = gate[i];
T up_value = up[i];
output[i] = v28_swiglu<T>(gate_value, up_value);
