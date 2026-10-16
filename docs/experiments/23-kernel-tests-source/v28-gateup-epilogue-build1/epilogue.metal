// Sigmoid expression and typed intermediate assignments match MLX0.32.3
// unary_ops.h:295-300 and compiled.cpp:191-207; Apple Inc. MIT license.
struct v28_sigmoid {
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
template <typename T>
inline T v28_swiglu(T gate, T up) {
  T sigmoid_value = v28_sigmoid{}(gate);
  T silu_value = gate * sigmoid_value;
  T result = silu_value * up;
  return result;
}
