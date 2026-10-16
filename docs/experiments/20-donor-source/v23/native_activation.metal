// Typed bodies pinned to MLX 0.32.3 unary_ops.h and binary_ops.h.
struct V18NativeSigmoid {
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct V18NativeMultiply {
  template <typename T>
  T operator()(T x, T y) thread {
    return x * y;
  }
};
struct V18BF16Activation {
    bfloat16_t sigmoid_value;
    bfloat16_t activated;
};
inline V18BF16Activation v18_bf16_activation(bfloat16_t u) {
    bfloat16_t sigmoid_value = V18NativeSigmoid()(u);
    bfloat16_t activated = V18NativeMultiply()(u, sigmoid_value);
    return {sigmoid_value, activated};
}
