// Copyright © 2023 Apple Inc. MIT License, see LICENSE-MLX-MIT.
// Exact installed MLX 0.32.3 float functor bodies, identifier renaming only.
struct V76FloatLimits { static constexpr constant float max = metal::numeric_limits<float>::infinity(); };
inline float v76_native_log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == V76FloatLimits::max) {
    return V76FloatLimits::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return x * (metal::log(xp1) / (xp1 - 1.0f));
}
struct V76NativeSigmoid {
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct V76NativeExp {
  template <typename T>
  T operator()(T x) thread {
    return metal::precise::exp(x);
  };
};
struct V76NativeNegative {
  template <typename T>
  T operator()(T x) thread {
    return -x;
  };
};
struct V76NativeAdd {
  template <typename T>
  T operator()(T x, T y) thread {
    return x + y;
  }
};
struct V76NativeMultiply {
  template <typename T>
  T operator()(T x, T y) thread {
    return x * y;
  }
};
struct V76NativeLogAddExp {
  template <typename T>
  T operator()(T x, T y) thread {
    if (metal::isnan(x) || metal::isnan(y)) {
      return metal::numeric_limits<T>::quiet_NaN();
    }
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + v76_native_log1p(metal::exp(minval - maxval)));
  };
};
