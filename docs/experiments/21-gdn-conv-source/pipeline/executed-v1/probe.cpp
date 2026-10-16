// CPU-built only. Running --query-native-pipeline initializes Metal; root GPU owner only.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include "mlx/backend/metal/device.h"
#include "mlx/version.h"

int main(int argc, char** argv) {
  if (argc != 2 || std::string(argv[1]) != "--query-native-pipeline") {
    std::cerr << "Usage (GPU owner only): " << argv[0]
              << " --query-native-pipeline\n";
    return 2;
  }
  try {
    const std::string version = mlx::core::version();
    if (version != "0.32.3") throw std::runtime_error("Requires MLX 0.32.3, found " + version);
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
    const std::string name = "col_reduce_small_1_reduce_sumbfloat16";
    // Official v0.32.3 no-JIT get_reduce_kernel delegates to this exact call.
    // Do not rebuild a shader or override native library compiler options.
    auto* kernel = d.get_kernel(name);
    if (!kernel) throw std::runtime_error("Native pipeline unavailable");
    const size_t m = kernel->maxTotalThreadsPerThreadgroup();
    const size_t x = std::min((size_t(10240) + 4 - 1) / 4, size_t(32));
    const size_t y = std::min(size_t(8), std::min(m / x, size_t(4)));
    if (!y) throw std::runtime_error("Invalid M for native reduction geometry");
    std::cout << "{\n"
      << "  \"schema\": \"v23-native-reduction-pipeline-v1\",\n"
      << "  \"mlx_version\": \"" << version << "\",\n"
      << "  \"device\": \"" << d.mtl_device()->name()->utf8String() << "\",\n"
      << "  \"architecture\": \"" << d.get_architecture() << "\",\n"
      << "  \"kernel\": \"" << name << "\",\n"
      << "  \"factory\": \"native-nojit get_reduce_kernel -> Device.get_kernel\",\n"
      << "  \"native_library_compile_options_overridden\": false,\n"
      << "  \"max_total_threads_per_threadgroup\": " << m << ",\n"
      << "  \"thread_execution_width\": " << kernel->threadExecutionWidth() << ",\n"
      << "  \"static_threadgroup_memory_bytes\": " << kernel->staticThreadgroupMemoryLength() << ",\n"
      << "  \"selector_assumptions\": {\"dtype\":\"bfloat16\",\"shape\":[1,4,10240],\"strides\":[40960,10240,1],\"row_contiguous\":true,\"axis\":1,\"ndim_template\":1,\"index_type\":\"int\"},\n"
      << "  \"assumptions_observed_by_this_tool\": false,\n"
      << "  \"n_reads\": 4,\n"
      << "  \"threadgroup\": [" << x << "," << y << ",1],\n"
      << "  \"grid_threadgroups\": [80,1,1],\n"
      << "  \"sum_y_derived_from_measured_pipeline\": " << y << ",\n"
      << "  \"actual_reduce_dispatch_observed\": false,\n"
      << "  \"dispatches_submitted_by_tool\": 0\n}\n";
  } catch (const std::exception& e) {
    std::cerr << "native reduction pipeline probe failed: " << e.what() << "\n";
    return 1;
  }
}
