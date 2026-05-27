// cpu_baseline.cpp — Per-branch CPU reference implementation
// This file is linked into the benchmark harness if present.
// It provides the gold-standard output that GPU kernels are compared against.

#include <vector>
#include <cstdlib>

// CPU reference for hello_kernel: output[i] = input[i] * 2.0f
// Match the exact algorithm of the OpenCL kernel.
void cpu_baseline(const std::vector<float>& input, std::vector<float>& output, int N) {
    output.resize(N);
    for (int i = 0; i < N; i++) {
        output[i] = input[i] * 2.0f;
    }
}
