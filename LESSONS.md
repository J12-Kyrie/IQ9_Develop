# LESSONS.md — transpose_to_patch Branch

## Kernel Profile
- **Operation**: 6D-to-2D transpose [T=2, H, W, C=3] -> [seq_len, input_dim=1536]
- **Bottleneck**: Inner loop with 1536 iterations, 7 integer divisions per element (~8.4M total divisions per dispatch)
- **Adreno 663 has no hardware division unit** — divisions are extremely expensive
- **Performance target**: Reduce from ~4.6ms toward sub-1ms

## Effective Optimizations
(To be filled by /optimize experiments)

## Ineffective Optimizations
(To be filled by /optimize experiments)

## Numerical Traps
- Input is float32 normalized image data (no precision issues expected)
- Output is float32 patch embeddings

## Adreno 663 Compiler Bugs (exp_2–exp_6)

### Bug 1: Compile-time constant divisors (2+ #defines)
- **Trigger**: Using 2+ `#define` constants as divisors in the same kernel (e.g., `#define TPS_PP 512` + `#define PATCH_AREA 256`)
- **Symptom**: Garbage output values (max_diff=3.9e+36)
- **Workaround**: Use only 1 `#define` divisor, or use runtime-computed values
- **Safe**: `#define INPUT_DIM 1536` (loop bound only, not a divisor)

### Bug 2: float4 vectorized loads
- **Trigger**: Casting `__global float*` to `__global float4*` for vectorized loads
- **Symptom**: Wrong output values (max_diff=3.4e+35)
- **Note**: Even with 16-byte aligned addresses, fails on Adreno 663

### Bug 3: vload3
- **Trigger**: Using `vload3(offset, ptr)` to read 3 consecutive floats
- **Symptom**: inf values, 28ms runtime (6x slower than baseline)
- **Note**: vload3 is fundamentally broken on Adreno 663

### Key Insight
The Adreno 663 OpenCL compiler is extremely sensitive to code complexity. Simple scalar loops with runtime values work correctly. Any form of vectorization or compile-time constant optimization triggers compiler bugs.
