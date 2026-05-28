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

## exp_8–exp_10 Findings

### What Does NOT Help
- **4 sequences per work-item (exp_8)**: 4.315ms — slightly slower than exp_7. Extra loop overhead and no spatial locality benefit.
- **2D-style dispatch (exp_9)**: 623ms — catastrophic. 4x more work-items with worse memory access patterns.
- **Incremental row pointer (exp_10, 10b)**: 4.196-4.257ms — slightly slower. Compiler already optimizes row address computation.

### Size-Dependent Performance
- **exp_7** (reordered loops) is best for H<=448
- **exp_3** (nested loops) is better for H>448
- Crossover at ~H=480

| H   | exp_1 (ms) | exp_3 (ms) | exp_7 (ms) | Best  |
|-----|-----------|-----------|-----------|-------|
| 128 | 1.282     | 0.579     | 0.483     | exp_7 |
| 224 | 1.851     | 0.996     | 0.979     | exp_7 |
| 320 | 2.338     | 2.263     | 2.163     | exp_7 |
| 448 | 4.695     | 4.210     | 4.069     | exp_7 |
| 544 | 4.785     | 3.815     | 5.920     | exp_3 |
| 672 | 7.306     | 5.592     | 9.146     | exp_3 |
| 800 | 9.772     | 8.493     | 16.106    | exp_3 |
| 896 | 12.556    | 11.047    | 19.809    | exp_3 |

### Adreno 663 Optimization Constraints Summary
1. No vectorization (float4, vload3 broken)
2. No 2+ compile-time constant divisors (#define)
3. No complex index precomputation (ternary operators, many locals)
4. Simple scalar loops with runtime values work best
5. Memory bandwidth is the primary bottleneck (~13.7MB data movement)
6. Loop order matters: (t, ph, pw, c) with c innermost best for small images
