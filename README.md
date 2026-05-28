# transpose_to_patch — OpenCL Kernel Optimization

Optimization of the `transpose_to_patch` kernel for Qualcomm QCS9075 / Adreno 663 GPU.

## Kernel Description

Transposes normalized image patches from 6D layout `[T=2, H, W, C=3]` to 2D VIT input format `[seq_len=784, input_dim=1536]`. This kernel is part of the Qwen3ViL vision-language model pipeline.

**Input**: `float[T * H * W * C]` — normalized image data
**Output**: `float[seqLength * input_dim]` — patch-embedded VIT input

## Performance Results

| Metric | Value |
|--------|-------|
| **Baseline (exp_1)** | 4.695ms |
| **Best (exp_20c)** | **0.764ms** |
| **Speedup** | **6.14x** (-83.7%) |
| **Theoretical minimum** | ~0.69ms |
| **Gap to theoretical** | 10.7% |

> **Note**: Baseline (exp_1) is the Qualcomm-provided example implementation.

### Full Benchmark (exp_20c)

| Problem Size | seqLength | Kernel_ms | Correct |
|-------------|-----------|-----------|---------|
| H=128, W=128 | 64 | 0.095 | PASS |
| H=224, W=224 | 196 | 0.230 | PASS |
| H=320, W=320 | 400 | 0.414 | PASS |
| **H=448, W=448** | **784** | **0.764** | **PASS** |
| H=544, W=544 | 1156 | 1.112 | PASS |
| H=672, W=672 | 1764 | 1.674 | PASS |
| H=800, W=800 | 2500 | 2.347 | PASS |
| H=896, W=896 | 3136 | 2.915 | PASS |

## Optimization History

| Exp | Description | H=448_ms | Status |
|-----|-------------|----------|--------|
| 1 | **Baseline (Qualcomm Example)** | 4.695 | PASS |
| 3 | Nested loops + strength reduction | 4.210 | PASS |
| 7 | Reordered loops for input coalescing | 4.069 | PASS |
| 8 | 4 sequences per work-item | 4.315 | PASS |
| 9 | 2D-style dispatch | 623.157 | PASS |
| 10 | Incremental row pointer (t,c,ph,pw) | 4.196 | PASS |
| 11 | vload3 + block_base (WG=auto) | 3.079 | PASS |
| 12 | Block-level dispatch + vload3 | 3.027 | PASS |
| 15 | Local memory tiling (static __local) | FAIL | FAIL |
| 16 | **WG size sweep: WG=8 optimal** | **0.915** | PASS |
| 17 | Output write coalescing (3 passes) | 1.404 | PASS |
| 18 | 2 patches per WI | 1.005 | PASS |
| 19 | #pragma unroll 4 on pw | 0.875 | PASS |
| 20 | Manual unroll ph=2 rows | 0.777 | PASS |
| **20c** | **Manual unroll ph=2 + t explicit** | **0.764** | **PASS** |

## Key Findings

### 1. Work-Group Size is Critical
Driver auto-tune (WG=0) selects large WG (~32+) which is catastrophic for this kernel. **WG=8 is optimal**, giving 3.36x speedup over auto-tune.

| WG Size | H=448_ms |
|---------|----------|
| auto | 3.077 |
| 128 | 3.177 |
| 64 | 2.890 |
| 32 | 1.907 |
| 16 | 1.449 |
| **8** | **0.915** |
| 4 | 0.986 |

### 2. Adreno 663 Compiler Bugs
- **float4 loads**: Produce wrong results
- **Multi-#define divisors**: Compiler garbage with 2+ #define divisors
- **vload3**: Works correctly (contrary to initial assumption)

### 3. Loop Unrolling Sweet Spot
- `#pragma unroll 4` on pw loop: ~4% improvement
- Manual unroll ph=2 rows: ~11% improvement
- Manual unroll ph=4 rows: **WORSE** (register spilling)
- Full #pragma unroll: **WORSE** (register spilling)

### 4. What Didn't Work
- Local memory tiling: Correctness issues on Adreno 663
- Output write coalescing (3 passes): Input re-read cost > coalescing benefit
- Multiple patches per WI: Reduced parallelism
- Block-level dispatch: Reduced parallelism for small sizes

## Build & Run

```bash
cd /mnt/workspace/auto-transpose_to_patch-kernel/build
cmake ../host && make -j$(nproc)
./benchmark --quick    # Quick validation
./benchmark --full     # Full benchmark
```

## Platform

- **Device**: Qualcomm QCS9075
- **GPU**: Adreno 663
- **OpenCL**: 3.0
- **Build flags**: `-cl-fast-relaxed-math -cl-mad-enable`
