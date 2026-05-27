# Experiment 1 — Bootstrap (Baseline)

**Date**: 2026-05-27
**Description**: Initial baseline measurement of production transpose_to_patch kernel
**Status**: ACCEPT

## Results

| Problem Size | Kernel_ms |
|-------------|-----------|
| H=128 | 1.282 |
| H=224 | 1.851 |
| H=320 | 2.338 |
| H=448 | 4.695 |
| H=544 | 4.785 |
| H=672 | 7.306 |
| H=800 | 9.772 |
| H=896 | 12.556 |

**Production target (H=448)**: 4.695ms
**Correctness**: PASS (all sizes)

## Analysis
- Inner loop: 1536 iterations per work-item with 7 integer divisions per element
- Total divisions per dispatch: 784 * 1536 * 7 = 8.4M (for H=448)
- Adreno 663 has no hardware division unit — this is the primary bottleneck
