# Experiment 3 — Nested Loops + Strength Reduction

**Date**: 2026-05-27
**Status**: ACCEPT

## Hypothesis
Replace flat 1536-iteration loop (7 divisions/element) with 4-level nested loop (C × tps × patch_size × patch_size).
Inner loop has ZERO divisions — only increment and compare.

## Key Insight
Adreno 663 has no hardware division unit. The flat loop does 8.4M integer divisions per dispatch.
Nested loops eliminate ALL divisions from the inner loop by decomposing elemIdx into (c, t, ph, pw) via loop counters.

## Results

| Problem Size | exp_1 (ms) | exp_3 (ms) | Speedup |
|-------------|-----------|-----------|---------|
| H=128 | 1.282 | 0.579 | 2.21x |
| H=224 | 1.851 | 1.000 | 1.85x |
| H=320 | 2.338 | 2.262 | 1.03x |
| H=448 | 4.695 | 4.214 | 1.11x |
| H=544 | 4.785 | 3.820 | 1.25x |
| H=672 | 7.306 | 5.619 | 1.30x |
| H=800 | 9.772 | 8.514 | 1.15x |
| H=896 | 12.556 | 11.039 | 1.14x |

**Production target (H=448)**: 4.214ms (was 4.695ms, -10.3%)
**Correctness**: PASS (all sizes)
