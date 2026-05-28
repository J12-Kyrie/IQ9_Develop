# Experiment Summary

| Exp | Date | Description | Kernel_ms (H=448) | Pass |
|-----|------|-------------|-------------------|------|
| 1 | 2026-05-27 | Bootstrap baseline | 4.695 | PASS |
| 2 | 2026-05-27 | Compile-time constants (flat loop) | - | FAIL (Adreno bug) |
| 2f | 2026-05-27 | TPS_PP #define only | 4.632 | PASS |
| 2g | 2026-05-27 | TPS_PP + PATCH_AREA #defines | - | FAIL (Adreno bug) |
| 3 | 2026-05-27 | Nested loops + strength reduction | 4.210 | PASS |
| 4 | 2026-05-27 | float4 vectorized loads | - | FAIL (correctness) |
| 4c | 2026-05-27 | Precomputed bases + ternary | 9.482 | PASS |
| 5 | 2026-05-27 | Manual pw unroll | 4.422 | PASS |
| 5b | 2026-05-27 | Precomputed pw offsets | 4.426 | PASS |
| 6 | 2026-05-27 | vload3 channel reads | - | FAIL (inf values) |
| 7 | 2026-05-27 | Reordered loops for input coalescing | 4.069 | PASS |
| 8 | 2026-05-28 | 4 sequences per work-item | 4.315 | PASS |
| 9 | 2026-05-28 | 2D-style dispatch | 623.157 | PASS |
| 10 | 2026-05-28 | Incremental row pointer (t,c,ph,pw) | 4.196 | PASS |
| 10b | 2026-05-28 | exp_7 + incremental row | 4.257 | PASS |
