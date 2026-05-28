# Experiment Summary

| Exp | Date | Description | Kernel_ms (H=448) | Pass |
|-----|------|-------------|-------------------|------|
| 1 | 2026-05-27 | Bootstrap baseline | 4.695 | PASS |
| 3 | 2026-05-27 | Nested loops + strength reduction | 4.210 | PASS |
| 7 | 2026-05-27 | Reordered loops for input coalescing | 4.069 | PASS |
| 8 | 2026-05-28 | 4 sequences per work-item | 4.315 | PASS |
| 9 | 2026-05-28 | 2D-style dispatch | 623.157 | PASS |
| 10 | 2026-05-28 | Incremental row pointer (t,c,ph,pw) | 4.196 | PASS |
| 10b | 2026-05-28 | exp_7 + incremental row | 4.257 | PASS |
| 11 | 2026-05-28 | Reference-inspired vload3 + block_base | 3.079 | PASS |
| 12 | 2026-05-28 | Block-level dispatch + vload3 | 3.027 | PASS |
| 13 | 2026-05-28 | vload3 + incremental row | 3.079 | PASS |
| 14 | 2026-05-28 | vload3 + bitwise merge decode | 3.079 | PASS |
| 15 | 2026-05-28 | Local memory tiling (static __local) | FAIL | FAIL |
| 16 | 2026-05-28 | WG size sweep: WG=8 optimal | 0.915 | PASS |
| 17 | 2026-05-28 | Output write coalescing (3 passes) | 1.404 | PASS |
| 18 | 2026-05-28 | 2 patches per WI | 1.005 | PASS |
| 19 | 2026-05-28 | #pragma unroll 4 on pw | 0.875 | PASS |
| 19c | 2026-05-28 | #pragma unroll 8 on pw | 0.907 | PASS |
| 20 | 2026-05-28 | Manual unroll ph=2 rows | 0.777 | PASS |
| 20b | 2026-05-28 | Manual unroll ph=4 rows | 1.131 | PASS |
| 20c | 2026-05-28 | Manual unroll ph=2 + t explicit | 0.764 | PASS |
