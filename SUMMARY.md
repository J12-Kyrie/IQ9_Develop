# Experiment Summary

| Exp | Date | Description | Kernel_ms (H=448) | Pass |
|-----|------|-------------|-------------------|------|
| 1 | 2026-05-27 | Bootstrap baseline | 4.695 | PASS |
| 2 | 2026-05-27 | Compile-time constants (flat loop) | - | FAIL (Adreno bug) |
| 2f | 2026-05-27 | TPS_PP #define only | 4.632 | PASS |
| 2g | 2026-05-27 | TPS_PP + PATCH_AREA #defines | - | FAIL (Adreno bug) |
| 3 | 2026-05-27 | Nested loops + strength reduction | 4.214 | PASS |
