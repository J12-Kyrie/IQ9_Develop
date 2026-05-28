# Decision Context — Post exp_16 (Continue Optimization)

## Current State
- **Best performance**: 0.915ms at H=448 (exp_16, WG=8)
- **Baseline**: 4.695ms (exp_1)
- **Improvement**: -80.5% vs baseline (5.14x speedup)
- **Previous target**: sub-1ms ← ACHIEVED
- **New goal**: Push for maximum performance
- **Correctness**: PASS at all sizes (H=128 to H=896)

## Full Benchmark (WG=8)
| Size | Kernel_ms |
|------|-----------|
| H=128 | 0.148 |
| H=224 | 0.260 |
| H=320 | 0.559 |
| H=448 | 0.914 |
| H=544 | 1.243 |
| H=672 | 1.783 |
| H=800 | 2.453 |
| H=896 | 3.041 |

## Experiment History
| Exp | Description | Result | Improvement |
|-----|-------------|--------|-------------|
| 1 | Baseline | 4.695ms | - |
| 11 | vload3 + block_base (WG=auto) | 3.079ms | -34.4% |
| 15 | Local memory tiling (static __local) | FAIL | correctness bug |
| 16 | WG=8 | 0.915ms | -70.3% vs exp_11 |

## Known Dead Ends
- float4 loads: Adreno 663 compiler bug
- Multi-#define divisors: compiler garbage
- Block-level dispatch (exp_12): reduces parallelism
- Incremental row pointer: no improvement
- Bitwise merge decode: no improvement
- Local memory tiling (exp_15): correctness FAIL (static __local)

## Unexplored Directions
1. **Debug local memory tiling**: The approach is sound but had correctness issues. Need to fix.
2. **Output coalescing**: Write output with vector stores (vstore4/vstore3)
3. **2D dispatch**: Use (seqIdx, channel_group) to parallelize input_dim
4. **Register blocking**: Process multiple patches per WI
5. **Kernel splitting**: Separate read-transpose-write phases
6. **WG=4**: Test if even smaller WG helps (WG=8 was best so far)

## Key Question
Which direction has the highest ceiling for further improvement?
