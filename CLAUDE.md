# CLAUDE.md — transpose_to_patch Branch

## Kernel: transpose_to_patch
- **Function**: `transpose_to_patch`
- **Purpose**: Transpose normalized image [T,H,W,C] into VIT patch embeddings [seq_len, input_dim]
- **Production params**: T=2, H=448, W=448, C=3, patch_size=16, merge_size=2, temporal_patch_size=2, input_dim=1536
- **seq_len**: 784 (for 448x448)
- **NDRange**: 1D, global_size = seqLength, local_work_size = NULL (driver auto-tune)
- **Performance baseline**: ~4.6ms on production kernel
- **Target**: Sub-1ms

## Non-Negotiable Rules
See parent CLAUDE.md for framework rules.

## Optimization Focus
The primary bottleneck is the inner loop (1536 iterations) with integer divisions per element.
Key optimization directions:
1. Eliminate integer divisions (precomputed lookup tables, strength reduction)
2. Loop unrolling with precomputed coordinates
3. Memory access pattern optimization (coalescing)
