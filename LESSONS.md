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
