# Experiment 15 — Local Memory Tiling with Cooperative Work-Group Loading

## Hypothesis

The current kernel (exp_11) has each work-item independently reading all 1536 input floats for its patch via 512 vload3 calls from global memory. This creates massive global memory traffic: 784 work-items x 512 reads = 401,408 global memory transactions per dispatch, with each work-item bearing the full ~200-cycle global memory latency on every read.

By having a work-group cooperatively load the patch data into `__local` memory once, then reading from fast local memory (~4 cycles) for the output computation, we can:
1. Reduce per-work-item global reads from 512 to 48 (10.7x fewer)
2. Achieve better global memory coalescing during the loading phase (32 consecutive vload3 calls per iteration)
3. Trade expensive global reads for cheap local reads in the hot path
4. Maintain the existing output write pattern (sequential, already coalesced)

This is the first experiment to exploit Adreno 663 local memory. If local memory works correctly and performs well, it opens the door to further tiling strategies (multi-patch groups, register blocking).

## Implementation Strategy

**Core idea**: Restructure from "1 work-item = 1 patch, reads everything" to "1 work-group = 1 patch, cooperatively loads, individually computes."

### Phase 1 — Cooperative Loading
All 32 work-items in the group participate in loading the 1536-float patch tile from global memory into `__local` memory. Each work-item loads 48 elements (16 iterations x 3 floats via vload3). Loading is sequential per work-item (coalesced across the group).

### Phase 2 — Barrier
`barrier(CLK_LOCAL_MEM_FENCE)` ensures all data is visible before computation.

### Phase 3 — Output Computation
Each work-item processes 48 elements (every 32nd element, stride = group size). For each element, it reads 3 floats from local memory and writes them to 3 output channels. Output writes are sequential per work-item (coalesced).

### Index Decomposition
Same as exp_11 — sequential seqIdx maps to merged spatial grid positions. No change to the tIdx/hIdx/wIdx/mergeH/mergeW decomposition.

## Work-Group Configuration

- **Group size**: 32 work-items
- **Dispatch**: `ceil(seqLength / 32)` groups, each with global size = `seqLength` (kernel has early-exit `if (seqIdx >= seqLength)`)
- **For H=448**: 784 / 32 = 25 groups (with 24 groups at 32 WIs, 1 group at 16 WIs — or 25 x 32 with early-exit)

### Why 32?
- Maps to Adreno 663 wavefront size (typically 32 or 64 for Qualcomm)
- 32 work-items reading 48 elements each = 1536 total (exact fit for patch tile)
- During cooperative loading: 32 consecutive vload3 calls per iteration = excellent coalescing
- Occupancy: 32KB local / 6KB per group = 5 concurrent groups per SP (no occupancy limit)
- 49 groups for H=448 >> 4 SPs, so all SPs stay fully busy

### Alternative: Group size 64
- Each work-item loads 24 elements, computes 24 elements
- Fewer groups (13), higher per-WI register pressure
- Less coalescing during loading (16 consecutive vload3 per iteration)
- **Tune if 32 underperforms**

## Local Memory Layout

```c
__local float tile[1536];  // 6144 bytes — fits in 32KB with room for 5 concurrent groups
```

Layout matches the output element ordering:
```
tile[0..511]   = channel 0 (R): t=0,ph=0,pw=0 .. t=1,ph=15,pw=15
tile[512..1023] = channel 1 (G): same spatial ordering
tile[1024..1535] = channel 2 (B): same spatial ordering
```

Within each channel, the 512 elements are ordered as: `t * (patch_size^2) + ph * patch_size + pw` — identical to the output write pattern in exp_11.

No padding needed (32 banks x 4 bytes, stride-32 access pattern distributes evenly across banks).

## Cooperative Loading Pattern

```c
// Each work-item loads 48 elements = 16 vload3 calls
for (int i = localId; i < 512; i += groupSize) {
    float3 rgb = vload3(0, input + block_base + (i / (ps*ps)) * HW_C
                        + ((i / ps) % ps) * WC + (i % ps) * C);
    tile[i]        = rgb.x;   // channel 0
    tile[i + 512]  = rgb.y;   // channel 1
    tile[i + 1024] = rgb.z;   // channel 2
}
```

**Access pattern per iteration**:
- Work-item 0: reads `input[block_base + offset_0]`
- Work-item 1: reads `input[block_base + offset_1]`
- ...
- Work-item 31: reads `input[block_base + offset_31]`

Where offsets are consecutive spatial positions in the same row → consecutive global addresses → coalesced.

**Note on index computation**: The inner loop computes the global address for each of the 512 spatial positions. This requires `i / (ps*ps)`, `(i / ps) % ps`, `i % ps` — these are integer divisions on Adreno. However:
- `ps = 16` is a power of 2, so the compiler may optimize to shifts/masks
- This overhead is amortized across 32 work-items (each WI only does 16 iterations)
- The divisions are in the loading phase, not the hot output loop

## Synchronization Points

1. **After cooperative loading**: `barrier(CLK_LOCAL_MEM_FENCE)` — ensures all 1536 floats are visible in local memory before any work-item reads
2. **No barrier needed after output writes** — each work-item writes to independent output locations, no read-after-write dependency

## Full Kernel Pseudocode

```c
__kernel void transpose_to_patch(
    __global const float* input, __global float* output,
    const int T, const int H, const int W, const int C,
    const int patch_size, const int merge_size,
    const int temporal_patch_size, const int input_dim)
{
    int seqIdx = get_global_id(0);
    int localId = get_local_id(0);
    int groupSize = get_local_size(0);

    // ... seqIdx decomposition (same as exp_11) ...
    // ... block_base precomputation (same as exp_11) ...

    __local float tile[1536];

    int tps = temporal_patch_size;
    int ps = patch_size;
    int pp = ps * ps;
    int tpp = tps * pp;
    int WC = W * C;
    int HW_C = H * WC;

    // Phase 1: Cooperative loading
    for (int i = localId; i < 512; i += groupSize) {
        int t = i / pp;
        int r = i % pp;
        int ph = r / ps;
        int pw = r % ps;
        int addr = block_base + t * HW_C + ph * WC + pw * C;
        float3 rgb = vload3(0, input + addr);
        tile[i]        = rgb.x;
        tile[i + 512]  = rgb.y;
        tile[i + 1024] = rgb.z;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Output computation
    int outBase = seqIdx * input_dim;
    for (int localIdx = localId; localIdx < input_dim; localIdx += groupSize) {
        int ch = localIdx / tpp;
        int elemIdx = localIdx % tpp;
        output[outBase + localIdx] = tile[ch * 512 + elemIdx];
    }
}
```

## Expected Impact

### Memory Traffic Analysis

| Metric | exp_11 (current) | exp_15 (proposed) |
|--------|-----------------|-------------------|
| Global reads per WI | 512 vload3 | 16 vload3 (cooperative phase) |
| Global reads per group | 512 x 32 = 16,384 | 16 x 32 = 512 |
| Total global reads (H=448) | 784 x 512 = 401,408 | 25 x 512 = 12,800 |
| Read coalescing | 32 consecutive per iteration | 32 consecutive per iteration (same) |
| Local reads per WI | 0 | 48 (from tile) |
| Global writes per WI | 48 | 48 (unchanged) |
| **Global read reduction** | — | **31.4x fewer** |

### Latency Model

- Local memory read: ~4 cycles
- Global memory read (L1 hit): ~50 cycles
- Global memory read (L1 miss): ~200+ cycles
- Barrier (within SP): ~10-20 cycles

Current: 512 global reads x ~100 cycles (amortized hit/miss) = ~51,200 cycles per WI
Proposed: 16 global reads x ~100 cycles + 48 local reads x 4 cycles + barrier ~20 cycles = ~1,812 cycles per WI
**Theoretical speedup: ~28x on memory path**

In practice, the speedup will be limited by:
- Occupancy (how many groups can be in-flight)
- ALU overhead for index computation in loading phase
- Write bandwidth (unchanged)

### Expected Time

- **Optimistic**: ~1.0-1.2ms (approaching bandwidth limit)
- **Realistic**: ~1.5-2.0ms (accounting for overhead, barrier, index computation)
- **Pessimistic**: ~2.5ms (if Adreno local memory has hidden costs or compiler issues)
- **Speedup vs exp_11**: 35-65% reduction (3.079ms → 1.0-2.0ms)

## Risk Assessment

### Risk 1: Adreno 663 Local Memory Performance (MEDIUM)
- **Concern**: Unknown whether `__local` memory on Adreno 663 provides the expected latency advantage over L1-cached global reads. Tile-based GPUs may have different local memory characteristics.
- **Mitigation**: This is an exploration experiment. If local memory performs no better than cached global reads, the kernel will still be correct and we'll have data on Adreno local memory behavior.
- **Fallback**: If local memory is slow, try direct output without tiling (pure coalescing experiment).

### Risk 2: Division Overhead in Loading Phase (LOW)
- **Concern**: The cooperative loading loop needs `i / pp`, `i / ps`, `i % pp`, `i % ps` for address computation. Integer divisions are expensive on Adreno (no hardware divider).
- **Mitigation**: `ps = 16` is a power of 2 — compiler should optimize to bit shifts/masks. If not, can precompute a lookup table or use `__constant` memory for the 512 addresses.
- **Fallback**: Precompute all 512 addresses once using bitwise operations for power-of-2 patch_size.

### Risk 3: Compiler Bugs with __local (MEDIUM)
- **Concern**: Adreno 663 compiler has shown sensitivity to code complexity (LESSONS.md). Adding `__local` arrays and barriers may trigger new compiler bugs.
- **Mitigation**: Keep the kernel structure simple. Use only one `__local` array. Avoid combining `__local` with `#define` constants (Bug 1).
- **Validation**: Run correctness check (tolerance=0.001) before benchmarking.

### Risk 4: Occupancy Drop (LOW)
- **Concern**: Local memory usage (6KB per group) limits concurrent groups per SP to 5.
- **Mitigation**: 49 groups total >> 4 SPs, so all SPs have multiple waves. Occupancy is not a bottleneck.

### Risk 5: Incorrect Element Mapping (LOW)
- **Concern**: The stride-based assignment (`localIdx = localId, localId+32, ...`) may not correctly map to the same elements as exp_11.
- **Mitigation**: Both phases use the same `tile[0..1535]` layout matching the output `outBase + 0..1535`. Correctness test will catch any mapping error.

## Test Plan

1. **Correctness**: `python3 test_kernel.py --size quick` — verify max_diff < 0.001 vs CPU baseline
2. **Quick benchmark**: `python3 bench_kernel.py --size quick` — measure H=448 and H=224
3. **Full benchmark**: `python3 bench_kernel.py --size stride` — all sizes for comparison table
4. **If PASS and < 2.0ms**: Update BEST.md, promote to solution/kernel.cl
5. **If PASS but > 2.0ms**: Analyze profile, consider group size tuning (16 or 64)
6. **If FAIL**: Debug with `--dump-output`, check for Adreno compiler bugs
