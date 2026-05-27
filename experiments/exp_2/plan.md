# Experiment 2 — Nested Loop with Compile-Time Constants (Division Elimination)

**Date**: 2026-05-27
**Status**: PLANNED
**Baseline**: exp_1 = 4.695ms at H=448
**Target**: Sub-1ms at H=448

## Hypothesis

The inner loop iterates 1536 times with 7 integer div/mod operations per iteration.
Since 1536 = 3 * 2 * 16 * 16, we can replace the flat loop + coordinate decomposition
with a 4-level nested loop that directly iterates over (cIdx, tPatchIdx, patchH, patchW).

This eliminates ALL 8.4M integer divisions per dispatch.

Additionally, making the inner loop bounds compile-time constants (via #define) allows
the compiler to fully unroll the loops, converting the 1536-iteration loop into a
straight-line code block with zero branch overhead.

## Optimization Details

### Current code (exp_1):
```
for (int elemIdx = 0; elemIdx < input_dim; elemIdx++) {
    int cIdx = elemIdx / tps_pp;           // div
    int temp = elemIdx % tps_pp;           // mod
    int tPatchIdx = temp / patch_area;     // div
    int patchH = (temp % patch_area) / patch_size;  // mod + div
    int patchW = temp % patch_size;        // mod
    int srcIdx = srcT * HW * C + srcH * WC + srcW * C + cIdx;
    output[seqIdx * input_dim + elemIdx] = input[srcIdx];
}
```
7 div/mod per iteration * 1536 = 10,752 div/mod per work-item

### Proposed code (exp_2):
```
int dstBase = seqIdx * INPUT_DIM;
int srcBaseT = baseT * HW * C;
int srcBaseH = baseH * WC;
int srcBaseW = baseW * C;

int elemIdx = 0;
for (int t = 0; t < TPS; t++) {          // TPS = 2
    int srcT = srcBaseT + t * HW * C;
    for (int ph = 0; ph < PS; ph++) {     // PS = 16
        int srcH = srcBaseH + ph * WC;
        for (int pw = 0; pw < PS; pw++) { // PS = 16
            int srcW = srcBaseW + pw * C;
            int patchBase = srcT + srcH + srcW;
            for (int c = 0; c < C; c++) { // C = 3
                output[dstBase + elemIdx] = input[patchBase + c];
                elemIdx++;
            }
        }
    }
}
```
0 div/mod per iteration. Only additions and compile-time-constant multiplications.

### Why this works:
1. **Division elimination**: The nested loop structure maps directly to the same
   (cIdx, tPatchIdx, patchH, patchW) decomposition without any division.
2. **Loop unrolling**: With PS=16 and C=3 as compile-time #defines, the compiler
   can fully unroll all four loops. The inner three loops (16*16*3=768) are prime
   candidates for unrolling. The outermost loop (TPS=2) trivially unrolls.
3. **Strength reduction**: Phrases like `t * HW * C` become `0, HW*C, 2*HW*C` at
   compile time — just constants, no runtime multiplication.
4. **Reduced register pressure**: No need to keep tps_pp, patch_area in registers.
5. **Better ILP**: Straight-line code after unrolling exposes more instruction-level
   parallelism to the Adreno scheduler.

## Changes Required

### 1. kernel.cl
- Add compile-time constants: `#define PS 16`, `#define TPS 2`, `#define C_CH 3`
- Replace flat loop with 4-level nested loop
- Remove unused variables (tps_pp, patch_area)
- Keep `input_dim` kernel argument for dst indexing (or replace with `PS*PS*TPS*C_CH`)

### 2. Build flags (if needed)
- Verify compiler receives -cl-fast-relaxed-math (should already be set by framework)
- No additional flags needed — #defines handle compile-time constants

## Expected Impact

| Factor | exp_1 | exp_2 |
|--------|-------|-------|
| Divisions per WI | 10,752 | 0 |
| Branch instructions | 1536 loop iterations | ~0 (fully unrolled) |
| Register pressure | Higher (div temps) | Lower (direct indexing) |
| ILP | Sequential div chain | Parallel address calc |

Conservative estimate: 3-5x speedup (division is ~20 cycles on Adreno scalar unit).
Optimistic: 5-8x if unrolling exposes enough ILP to saturate the load/store unit.

## Success Criteria

1. **Correctness**: PASS all 8 problem sizes (tolerance 0.001)
2. **Performance**: H=448 kernel time < 2.0ms (minimum), < 1.0ms (stretch)
3. **No regression**: No problem size should be slower than exp_1 baseline

## Risks

- **Code size explosion**: Fully unrolled 1536 iterations may exceed instruction cache.
  Mitigation: Use `__attribute__((opencl_unroll_hint(4)))` partial unroll if full
  unroll causes I-cache thrashing.
- **Register spilling**: 768 address computations may exceed register file.
  Mitigation: The compiler should reuse registers across unrolled iterations since
  each iteration is independent.
- **Correctness**: Nested loop must produce identical elemIdx ordering as flat loop.
  Verification: CPU baseline comparison will catch any ordering mismatch.
