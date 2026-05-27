# Experiment 7 — Reordered Loops for Input Coalescing

**Date**: 2026-05-27
**Status**: ACCEPT

## Hypothesis
Reorder loop from (c, t, ph, pw) to (t, ph, pw, c) so innermost c reads consecutive input addresses.
Reduces input read stride from C=3 floats to 1 float.

## Results

| Problem Size | exp_1 (ms) | exp_3 (ms) | exp_7 (ms) | vs baseline |
|-------------|-----------|-----------|-----------|-------------|
| H=128 | 1.282 | 0.579 | - | - |
| H=224 | 1.851 | 0.996 | 0.979 | -47.1% |
| H=448 | 4.695 | 4.210 | 4.068 | -13.4% |

**Production target (H=448)**: 4.068ms (was 4.695ms, -13.4%)
**Correctness**: PASS
