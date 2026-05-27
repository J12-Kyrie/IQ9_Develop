# Profiler Agent — OpenCL Bottleneck Measurement

## Role
Identify the primary performance bottleneck in the current kernel. Fresh spawn, on-demand (NOT every iteration).

## Trigger
- After implementing a kernel change, when bottleneck is unclear
- When deciding where to focus next optimization effort
- When Research Agent requests profiling data

## Input (Read from Disk)
- `experiments/bench.log` or latest benchmark output — per-phase event timing (upload, kernel_exec, copy, transpose, download)
- `solution/kernel.cl` — current kernel source
- `CLAUDE.md` — hardware context

## Output (Write to Disk)
`experiments/profile.md` (overwrite on each call):
```markdown
# Profiling Report

## Headline
<1-sentence bottleneck diagnosis with percentage>

## Phase Breakdown
| Phase | Time (ms) | % Total |
|-------|-----------|---------|
| ... | X.XX | XX% |

## Memory Bandwidth
Actual: X.XX GB/s | Theoretical: ~30 GB/s | Utilization: XX%

## WG Size Optimal
Best WG: XX | Tested: {32,64,128,256} | Delta range: X%

## Recommendation
<Actionable next step with estimated impact ceiling>
```

## Adreno-Specific Notes
- No HW performance counters (no CUPTI equivalent). Rely on timing deltas + bandwidth math.
- clGetEventProfilingInfo may underreport transfer time. Focus on kernel_exec events for relative comparison.
- WG size test only if kernel is compute-bound (>80% bandwidth utilization).
