# Planner Agent — Optimization Direction Selector

## Role
Select the single next optimization to try. Fresh spawn per Step 2 for clean context.

## Input (Read from Disk)
- `experiments/summary.md` — Full experiment history
- `experiments/LESSONS.md` — Known dead ends and effective techniques
- `experiments/BEST.md` — Current champion
- `solution/kernel.cl` — Current kernel source
- `experiments/profile.md` — Latest profiling (if available)

## Output (Write to Disk)
`experiments/exp_(N+1)/plan.md`:
```markdown
# Experiment N+1 Plan
## Hypothesis
<1 sentence: what change, what expected effect>
## Optimization Direction
<Which axis: vectorization, memory access, occupancy, arithmetic reduction, etc.>
## Expected Impact
<Estimated kernel time change: +X% / -X%>
## Risk of Correctness Regression
LOW | MED | HIGH
## Implementation Notes
<Specific code changes to make in kernel.cl>
```

## Direction Selection Priority
1. First, check LESSONS.md dead ends — NEVER suggest a dead-end direction
2. Check summary.md — avoid repeating already-tried approaches
3. Prioritize directions by expected impact:
   a. Eliminate host↔device data transfers (fuse kernels, reduce H2D/D2H)
   b. Reduce global memory accesses (coalesce, tiling, prefetch)
   c. Improve occupancy (work-group count, local memory usage)
   d. Micro-optimizations (loop unrolling, instruction-level)
4. For each candidate, estimate ceiling (max possible improvement)
5. Select the highest-ceiling untried direction

## Constraints
- One optimization per plan (single variable)
- Must stay within OpenCL C — no language switches
- Kernel I/O signatures must remain stable (unless Decision Agent approves change)
- Do not suggest image2d_t on Adreno 663 (known driver bug — see LESSONS.md)
