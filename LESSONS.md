# LESSONS.md — Common GPU Optimization Knowledge (Main Branch)

> Kernel-agnostic GPU optimization experience. Branch-specific lessons go in branch LESSONS.md.

## Adreno 663 Hardware (QCS9075)

### Confirmed Constraints
- **read_imagef broken**: In-kernel read_imagef returns garbage for all formats (CL_FLOAT, CL_UNORM_INT8, CL_UNSIGNED_INT8, CL_BGRA). clEnqueueReadImage from host works correctly. Do NOT use image2d_t in OpenCL kernels on this GPU.
- **Compiler auto-vectorizes uchar loads**: Manual vload4 on uchar has zero benefit. Trust the compiler.
- **Driver auto-tunes work-group size**: Explicit WG size tuning (32/64/128/256) is flat within 1%. Pass NULL as local_work_size.
- **Scalar architecture**: Not SIMD. vload3 on float decomposes to 3 scalar ld.global.f32 instructions — may be slower than direct indexing. Test before adopting.
- **Occupancy dominates**: For memory-bound kernels, more work-items (higher WG count) beats per-WI code quality. Target ≥50 work-groups for latency hiding.

### clGetEventProfilingInfo
- May underreport H2D/D2H transfer time on Adreno. Event measures API enqueue, not DMA completion. Use total wall-clock time as primary metric.

## Effective Optimization Techniques (Universal)

### Always Try First
- **Fused kernels**: Merge operations sharing the same NDRange iteration space. Single biggest win — eliminates intermediate H2D/D2H round-trips.
- **Persistent GPU buffers**: Allocate once, reuse across frames. clCreateBuffer (~50-200us) adds up.

### When Applicable
- **Loop unrolling**: Beneficial when iteration count is small (<20) and compile-time known. Removes branch overhead.
- **Precomputed base offsets**: Pull common address calculations out of inner loops. Reduces per-iteration integer arithmetic.

## Ineffective Techniques on Adreno

- image2d_t + read_imagef (driver bug — see above)
- Manual vectorization of uchar loads (compiler does it)
- Explicit work-group size tuning (driver does it)
- vload3/vload4 on float data (scalar arch — no HW vector unit)

## Branch LESSONS Template

Branch-specific LESSONS.md should contain only:
- Effective optimizations found for THIS kernel (with experiment numbers)
- Ineffective optimizations specific to THIS kernel's workload
- Numerical traps found during optimization

Format:
```markdown
## Effective Optimizations
- [Exp 8] Loop unrolling in transpose: -13%

## Ineffective Optimizations
- [Exp 5] image2d_t: DEAD END (bug)

## Numerical Traps
- bf16 accumulator overflows at large seq_len
```
