# CLAUDE.md — Branch Configuration

## Kernel Identity
- **Name**: [REPLACE_WITH_KERNEL_NAME]
- **Description**: [REPLACE_WITH_KERNEL_DESCRIPTION]
- **Branch**: [REPLACE_WITH_BRANCH_NAME]

## Kernel Configuration

This branch MUST have a `kernel.toml` in the project root. It is the single source of truth for:
- Kernel function name(s) and argument signatures
- Buffer sizes and scalar parameters
- NDRange dimensions
- Problem sizes for quick/stride/full benchmark levels
- Correctness verification mode and tolerance

**Do NOT hardcode kernel-specific constants in benchmark.cpp.** All domain knowledge lives in `kernel.toml`.

### kernel.toml Structure

| Section | Purpose |
|---------|---------|
| `[kernel]` | Name and description |
| `[[kernels]]` | Function name, ndim, global/local work size |
| `[[kernels.args]]` | Positional args: type (buffer_read/write/rw, int, float), size expression, source |
| `[problem_sizes]` | quick/stride/full arrays of variable dicts |
| `[correctness]` | enabled, mode (cpu_baseline/gold_file/none), tolerance |

### Per-branch CPU Baseline

If `correctness.mode = "cpu_baseline"`, provide `solution/cpu_baseline.cpp` with:
```cpp
void cpu_baseline(const std::vector<float>& input, std::vector<float>& output, int N);
```
The benchmark harness links it automatically if the file exists.

## Kernel I/O Specification
[REPLACE: Document input/output types, shapes, dimensionality here AND in kernel.toml]

## Numerical Traps
[REPLACE: Precision requirements, accumulator types, known pitfalls]

## Inherited Rules
All non-negotiable rules from Main/CLAUDE.md apply.
Agent architecture and dispatch rules from Main/.claude/ apply.
Benchmark infrastructure from Main/template/host/ reads kernel.toml for configuration.

## Branch LESSONS.md
This branch maintains its own LESSONS.md for kernel-specific experience.
PR only common GPU lessons back to Main/LESSONS.md after convergence.
