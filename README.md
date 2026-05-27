# auto-opencl-kernel

Autonomous OpenCL GPU kernel optimization framework for Qualcomm Adreno embedded GPUs. Uses Claude Code multi-agent orchestration to iteratively optimize kernel performance with zero human intervention.

**Platform**: Qualcomm QCS9075 / Adreno 663 | **Language**: OpenCL C 3.0 + C++17 | **Device**: `ubuntu@192.168.100.102`

## Quick Start

### 1. Create a Kernel Branch

```bash
git checkout main
git checkout -b my-kernel
```

Each kernel optimization campaign lives in its own branch. The `template/` directory provides the skeleton.

### 2. Define kernel.toml (Required)

Before any optimization, you must define `kernel.toml` in the branch root. This is the single source of truth for the benchmark harness — no kernel-specific code goes in `benchmark.cpp`.

```toml
[kernel]
name = "my_kernel"
description = "What this kernel does"

[[kernels]]
function = "my_kernel"          # Must match __kernel void name in kernel.cl
ndim = 1                        # 1 or 2 (NDRange dimensions)
global_size = "N"               # Expression: "N" or "[W, H]"
local_work_size = 0             # 0 = NULL (let driver auto-tune)

[[kernels.args]]
name = "input"
type = "buffer_read"            # buffer_read | buffer_write | buffer_rw | int | float
size = "N * 4"                  # Buffer size in bytes (expression with problem_size vars)

[[kernels.args]]
name = "output"
type = "buffer_write"
size = "N * 4"

[[kernels.args]]
name = "N"
type = "int"
source = "problem_size"         # Auto-filled from problem_sizes

[problem_sizes]
quick  = [{N = 1024}, {N = 65536}]                              # 2 sizes, ~30s
stride = [{N = 1024}, {N = 16384}, {N = 65536}, {N = 1048576}]  # 4 sizes, ~2min
full   = [{N = 1024}, {N = 4096}, {N = 16384}, {N = 65536},     # 8 sizes, ~8min
          {N = 262144}, {N = 1048576}, {N = 4194304}, {N = 16777216}]

[correctness]
enabled = true
mode = "cpu_baseline"           # cpu_baseline | gold_file | none
tolerance = 0.001               # Max absolute difference for PASS
```

### 3. Write Your Kernel

Edit `solution/kernel.cl` — this is the **only file the optimizer modifies** during optimization.

```c
__kernel void my_kernel(
    __global const float* input,
    __global float* output,
    const int N)
{
    int idx = get_global_id(0);
    if (idx >= N) return;
    output[idx] = input[idx] * 2.0f;
}
```

### 4. Write CPU Baseline (Optional)

If `correctness.mode = "cpu_baseline"`, provide `solution/cpu_baseline.cpp`:

```cpp
#include <vector>
void cpu_baseline(const std::vector<float>& input, std::vector<float>& output, int N) {
    output.resize(N);
    for (int i = 0; i < N; i++) {
        output[i] = input[i] * 2.0f;
    }
}
```

The benchmark harness links this automatically if the file exists.

### 5. Build and Run

```bash
# Build on device
ssh ubuntu@192.168.100.102
cd /mnt/workspace/auto-opencl-kernel
mkdir -p build && cd build
cmake ../template/host && make -j$(nproc)

# Run benchmark
./benchmark --quick                    # Quick: 2 problem sizes
./benchmark --stride 2                 # Stride: 4 problem sizes
./benchmark --full                     # Full: 8 problem sizes
./benchmark --output /tmp/bench.log    # Save results
```

### 6. Run the Optimization Loop

```
/optimize
```

This launches the fully autonomous optimization loop. See [Optimization Loop](#optimization-loop) for details.

## Architecture

### Multi-Agent System

```
Optimizer (main loop)
  ├── Planner Agent      — selects next optimization direction
  ├── Decision Agent     — ACCEPT / ROLLBACK / AB_TEST / EXPLORE / STOP
  ├── Profiler Agent     — identifies bottleneck (on-demand)
  └── Research Agent     — diagnoses plateaus (5+ neutral experiments)
```

Each agent spawns with a **fresh context window** and communicates via disk artifacts in `experiments/`. No agent has memory of prior conversations — decisions are based solely on files on disk.

### Optimization Loop (`/optimize`)

```
Step 0: Bootstrap (first run: baseline benchmark → exp_1)
Step 1: Assess   (read state, check SSH, check in-flight experiments)
Step 2: Plan      → Planner Agent picks direction
Step 3: Implement (edit solution/kernel.cl only)
Step 4: Validate  → /benchmark quick (2 sizes, ~30s)
Step 5: Measure   → /benchmark stride 2 (4 sizes, ~2min)
Step 6: Log       → /log-experiment (ALWAYS, including failures)
Step 7: Decide    → Decision Agent (ALWAYS)
Step 8: Budget    (full benchmark every 5 iterations)
```

### Benchmark Strategy

| Level | Flag | Problem Sizes | Time | Purpose |
|-------|------|---------------|------|---------|
| quick | `--quick` | 2 | ~30s | Compile + correctness check |
| stride N | `--stride N` | 4 (N=2) | ~2min | Default per-iteration |
| full | `--full` | all 8 | ~8min | Confirm new best |

Problem sizes are defined per-branch in `kernel.toml` — the harness is kernel-agnostic.

## Repository Layout

```
auto-opencl-kernel/
├── CLAUDE.md                   # Framework constitution (non-negotiable rules)
├── LESSONS.md                  # Common GPU optimization knowledge
├── kernel.toml                 # Per-branch kernel configuration (REQUIRED)
├── config.toml.example         # Device configuration template
│
├── .claude/
│   ├── commands/
│   │   ├── optimize.md         # /optimize loop with agent dispatch
│   │   ├── benchmark.md        # /benchmark 3-level strategy
│   │   └── log-experiment.md   # Experiment logging
│   └── agents/
│       ├── planner.md          # Planner Agent definition
│       ├── profiler.md         # Profiler Agent definition
│       ├── research.md         # Research Agent definition
│       └── decision.md         # Decision Agent definition
│
├── template/                   # Kernel project skeleton
│   ├── CLAUDE.md               # Branch-level config
│   ├── LESSONS.md              # Branch-level experience log
│   ├── kernel.toml             # Example kernel config (hello_kernel)
│   ├── solution/
│   │   ├── kernel.cl           # Agent-editable (SINGLE FILE)
│   │   ├── baseline.cl         # CPU reference docs (read-only)
│   │   └── cpu_baseline.cpp    # Per-branch CPU reference (optional)
│   ├── host/
│   │   ├── benchmark.cpp       # Generic benchmark runner (reads kernel.toml)
│   │   ├── CMakeLists.txt      # Build config
│   │   └── opencl_accelerator.h # OpenCL boilerplate
│   └── experiments/
│       ├── BEST.md             # Golden reference for rollback
│       └── summary.md          # Experiment index
│
└── template/experiments/       # Experiment artifacts (per-iteration)
    ├── exp_1/                  # Each experiment folder
    │   ├── plan.md             # What to try
    │   ├── kernel.cl snapshot  # Kernel at this point
    │   ├── bench.log           # Raw benchmark output
    │   └── result.md           # Outcome + lessons
    ├── decision.md             # Latest decision (from Decision Agent)
    ├── decision_context.md     # Latest context for Decision Agent
    └── profile.md              # Latest profiling report
```

## Adreno 663 Constraints

Hard-won knowledge from optimization campaigns on this GPU:

| Constraint | Impact |
|------------|--------|
| `read_imagef` is broken | Never use `image2d_t` in kernels — returns garbage |
| Compiler auto-vectorizes uchar loads | Manual `vload4` has zero benefit |
| Driver auto-tunes work-group size | Pass `NULL` as `local_work_size` |
| Scalar architecture (not SIMD) | `vload3` on float = 3 scalar loads, no HW vector unit |
| Occupancy dominates | More work-items beats per-WI code quality for memory-bound kernels |
| `clGetEventProfilingInfo` underreports H2D/D2H | Use wall-clock as primary metric for transfers |

## Kernel Configuration Reference

### kernel.toml Schema

| Section | Field | Type | Description |
|---------|-------|------|-------------|
| `[kernel]` | `name` | string | Kernel display name |
| `[kernel]` | `description` | string | One-line description |
| `[[kernels]]` | `function` | string | OpenCL kernel function name |
| `[[kernels]]` | `ndim` | int | NDRange dimensions (1 or 2) |
| `[[kernels]]` | `global_size` | string | Expression: `"N"`, `"[W, H]"` |
| `[[kernels]]` | `local_work_size` | int | 0 = NULL (driver decides) |
| `[[kernels.args]]` | `name` | string | Argument name (for logging) |
| `[[kernels.args]]` | `type` | string | `buffer_read`, `buffer_write`, `buffer_rw`, `int`, `float` |
| `[[kernels.args]]` | `size` | string | Buffer size expression (bytes). Variables from problem_sizes |
| `[[kernels.args]]` | `source` | string | `"problem_size"` or variable name |
| `[[kernels.args]]` | `value` | number | Fixed literal value |
| `[problem_sizes]` | `quick` | array | 2 problem sizes for quick benchmark |
| `[problem_sizes]` | `stride` | array | 4 problem sizes for stride benchmark |
| `[problem_sizes]` | `full` | array | 8 problem sizes for full benchmark |
| `[correctness]` | `enabled` | bool | Enable correctness checking |
| `[correctness]` | `mode` | string | `cpu_baseline`, `gold_file`, `none` |
| `[correctness]` | `tolerance` | float | Max absolute difference for PASS |

### Multi-Kernel Pipelines

For kernels with multiple stages (e.g., resize + transpose), define multiple `[[kernels]]` sections in order:

```toml
[[kernels]]
function = "resize_bilinear_normalized"
ndim = 2
global_size = "[dst_w, dst_h]"

[[kernels.args]]
name = "src"
type = "buffer_read"
size = "src_h * src_stride"
# ... more args ...

[[kernels]]
function = "transpose_to_patch"
ndim = 1
global_size = "seq_len * 2"
local_work_size = 64

[[kernels.args]]
name = "norm_buf"
type = "buffer_read"
size = "dst_w * dst_h * 3 * 4"
# ... more args ...
```

Kernels execute in declaration order. Events chain automatically (each kernel waits for the previous).

### Expression Language

Size expressions support:
- **Variables**: from `[problem_sizes]` dicts (e.g., `N`, `src_w`, `src_h`)
- **Literals**: integers and floats
- **Multiplication**: `N * 4`, `src_h * src_stride`, `W * H * 3 * 4`
- **Problem size binding**: `source = "problem_size"` auto-fills the first variable from the problem size dict

## Non-Negotiable Rules

These rules govern the optimization loop and cannot be overridden:

1. **Stay in OpenCL** — Don't switch to GLSL, Vulkan Compute, or CPU NEON
2. **Absolute latencies only** — Report microseconds/milliseconds, never speedup ratios
3. **One optimization per iteration** — Single variable change per experiment
4. **No GPU locally** — All build/benchmark via SSH on target device
5. **Log every experiment** — Including failures. Never skip `/log-experiment`
6. **Never stop the loop** — Only Decision Agent can issue STOP
7. **No benchmark gaming** — No pre-warmed queues, no hiding transfer time
8. **Full autonomous mode** — All decisions go through Decision Agent

## Knowledge Management

### Two-Tier LESSONS System

| Tier | Location | Scope |
|------|----------|-------|
| Common | `LESSONS.md` (main) | GPU optimization knowledge across all kernels |
| Branch | `LESSONS.md` (branch) | Kernel-specific experiments and findings |

When a branch completes, only common LESSONS entries merge back to main.

### Experiment Tracking

Every experiment is logged with:
- `plan.md` — What to try and why
- `kernel.cl` — Snapshot of the kernel at this point
- `bench.log` — Raw benchmark output
- `result.md` — Outcome, metrics, and lessons learned

The Decision Agent uses these artifacts to make informed accept/rollback/explore decisions.

## Commands

| Command | Description |
|---------|-------------|
| `/optimize` | Run the full autonomous optimization loop |
| `/benchmark --quick` | Quick benchmark (2 sizes, ~30s) |
| `/benchmark --stride 2` | Stride benchmark (4 sizes, ~2min) |
| `/benchmark --full` | Full benchmark (8 sizes, ~8min) |
| `/log-experiment` | Log current experiment to `experiments/` |

## Dependencies

- **OpenCL** — `libOpenCL-dev` on device
- **CMake** >= 3.14
- **C++17** compiler (GCC/Clang)
- **SSH access** to target device (build + benchmark)
- **Claude Code** with oh-my-claudecode for agent orchestration

## License

MIT
