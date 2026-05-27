# CLAUDE.md — auto-opencl-kernel Framework (Main Branch)

## Framework Identity
Production-grade autonomous OpenCL GPU kernel optimization framework for Qualcomm Adreno embedded GPUs. Kernel-agnostic — each kernel optimization campaign lives in its own branch.

- **Platform**: Qualcomm Adreno (tested: QCS9075 Adreno 663)
- **Language**: OpenCL C 3.0 with C++17 host
- **Repository**: https://github.com/J12-Kyrie/auto-opencl-kernel
- **Device**: ubuntu@192.168.100.102:/mnt/workspace/auto-opencl-kernel/

## Non-Negotiable Rules

### 1. Stay in OpenCL
Don't switch to GLSL, Vulkan Compute, or CPU NEON. Fix the kernel, don't flee the language.

### 2. Absolute Latencies Only
Report microseconds (us) and milliseconds (ms) from clGetEventProfilingInfo. Never speedup ratios.

### 3. One Optimization Per Iteration
Single variable change. A/B paired test for <5% differences.

### 4. No GPU Locally
All build/benchmark via SSH on target device. No local emulation.

### 5. Log Every Experiment
Including failures. Never skip /log-experiment. Never overwrite an existing result.md.

### 6. Never Stop the Loop
Only Decision Agent can issue STOP. max_iterations is a soft threshold.

### 7. No Benchmark Gaming
No pre-warmed queues, no buffer reuse tricks, no hiding transfer time.

### 8. anysearch Allowed for Research
Research Agent can use anysearch for OpenCL/Adreno documentation.

### 9. Full Autonomous Mode
ALL decisions go through Decision Agent (fresh spawn each time).
No user prompts. No interactive pauses. Optimizer executes, Agents decide.

## Kernel Configuration QA (Pre-requisite)

**Before running `/optimize` or any optimization loop, the kernel configuration MUST be fully defined through QA interaction with the user.**

This is a gate — the optimizer must NOT start until all items below are confirmed:

| Item | Example |
|------|---------|
| Kernel function name(s) | `resize_bilinear_normalized`, `transpose_to_patch` |
| Input/output buffer types and sizes | `uchar* src [h*stride]`, `float* dst [448*448*3]` |
| Scalar parameters and their semantics | `src_w`, `src_h`, `mean_r`, `grid_h` |
| NDRange dimensions | 2D `[dst_w, dst_h]` or 1D `[N]` |
| Problem sizes for quick/stride/full | `[{320,240}, {1920,1080}]` |
| Correctness check strategy | CPU baseline / gold-file / tolerance-only |
| Numerical traps | Accumulator precision, saturation bounds |
| Performance target or bottleneck hypothesis | Memory-bound? Compute-bound? Target latency? |

**QA Flow:**
1. User provides kernel.cl (or describes the operation)
2. Optimizer asks clarifying questions for any ambiguous items above
3. User confirms -> Optimizer writes `kernel.toml`
4. Only then: branch setup + `/optimize`

**Never assume** kernel semantics, buffer shapes, or correctness criteria. If the kernel.cl signature is ambiguous, ask. If the user says "optimize this kernel" without config details, start the QA — do not guess.

## Agent Architecture

```
Main Optimizer  ──spawn──▶ Planner Agent     (Step 2: picks optimization direction)
                ──spawn──▶ Profiler Agent     (on-demand: measures bottleneck)
                ──spawn──▶ Research Agent     (plateau trigger: diagnoses strategy)
                ──spawn──▶ Decision Agent     (every decision point: ACCEPT/ROLLBACK/EXPLORE/STOP)
```

All agents communicate via disk artifacts in `experiments/`. Each spawn creates a clean context window. See `.claude/commands/optimize.md` for dispatch rules.

## LESSONS.md — Two-Tier System

| Tier | Location | Scope |
|------|----------|-------|
| Common | `Main/LESSONS.md` | GPU optimization knowledge applicable across all kernels |
| Branch | `branch/LESSONS.md` | Kernel-specific experiments and findings |

When a branch completes, PR only merges common LESSONS entries back to Main.

## Branch Model

```
main                           ← Framework core (this branch)
├── template/                  ← Kernel project skeleton
│
├── branch: image-preprocess   ← Full project (inherits framework)
│   ├── solution/kernel.cl     ← resize+normalize+transpose
│   └── experiments/           ← 10 experiments
│
└── branch: <next-kernel>      ← Each kernel = one branch
```

Create new kernel branch: `bash scripts/create_branch.sh <kernel-name>`

## Repository Layout (Main Branch)

```
auto-opencl-kernel/
├── CLAUDE.md                   # ← THIS FILE (framework constitution)
├── LESSONS.md                  # Common GPU optimization knowledge
├── config.toml.example         # Device configuration template
├── .gitignore
├── .claude/
│   ├── settings.json
│   ├── commands/
│   │   ├── optimize.md         # /optimize loop with Agent dispatch
│   │   ├── benchmark.md        # /benchmark 3-level strategy
│   │   └── log-experiment.md   # Experiment logging
│   └── agents/
│       ├── planner.md          # Planner Agent definition
│       ├── profiler.md         # Profiler Agent definition
│       ├── research.md         # Research Agent definition
│       └── decision.md         # Decision Agent definition (★)
├── scripts/
│   ├── run_benchmark.sh        # SSH build + run + collect
│   ├── create_branch.sh        # Template → new kernel branch
│   └── pack_solution.py        # Archive for submission
└── template/                   # Kernel project skeleton
    ├── CLAUDE.md               # Branch-level config
    ├── LESSONS.md              # Branch-level experience log
    ├── solution/
    │   ├── kernel.cl           # Agent-editable (SINGLE FILE)
    │   └── baseline.cl         # CPU reference (read-only)
    ├── host/
    │   ├── benchmark.cpp       # Generic benchmark runner
    │   ├── CMakeLists.txt      # Build config
    │   └── opencl_accelerator.h # OpenCL boilerplate
    └── experiments/
        ├── BEST.md             # Golden reference for rollback
        ├── summary.md          # Experiment index
        └── .gitkeep
