# /benchmark — OpenCL Kernel Benchmark Command

## 3-Level Strategy

| Level | Flag | Resolutions | Time | Purpose |
|-------|------|-------------|------|---------|
| quick | --quick | 2 | ~30s | Compile + correctness |
| stride N | --stride N | 4 (N=2) | ~2min | Default iteration |
| full | --full | 8 | ~8min | Confirm new best |

## Resolution Test Set
{320x240, 640x480, 800x600, 1024x768, 1280x720, 1600x900, 1920x1080, 3840x2160}

## Execution Flow
```bash
# 1. Build on device
sshpass -p "$SSH_PASS" ssh ubuntu@<device> \
  "cd <workspace>/build && cmake ../host && make -j\$(nproc)"

# 2. Run benchmark
sshpass -p "$SSH_PASS" ssh ubuntu@<device> \
  "<workspace>/build/benchmark <flag> --output /tmp/bench.log"

# 3. Fetch results
sshpass -p "$SSH_PASS" scp ubuntu@<device>:/tmp/bench.log experiments/exp_N/bench.log
```

## Output Format (bench.log)
```
# OpenCL Kernel Benchmark
Resolution   Upload_ms  Kernel_ms  Copy_ms  Transp_ms  Down_ms  Total_ms  Stdev_ms
320x240          x.xxx     x.xxx    x.xxx     x.xxx    x.xxx    x.xxx    x.xxx
# Correctness: max_diff=X.XXXXXX PASS|FAIL
# GPU Temp Before: XX.XC / After: XX.XC (delta: +X.XC)
# Overall: PASS|FAIL
```

## A/B Paired Test
When |delta| <= 5% needs confirmation:
1. Run benchmark with current kernel → bench_A.log
2. Restore best kernel: `cp experiments/exp_{BEST}/kernel.cl solution/kernel.cl`
3. Run benchmark with best kernel → bench_B.log
4. Run 3 alternating pairs (A,B,A,B,A,B)
5. Compare per-resolution mean deltas
