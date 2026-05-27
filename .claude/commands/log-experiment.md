# /log-experiment — Experiment Logging Automation

## Flow

### 1. Pick Experiment Folder
- Find highest N in experiments/exp_N/
- If exp_N/plan.md exists but result.md absent → USE exp_N (reserved)
- Otherwise → CREATE exp_(N+1)
- Never overwrite existing result.md

### 2. Write Artifacts
```
cp solution/kernel.cl experiments/exp_N/kernel.cl
cp /tmp/bench.log experiments/exp_N/bench.log
```

### 3. Write result.md
```markdown
# Experiment N: <description>
Date: YYYY-MM-DD
Status: PASS | FAIL
Hypothesis: <from plan.md>

## Results
| Resolution | Kernel_ms | Total_ms |
## Lessons
- <What was learned>
```

### 4. Update summary.md
Append: `| Exp N | Date | Description | Kernel_ms | Total_ms | Pass | Notes |`

### 5. Update LESSONS.md (Branch)
If cross-experiment insight: append. If disproven: DELETE the old entry.
