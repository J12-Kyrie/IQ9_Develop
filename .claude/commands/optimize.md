# /optimize — Autonomous OpenCL Kernel Optimization Loop

## Overview
Fully autonomous 8-step loop. ALL decisions go through Decision Agent (fresh spawn per decision point). No human interaction.

## Step 0: BOOTSTRAP (conditional)
- If `experiments/summary.md` is empty → run `/benchmark full`, log as exp_1
- Skip to Step 2 if summary.md has entries

## Step 1: ASSESS
- Read `solution/kernel.cl`, `solution/baseline.cl`
- Read `experiments/summary.md`, `experiments/LESSONS.md`, `experiments/BEST.md`
- Check SSH to device
- Check folder reservation (plan.md exists without result.md → complete that exp first)

## Step 2: PLAN — Spawn Planner Agent
```
Write experiments/decision_context.md:
  "Current state: exp_N is best. Direction candidates: [list from LESSONS.md analysis].
   Question: which single optimization to try next?"

Spawn Planner Agent (fresh context):
  Agent({subagent_type: "custom", agent_path: ".claude/agents/planner.md",
         prompt: "Read experiments/summary.md, LESSONS.md, BEST.md, profile.md,
                  solution/kernel.cl. Select the next optimization direction.
                  Write plan to experiments/exp_(N+1)/plan.md."})

Read experiments/exp_(N+1)/plan.md → execute the plan.
```

### Direction Change Gate
If the plan involves a major direction change (different optimization axis), spawn Decision Agent:
```
Write decision_context.md: "Planner suggests switching from [current axis] to [new axis].
  Ceiling: [X%]. Previous directions tried: [list]. Confirm switch?"

Spawn Decision Agent → read decision.md → if SKIP, go back to Planner for alternative.
```

## Step 3: IMPLEMENT
- Edit ONLY `solution/kernel.cl` (single file constraint)
- One optimization per iteration
- Keep kernel function signatures stable

## Step 4: VALIDATE — /benchmark quick
- 2 resolutions (min, max). ~30 seconds.
- If build fails → fix kernel.cl, re-validate (max 2 attempts)
- If correctness fails → re-validate (max 2 attempts, then ROLLBACK)

## Step 5: MEASURE — /benchmark stride 2
- 4 resolutions. ~2 minutes.
- Spawn Profiler Agent if bottleneck is unclear (NOT every iteration):
  ```
  Agent({subagent_type: "custom", agent_path: ".claude/agents/profiler.md",
         prompt: "Read bench.log, kernel.cl. Identify primary bottleneck.
                  Write to experiments/profile.md."})
  ```

## Step 6: LOG — /log-experiment
- NEVER skip (including failures)
- Snapshot kernel.cl → experiments/exp_N/
- Save bench.log, write result.md
- Update summary.md index

## Step 7: DECIDE — Spawn Decision Agent (ALWAYS)
```
Write experiments/decision_context.md:
  "Current: exp_N, total_ms=X.XXX, delta=+X.X% vs best (exp_Y, X.XXXms).
   Correctness: PASS/FAIL. Question: ACCEPT, ROLLBACK, AB_TEST, or SKIP?"

Spawn Decision Agent (fresh context):
  Agent({subagent_type: "custom", agent_path: ".claude/agents/decision.md",
         prompt: "Read decision_context.md, summary.md, LESSONS.md, BEST.md.
                  Decide: ACCEPT/ROLLBACK/AB_TEST/EXPLORE/SKIP/STOP.
                  Write decision to experiments/decision.md."})

Read experiments/decision.md → execute the decision.
```

### Decision Execution
- ACCEPT → update BEST.md, continue to next experiment
- ROLLBACK → `cp experiments/exp_{BEST}/kernel.cl solution/kernel.cl`, rebuild, verify
- AB_TEST → run paired A/B benchmark (3x alternating), re-spawn Decision Agent
- EXPLORE → branch from best kernel, try new direction for 2 quick experiments
- STOP → declare convergence, write CONVERGENCE.md, exit loop

## Step 8: BUDGET
- Default: /benchmark stride 2 (~2 min)
- /benchmark full only for: confirming new best, or every 5 iterations
- If experiment count > 50: increase default stride to 4

## Agent Dispatch Summary

| Step | Agent | Frequency | Context |
|------|-------|-----------|---------|
| Step 2 | Planner Agent | Every iteration | Fresh |
| Step 2 (direction change) | Decision Agent | On direction switch | Fresh |
| Step 5 | Profiler Agent | On-demand (bottleneck unclear) | Fresh |
| Step 7 | Decision Agent | Every iteration | Fresh |
| Plateau (5+ neutral) | Research Agent | On trigger | Fresh |
| Exploration (10 rounds) | Decision Agent | On trigger | Fresh |

## Plateau Detection
5+ consecutive experiments within ±5% → spawn Research Agent:
```
Agent({subagent_type: "custom", agent_path: ".claude/agents/research.md",
       prompt: "Read summary.md, LESSONS.md, profile.md, workload_profile.md.
                Diagnose plateau. Quantify direction ceilings. Write exp_(N+1)/plan.md."})
```

## Exploration Mode (Every 10 Rounds)
1. Write decision_context.md with available unexplored directions
2. Spawn Decision Agent → read decision.md
3. If EXPLORE: branch from best kernel, run 2 quick experiments
4. If SKIP: continue current trajectory

## Stop Condition
Decision Agent returns STOP → write CONVERGENCE.md, exit loop.
