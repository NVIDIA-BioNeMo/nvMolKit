# Skill Benchmark: nvmolkit-usage

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvmolkit-usage`
- Evaluation date: 2026-09-18
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 12 evaluation tasks (12 positive)
- Dataset digest: `sha256:ce8098e0dd2fc0698933b7d4d303fe13bdd1d9be7e87d142bdfc44baae7a9f90` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.6% — baseline ran, but no comparable score was available; uplift unavailable | 93.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 70.8% → 95.8% (+25.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 96.7% → 100.0% (+3.3 points) | 93.3% → 100.0% (+6.7 points) |
| Discoverability | 94.2% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 90.3% → 90.8% (+0.5 points) | 84.3% → 86.5% (+2.2 points) |
| Efficiency | 82.2% — baseline ran, but no comparable score was available; uplift unavailable | 85.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 5,388,789 | 6,485,536 | -1,096,747 | -16.91% | skill 12/12; base 12/12 |
| claude-code | nvmolkit-usage-001 | 206,219 | 690,603 | -484,384 | -70.14% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-002 | 1,257,795 | 177,892 | +1,079,903 | +607.06% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-003 | 285,979 | 297,646 | -11,667 | -3.92% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-004 | 245,297 | 588,508 | -343,211 | -58.32% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-005 | 648,715 | 740,940 | -92,225 | -12.45% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-006 | 74,549 | 561,517 | -486,968 | -86.72% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-007 | 302,671 | 550,115 | -247,444 | -44.98% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-008 | 1,160,762 | 708,853 | +451,909 | +63.75% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-009 | 69,766 | 416,700 | -346,934 | -83.26% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-010 | 389,838 | 1,152,170 | -762,332 | -66.16% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-011 | 69,928 | 61,294 | +8,634 | +14.09% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-012 | 677,270 | 539,298 | +137,972 | +25.58% | skill 1/1; base 1/1 |
| codex | All cases | 1,063,506 | 653,365 | +410,141 | +62.77% | skill 12/12; base 12/12 |
| codex | nvmolkit-usage-001 | 42,626 | 30,919 | +11,707 | +37.86% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-002 | 42,737 | 47,031 | -4,294 | -9.13% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-003 | 119,838 | 142,275 | -22,437 | -15.77% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-004 | 73,963 | 104,844 | -30,881 | -29.45% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-005 | 136,297 | 38,676 | +97,621 | +252.41% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-006 | 50,646 | 33,189 | +17,457 | +52.60% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-007 | 75,109 | 35,311 | +39,798 | +112.71% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-008 | 130,297 | 59,105 | +71,192 | +120.45% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-009 | 85,560 | 18,457 | +67,103 | +363.56% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-010 | 147,972 | 27,105 | +120,867 | +445.92% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-011 | 51,430 | 20,374 | +31,056 | +152.43% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-012 | 107,031 | 96,079 | +10,952 | +11.40% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 6,452,295 | 7,138,901 | -686,606 | -9.62% | skill 24/24; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nvmolkit-usage/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Large skill (5312 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/nvmolkit-usage/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (567 chars, recommend 50-150) (`skills/nvmolkit-usage/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/nvmolkit-usage/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/nvmolkit-usage/SKILL.md`)
- 5 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
