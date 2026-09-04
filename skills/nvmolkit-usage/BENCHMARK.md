# Skill Benchmark: nvmolkit-usage

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `nvmolkit-usage`
- Evaluation date: 2026-09-04
- Evaluator version: `1.5.4`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 12 evaluation tasks (12 positive)
- Dataset digest: `sha256:16ead845d201c386f3f059697aff38c0e6978ce5e90370c6548d7bade427c6fb` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
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
| Overall | 92.9% — baseline ran, but no comparable score was available; uplift unavailable | 93.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 79.2% → 100.0% (+20.8 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 100.0% → 96.7% (-3.3 points) | 90.0% → 100.0% (+10.0 points) |
| Discoverability | 98.3% — baseline ran, but no comparable score was available; uplift unavailable | 94.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 90.9% → 86.5% (-4.4 points) | 74.8% → 91.6% (+16.8 points) |
| Efficiency | 82.9% — baseline ran, but no comparable score was available; uplift unavailable | 79.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,424,872 | 8,158,676 | -3,733,804 | -45.76% | skill 12/12; base 12/12 |
| claude-code | nvmolkit-usage-001 | 156,923 | 680,876 | -523,953 | -76.95% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-002 | 1,754,150 | 606,737 | +1,147,413 | +189.11% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-003 | 116,736 | 1,024,480 | -907,744 | -88.61% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-004 | 71,735 | 577,415 | -505,680 | -87.58% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-005 | 636,342 | 650,870 | -14,528 | -2.23% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-006 | 73,458 | 485,443 | -411,985 | -84.87% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-007 | 161,930 | 1,434,838 | -1,272,908 | -88.71% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-008 | 642,554 | 495,284 | +147,270 | +29.73% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-009 | 70,609 | 500,318 | -429,709 | -85.89% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-010 | 271,450 | 1,020,801 | -749,351 | -73.41% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-011 | 70,100 | 31,452 | +38,648 | +122.88% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-012 | 398,885 | 650,162 | -251,277 | -38.65% | skill 1/1; base 1/1 |
| codex | All cases | 1,013,634 | 806,737 | +206,897 | +25.65% | skill 12/12; base 12/12 |
| codex | nvmolkit-usage-001 | 42,585 | 39,492 | +3,093 | +7.83% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-002 | 47,194 | 57,641 | -10,447 | -18.12% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-003 | 139,631 | 40,127 | +99,504 | +247.97% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-004 | 73,585 | 35,989 | +37,596 | +104.47% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-005 | 141,950 | 47,800 | +94,150 | +196.97% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-006 | 95,994 | 18,730 | +77,264 | +412.51% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-007 | 69,777 | 31,800 | +37,977 | +119.42% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-008 | 133,410 | 67,161 | +66,249 | +98.64% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-009 | 31,189 | 28,514 | +2,675 | +9.38% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-010 | 99,068 | 32,126 | +66,942 | +208.37% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-011 | 31,014 | 32,486 | -1,472 | -4.53% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-012 | 108,237 | 374,871 | -266,634 | -71.13% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,438,506 | 8,965,413 | -3,526,907 | -39.34% | skill 24/24; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 1 validator(s); 4 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'risk_tier' is not kebab-case (`skills/nvmolkit-usage/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nvmolkit-usage/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvmolkit-usage/SKILL.md`)
- **MEDIUM** SCHEMA/author_missing: Author not specified in metadata (`skills/nvmolkit-usage/SKILL.md`)

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
