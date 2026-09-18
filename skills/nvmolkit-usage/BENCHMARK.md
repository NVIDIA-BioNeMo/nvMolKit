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
| Overall | 94.9% — baseline ran, but no comparable score was available; uplift unavailable | 93.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 100.0% (+25.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 96.7% → 100.0% (+3.3 points) |
| Discoverability | 97.7% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 85.8% → 91.8% (+6.0 points) | 80.5% → 85.1% (+4.6 points) |
| Efficiency | 85.1% — baseline ran, but no comparable score was available; uplift unavailable | 84.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,953,168 | 7,333,864 | -2,380,696 | -32.46% | skill 12/12; base 12/12 |
| claude-code | nvmolkit-usage-001 | 156,476 | 585,885 | -429,409 | -73.29% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-002 | 2,344,638 | 477,914 | +1,866,724 | +390.60% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-003 | 250,932 | 704,638 | -453,706 | -64.39% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-004 | 73,034 | 291,291 | -218,257 | -74.93% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-005 | 279,108 | 848,364 | -569,256 | -67.10% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-006 | 73,663 | 993,072 | -919,409 | -92.58% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-007 | 296,983 | 647,857 | -350,874 | -54.16% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-008 | 693,717 | 533,167 | +160,550 | +30.11% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-009 | 70,445 | 703,340 | -632,895 | -89.98% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-010 | 410,907 | 940,665 | -529,758 | -56.32% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-011 | 69,949 | 62,232 | +7,717 | +12.40% | skill 1/1; base 1/1 |
| claude-code | nvmolkit-usage-012 | 233,316 | 545,439 | -312,123 | -57.22% | skill 1/1; base 1/1 |
| codex | All cases | 906,064 | 616,919 | +289,145 | +46.87% | skill 12/12; base 12/12 |
| codex | nvmolkit-usage-001 | 43,170 | 37,634 | +5,536 | +14.71% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-002 | 43,584 | 35,222 | +8,362 | +23.74% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-003 | 110,617 | 107,635 | +2,982 | +2.77% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-004 | 75,044 | 59,366 | +15,678 | +26.41% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-005 | 109,641 | 108,861 | +780 | +0.72% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-006 | 50,706 | 25,538 | +25,168 | +98.55% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-007 | 75,072 | 35,931 | +39,141 | +108.93% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-008 | 110,578 | 62,718 | +47,860 | +76.31% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-009 | 31,315 | 18,644 | +12,671 | +67.96% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-010 | 114,650 | 31,606 | +83,044 | +262.75% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-011 | 69,747 | 20,511 | +49,236 | +240.05% | skill 1/1; base 1/1 |
| codex | nvmolkit-usage-012 | 71,940 | 73,253 | -1,313 | -1.79% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,859,232 | 7,950,783 | -2,091,551 | -26.31% | skill 24/24; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/nvmolkit-usage/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nvmolkit-usage/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Large skill (5290 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/nvmolkit-usage/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'risk_tier' is not kebab-case (`skills/nvmolkit-usage/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nvmolkit-usage/SKILL.md`)
- 8 additional finding(s) are available in the full evaluation artifacts.

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
