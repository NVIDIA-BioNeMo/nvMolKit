## Description: <br>
Write code that calls the installed nvMolKit Python API for GPU-accelerated, batched RDKit-style operations including Morgan fingerprints, Tanimoto/cosine similarity, ETKDG conformer embedding, MMFF/UFF optimization, TFD, conformer RMSD, Butina clustering, substructure search, and maximum common substructure (MCS) search. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers writing GPU-accelerated cheminformatics code using the nvMolKit Python API for batched molecular operations such as fingerprinting, similarity search, conformer embedding, force field optimization, clustering, and substructure/MCS search. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [nvMolKit Documentation](https://nvidia-bionemo.github.io/nvMolKit/) <br>
- [nvMolKit Changelog](https://nvidia-bionemo.github.io/nvMolKit/changelog.html) <br>
- [nvMolKit Examples (Jupyter Notebooks)](https://github.com/NVIDIA-BioNeMo/nvMolKit/tree/main/examples) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
12 evaluation tasks with 3 attempts per task, each in an isolated sandbox pod. Dataset digest: sha256:ce8098e0dd2fc0698933b7d4d303fe13bdd1d9be7e87d142bdfc44baae7a9f90. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed — skill selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Whether the skill helped complete the user's goal, scored as equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage, scored as 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability, not Efficiency). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.6% | 93.4% |
| Security | 70.8% → 95.8% (+25.0 pp) | 100.0% → 100.0% (±0.0 pp) |
| Correctness | 96.7% → 100.0% (+3.3 pp) | 93.3% → 100.0% (+6.7 pp) |
| Discoverability | 94.2% | 95.0% |
| Effectiveness | 90.3% → 90.8% (+0.5 pp) | 84.3% → 86.5% (+2.2 pp) |
| Efficiency | 82.2% | 85.7% |

## Skill Version(s): <br>
0.6.0 (source: pyproject.toml, CHANGELOG, released 2026-08-13) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
