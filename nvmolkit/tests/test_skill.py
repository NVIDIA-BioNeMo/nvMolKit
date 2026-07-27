# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify the nvmolkit-usage skill snippets and portable eval definitions."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = REPO_ROOT / "skills" / "nvmolkit-usage" / "SKILL.md"
EVALS_PATH = SKILL_PATH.parent / "evals" / "evals.json"
TRIGGER_EVALS_PATH = SKILL_PATH.parent / "evals" / "trigger_evals.json"

_PY_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_python_blocks(skill_path: Path) -> list[str]:
    if not skill_path.is_file():
        raise FileNotFoundError(
            f"nvmolkit-usage agent skill not found at {skill_path}. "
            f"test_skill expects the skill to live at "
            f"skills/nvmolkit-usage/SKILL.md relative to the repo root; "
            f"update SKILL_PATH if the skill has moved."
        )
    text = skill_path.read_text()
    return [match.group(1) for match in _PY_BLOCK_RE.finditer(text)]


@pytest.mark.parametrize(
    "snippet_idx, snippet",
    list(enumerate(_extract_python_blocks(SKILL_PATH))),
)
def test_skill_snippet_runs(snippet_idx: int, snippet: str, tmp_path: Path) -> None:
    script_path = tmp_path / f"snippet_{snippet_idx}.py"
    script_path.write_text(snippet)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Skill snippet {snippet_idx} failed:\n"
        f"--- snippet ---\n{snippet}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n"
    )


def test_skill_evals_use_library_skill_schema() -> None:
    data = json.loads(EVALS_PATH.read_text())

    assert data["skill_name"] == "nvmolkit-usage"
    assert len(data["evals"]) > 0
    assert len({entry["id"] for entry in data["evals"]}) == len(data["evals"])
    for entry in data["evals"]:
        assert set(entry) == {
            "id",
            "prompt",
            "expected_output",
            "assertions",
            "expected_skill",
            "expected_script",
        }
        assert entry["id"].startswith("nvmolkit-usage-")
        assert entry["prompt"].strip()
        assert entry["expected_output"].strip()
        assert entry["assertions"]
        assert all(isinstance(assertion, str) and assertion.strip() for assertion in entry["assertions"])
        assert entry["expected_skill"] == "nvmolkit-usage"
        assert entry["expected_script"] is None


def test_skill_trigger_evals_cover_positive_and_negative_cases() -> None:
    data = json.loads(TRIGGER_EVALS_PATH.read_text())

    assert data["skill_name"] == "nvmolkit-usage"
    outcomes = {entry["expected"] for entry in data["trigger_evals"]}
    assert outcomes == {"trigger", "no_trigger"}
