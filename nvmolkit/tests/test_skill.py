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

"""Verify every Python code block in the nvmolkit-usage agent skill executes against the installed nvMolKit."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = REPO_ROOT / "agent-skills" / "nvmolkit-usage" / "SKILL.md"

_PY_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_python_blocks(skill_path: Path) -> list[str]:
    if not skill_path.is_file():
        raise FileNotFoundError(
            f"nvmolkit-usage agent skill not found at {skill_path}. "
            f"test_skill expects the skill to live at "
            f"agent-skills/nvmolkit-usage/SKILL.md relative to the repo root; "
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
