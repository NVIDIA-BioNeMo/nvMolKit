# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the CI runtime selection; run in an activated Conda env."""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("use_conda_runtime.sh")


class CondaRuntimeTests(unittest.TestCase):
    """Verify runtime search order and the ABI required by RDKit."""

    def _shell(self, command, **overrides):
        env = os.environ.copy()
        env.pop("LDFLAGS", None)
        env.pop("LD_LIBRARY_PATH", None)
        env.update(overrides)
        return subprocess.run(
            ["bash", "-eu", "-c", f"source {shlex.quote(str(SCRIPT))}\n{command}"],
            env=env,
            capture_output=True,
            text=True,
        )

    def test_paths_without_existing_flags(self):
        """Set both search paths in a clean environment."""
        result = self._shell('printf "%s\\n" "$LDFLAGS" "$LD_LIBRARY_PATH"', CONDA_PREFIX="/ci/conda")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["-L/ci/conda/lib -Wl,-rpath,/ci/conda/lib", "/ci/conda/lib"])

    def test_conda_precedes_existing_paths_and_preserves_flags(self):
        """Prefer Conda while retaining CUDA paths and existing linker options."""
        result = self._shell(
            'printf "%s\\n" "$LDFLAGS" "$LD_LIBRARY_PATH"',
            CONDA_PREFIX="/ci/conda",
            LDFLAGS="-L/old/lib -Wl,--as-needed",
            LD_LIBRARY_PATH="/usr/local/cuda/lib64:/old/lib",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "-L/ci/conda/lib -Wl,-rpath,/ci/conda/lib -L/old/lib -Wl,--as-needed",
                "/ci/conda/lib:/usr/local/cuda/lib64:/old/lib",
            ],
        )

    def test_empty_prefix_fails(self):
        """Reject an empty environment prefix before exporting invalid paths."""
        result = self._shell("true", CONDA_PREFIX="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Activate the CI conda environment first", result.stderr)

    def test_missing_prefix_fails(self):
        """Require activation before selecting the runtime."""
        result = subprocess.run(
            ["bash", "-eu", "-c", f"unset CONDA_PREFIX; source {shlex.quote(str(SCRIPT))}"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Activate the CI conda environment first", result.stderr)

    def test_cmake_links_and_loads_required_abi(self):
        """Link and load a real reference to the ABI symbol RDKit requires."""
        # Use a real linker and loader. This fails with an older runtime even
        # if the environment-variable assertions above all pass.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.30)\n"
                "project(runtime_check LANGUAGES CXX)\n"
                "add_executable(runtime_check main.cpp)\n"
            )
            (source / "main.cpp").write_text(
                'extern "C" void __cxa_call_terminate(void*);\n'
                "void (*volatile required_symbol)(void*) = &__cxa_call_terminate;\n"
                "int main() { return required_symbol == nullptr; }\n"
            )
            quoted = shlex.quote(directory)
            result = self._shell(
                f"cmake -S {quoted} -B {quoted}/build\n"
                f"cmake --build {quoted}/build\n"
                # Remove the override to verify that the embedded rpath works.
                f"env -u LD_LIBRARY_PATH {quoted}/build/runtime_check"
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
