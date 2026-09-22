#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert CMake's NVCC commands into commands clang-tidy can parse."""

import argparse
import json
import shlex
from pathlib import Path


REMOVED_FLAGS = {
    "--extended-lambda",
    "--expt-extended-lambda",
    "--ptxas-options=-v",
    "-forward-unknown-to-host-compiler",
    "-Werror=all-warnings",
    "-Wno-deprecated-gpu-targets",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--cuda-path", required=True, type=Path)
    parser.add_argument("--cuda-architecture", required=True)
    parser.add_argument("--dependency-prefix", required=True, type=Path)
    return parser.parse_args()


def command_arguments(entry):
    if "arguments" in entry:
        return entry["arguments"]
    return shlex.split(entry["command"])


def append_host_compiler_flags(converted, flags):
    converted.extend(flag for flag in flags.split(",") if flag != "-Werror")


def convert_arguments(entry, compiler, cuda_path, cuda_architecture, dependency_prefix):
    arguments = command_arguments(entry)
    converted = [compiler]
    index = 1

    while index < len(arguments):
        argument = arguments[index]
        if argument == "--options-file":
            response_file = Path(entry["directory"], arguments[index + 1]).resolve()
            converted.append(f"@{response_file}")
            index += 2
            continue
        if argument in {"--generate-code", "-gencode", "--ptxas-options"}:
            index += 2
            continue
        if argument.startswith(("--generate-code=", "-gencode=")):
            index += 1
            continue
        if argument == "-Xcompiler":
            append_host_compiler_flags(converted, arguments[index + 1])
            index += 2
            continue
        if argument.startswith("-Xcompiler="):
            append_host_compiler_flags(converted, argument.removeprefix("-Xcompiler="))
            index += 1
            continue
        if argument == "--default-stream=per-thread":
            converted.append("-fgpu-default-stream=per-thread")
            index += 1
            continue
        if argument == "--use_fast_math":
            converted.append("-ffast-math")
            index += 1
            continue
        if argument == "-x" and arguments[index + 1] == "cu":
            converted.extend(("-x", "cuda"))
            index += 2
            continue
        if argument in REMOVED_FLAGS:
            index += 1
            continue
        converted.append(argument)
        index += 1

    converted.extend(
        (
            f"--cuda-path={cuda_path}",
            f"--cuda-gpu-arch=sm_{cuda_architecture}",
            f"-isystem{cuda_path / 'include' / 'cccl'}",
            f"-isystem{dependency_prefix / 'include'}",
        )
    )
    # Clang 22 supports CUDA 12.9 but describes that support as partial.
    converted.append("-Wno-unknown-cuda-version")
    return converted


def main():
    args = parse_args()
    database = json.loads(args.input.read_text())
    converted = []
    for entry in database:
        if not entry["file"].endswith(".cu"):
            continue
        converted.append(
            {
                "directory": entry["directory"],
                "file": entry["file"],
                "arguments": convert_arguments(
                    entry,
                    args.compiler,
                    args.cuda_path,
                    args.cuda_architecture,
                    args.dependency_prefix,
                ),
            }
        )

    if not converted:
        raise RuntimeError("compile database contains no CUDA translation units")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2) + "\n")
    print(f"Prepared {len(converted)} CUDA compile commands")


if __name__ == "__main__":
    main()
