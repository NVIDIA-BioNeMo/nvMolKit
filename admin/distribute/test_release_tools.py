# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path


def load_module(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pin_wheel_rdkit = load_module("pin_wheel_rdkit")
retag_wheel = load_module("retag_wheel")
merge_simple_index = load_module("merge_simple_index")


def make_wheel(path: Path) -> None:
    dist_info = "nvmolkit-0.6.0.dist-info"
    purl = f"pkg:pypi/nvmolkit@0.6.0?file_name={path.name}"
    entries = {
        "nvmolkit/__init__.py": b"__version__ = '0.6.0'\n",
        f"{dist_info}/METADATA": (b"Metadata-Version: 2.1\nName: nvmolkit\nVersion: 0.6.0\nRequires-Dist: rdkit\n"),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nTag: cp311-cp311-manylinux_2_28_x86_64\n",
        f"{dist_info}/sboms/auditwheel.cdx.json": json.dumps(
            {
                "metadata": {
                    "component": {
                        "bom-ref": purl,
                        "purl": purl,
                        "version": "0.6.0",
                    }
                },
                "dependencies": [{"ref": purl}],
            }
        ).encode(),
        f"{dist_info}/RECORD": b"",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, data in entries.items():
            wheel.writestr(name, data)


def assert_valid_record(path: Path) -> None:
    with zipfile.ZipFile(path) as wheel:
        record_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/RECORD"))
        rows = csv.reader(io.StringIO(wheel.read(record_name).decode()))
        for name, digest, size in rows:
            if name == record_name:
                assert digest == "" and size == ""
                continue
            data = wheel.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            assert digest == f"sha256={expected}"
            assert size == str(len(data))


def test_pin_and_retag_wheel_refresh_metadata_and_record(tmp_path: Path) -> None:
    source = tmp_path / "nvmolkit-0.6.0-cp311-cp311-manylinux_2_28_x86_64.whl"
    make_wheel(source)

    assert pin_wheel_rdkit.rewrite_wheel(source, "2026.3.5", check=False)
    assert not pin_wheel_rdkit.rewrite_wheel(source, "2026.3.5", check=True)
    assert_valid_record(source)

    output = Path(retag_wheel.retag(str(source), str(tmp_path / "variants"), "rdkit2026.3.5"))
    assert output.name.startswith("nvmolkit-0.6.0+rdkit2026.3.5-")
    with zipfile.ZipFile(output) as wheel:
        metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_name).decode()
        assert "Version: 0.6.0+rdkit2026.3.5\n" in metadata
        assert "Requires-Dist: rdkit==2026.3.5\n" in metadata
        sbom_name = next(name for name in wheel.namelist() if name.endswith("auditwheel.cdx.json"))
        sbom = json.loads(wheel.read(sbom_name))
        component = sbom["metadata"]["component"]
        assert component["version"] == "0.6.0+rdkit2026.3.5"
        assert output.name in component["purl"]
        assert sbom["dependencies"][0]["ref"] == component["purl"]
    assert_valid_record(output)


def test_merge_index_preserves_history_and_replaces_matching_filename(tmp_path: Path) -> None:
    generated = tmp_path / "generated/index.html"
    destination = tmp_path / "destination/index.html"
    old_name = "nvmolkit-0.5.1-cp311.whl"
    replaced_name = "nvmolkit-0.6.0-cp311.whl"
    merge_simple_index.write_index(
        destination,
        [
            merge_simple_index.Link("https://old.example/0.5.1", old_name),
            merge_simple_index.Link("https://old.example/0.6.0", replaced_name),
        ],
    )
    merge_simple_index.write_index(
        generated,
        [merge_simple_index.Link("https://new.example/0.6.0", replaced_name)],
    )

    assert merge_simple_index.merge_index(generated, destination)
    assert merge_simple_index.read_links(destination) == [
        merge_simple_index.Link("https://old.example/0.5.1", old_name),
        merge_simple_index.Link("https://new.example/0.6.0", replaced_name),
    ]
    assert not merge_simple_index.merge_index(generated, destination)
