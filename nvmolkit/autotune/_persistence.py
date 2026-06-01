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

"""JSON save/load helpers for tuned configuration objects.

These helpers are usable without optuna installed, so users on conda-forge
can load a previously-tuned configuration and apply it without the autotune
extra.
"""

from __future__ import annotations

import json
from os import PathLike
from typing import Union

from nvmolkit.substructure import SubstructSearchConfig
from nvmolkit.types import HardwareOptions

_TYPE_TAG = "_nvmolkit_config_type"


def save(config: Union[HardwareOptions, SubstructSearchConfig], path: Union[str, PathLike]) -> None:
    """Persist ``config`` to a JSON file at ``path``.

    The serialized payload includes a type tag so :func:`load` can return the
    correct class without the caller specifying it.
    """
    if isinstance(config, HardwareOptions):
        payload = {_TYPE_TAG: "HardwareOptions", "fields": config.to_dict()}
    elif isinstance(config, SubstructSearchConfig):
        payload = {_TYPE_TAG: "SubstructSearchConfig", "fields": config.to_dict()}
    else:
        raise TypeError(
            f"Unsupported config type {type(config).__name__}; expected HardwareOptions or SubstructSearchConfig"
        )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load(path: Union[str, PathLike]) -> Union[HardwareOptions, SubstructSearchConfig]:
    """Load a configuration object previously saved with :func:`save`."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    type_tag = payload.get(_TYPE_TAG)
    fields = payload.get("fields", {})
    if type_tag == "HardwareOptions":
        return HardwareOptions.from_dict(fields)
    if type_tag == "SubstructSearchConfig":
        return SubstructSearchConfig.from_dict(fields)
    raise ValueError(f"Unrecognized configuration type tag: {type_tag!r}")
