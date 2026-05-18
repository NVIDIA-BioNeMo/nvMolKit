.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Cursor / agent skill
====================

If you use `Cursor <https://cursor.com/>`_ (or another agent that supports the
``SKILL.md`` format) to write code that calls nvMolKit, the repository ships an
agent skill you can copy into your own project. The skill teaches the agent the
public Python API surface with runnable recipes for common workflows.

Get the skill
-------------

The skill lives at
`agent-skills/nvmolkit-usage/ <https://github.com/NVIDIA-Digital-Bio/nvMolKit/tree/main/agent-skills/nvmolkit-usage>`_
in the nvMolKit GitHub repo.

Copy that directory into one of:

* your project's ``.cursor/skills/`` to make it available in that project, or
* ``~/.cursor/skills/`` to make it available in every project on your machine.

The skill is loaded by the agent when it sees a request that matches its
trigger terms (``nvmolkit``, ``Morgan fingerprint``, ``ETKDG``, ``MMFF``,
``UFF``, ``Tanimoto``, ``RDKit GPU``, etc.). It does not need any further
setup.

Scope
-----

The skill is aimed at developers calling the installed nvMolKit Python API.
Building nvMolKit from source is out of scope; for that, see the
:ref:`Installation <installation>` section of the overview page.
