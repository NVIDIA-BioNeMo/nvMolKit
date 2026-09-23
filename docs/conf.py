# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import datetime
import sys
from types import ModuleType

import nvidia_sphinx_theme  # noqa

current_year = datetime.datetime.now().year

project = "nvMolKit"
author = "NVIDIA Corporation & Affiliates"

if current_year == 2025:
    copyright = f"2025, {author}"
else:
    copyright = f"2025-{current_year}, {author}"

with open("../VERSION") as version_file:
    version = version_file.read().strip()
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "jupyter_sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["README.md", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The theme to use for HTML
html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]

# -- Other options -----------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
# -- Options for autodoc -----------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints_format = "short"


class _NativePlaceholder(type):
    """Placeholder for a native attribute; nested attributes (e.g. enum members) resolve to placeholders too."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _NativePlaceholder(name, (), {"__qualname__": f"{cls.__qualname__}.{name}"})

    def __repr__(cls):
        return cls.__qualname__


class _NativeExtensionStub(ModuleType):
    """Provide importable placeholders for unavailable compiled bindings."""

    def __init__(self, name):
        super().__init__(name)
        self.__file__ = f"<{name}>"
        self.__all__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _NativePlaceholder(name, (), {"__qualname__": name})


# API docs inspect the Python wrappers but do not execute GPU operations. Stub
# the compiled modules so the docs can be built on a hosted CPU runner.
for _module_name in (
    "_arrayHelpers",
    "_batchedForcefield",
    "_clustering",
    "_conformerRmsd",
    "_DataStructs",
    "_descriptors3d",
    "_embedMolecules",
    "_Fingerprints",
    "_mcs",
    "_mmffOptimization",
    "_substructure",
    "_TFD",
    "_types",
    "_uffOptimization",
):
    _qualified_name = f"nvmolkit.{_module_name}"
    sys.modules[_qualified_name] = _NativeExtensionStub(_qualified_name)
