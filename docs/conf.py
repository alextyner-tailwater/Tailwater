"""Sphinx configuration for the tailwater documentation site.

This config is what Read the Docs runs (per ``.readthedocs.yaml``). The
key decision here is `autodoc_mock_imports`: the heavy ML dependencies
(`torch`, `e3nn`, `torch_geometric`, `tbmodels`, `pybinding`,
`scipy`, ...) are mocked rather than installed, so the RTD build runs
fast and stays under the free-tier memory limit. Autodoc only needs
the import to "succeed" — it then introspects the docstrings from the
package's source, not from a real run.
"""

import os
import sys
from datetime import datetime


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
# The package lives under src/tailwater/. We point sys.path there so
# `autodoc` can import the package as `tailwater.*` directly. The same
# layout is what `pip install -e .` produces under the hood.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


# ---------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------
project   = "tailwater"
author    = "Tailwater"
copyright = f"{datetime.now().year}, {author}"

# Pulled live from the installed package so docs version always
# matches the code on the branch RTD is building.
try:
    from tailwater import __version__ as release
except Exception:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])


# ---------------------------------------------------------------------
# Sphinx extensions
# ---------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",          # pull from docstrings
    "sphinx.ext.autosummary",      # auto-generate summary tables
    "sphinx.ext.napoleon",         # parse Google / NumPy-style docstrings
    "sphinx.ext.viewcode",         # add [source] links
    "sphinx.ext.intersphinx",      # cross-link to numpy / torch / pymatgen docs
    "sphinx_autodoc_typehints",    # render type hints in the signature
    "sphinx_copybutton",           # copy-to-clipboard on code blocks
    "myst_parser",                 # support .md alongside .rst
]

autosummary_generate = True
autodoc_default_options = {
    "members":           True,
    "undoc-members":     False,
    "show-inheritance":  True,
    "member-order":      "bysource",
}

# Allow Sphinx to find the package's heavy deps without installing them.
# Each Mock entry "succeeds" as an import; downstream references like
# `torch.Tensor` become Mock objects whose attribute access is silent.
# This lets autodoc parse the source signatures even when the underlying
# library isn't on PYTHONPATH.
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "e3nn",
    "tbmodels",
    "pybinding",
    "pymatgen",
    "scipy",
    "h5py",
    "matplotlib",
    "tqdm",
    "seekpath",
    "requests",
    "bcrypt",
]

# Allow Markdown source files alongside reStructuredText.
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

# ---------------------------------------------------------------------
# Intersphinx — cross-links to external docs
# ---------------------------------------------------------------------
intersphinx_mapping = {
    "python":   ("https://docs.python.org/3",          None),
    "numpy":    ("https://numpy.org/doc/stable",       None),
    "scipy":    ("https://docs.scipy.org/doc/scipy",   None),
    "torch":    ("https://pytorch.org/docs/stable",    None),
    "pymatgen": ("https://pymatgen.org",               None),
}


# ---------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------
html_theme        = "sphinx_rtd_theme"
html_title        = f"tailwater {release}"
html_short_title  = "tailwater"
html_show_sphinx  = False
templates_path    = ["_templates"]
exclude_patterns  = ["_build", "Thumbs.db", ".DS_Store"]

# Theme options — collapse-by-default sidebar with the four top-level
# sections always visible.
html_theme_options = {
    "navigation_depth":      3,
    "collapse_navigation":   False,
    "sticky_navigation":     True,
    "titles_only":           False,
}


# ---------------------------------------------------------------------
# Napoleon (Google / NumPy docstring) tweaks
# ---------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring  = True
napoleon_use_param        = True
napoleon_use_rtype        = True


# ---------------------------------------------------------------------
# Type-hint rendering
# ---------------------------------------------------------------------
# `sphinx_autodoc_typehints` inlines type hints into the rendered
# signature. With mocked imports many hints resolve to opaque Mock
# objects; we set typehints_fully_qualified = False so the rendered
# label is just "Tensor" rather than "Mock.Tensor".
always_document_param_types = True
typehints_fully_qualified   = False
typehints_use_signature     = True


# ---------------------------------------------------------------------
# MyST (Markdown) settings
# ---------------------------------------------------------------------
myst_enable_extensions = [
    "deflist",              # support definition lists in .md files
    "colon_fence",          # ::: directive blocks
    "linkify",              # turn bare URLs into hyperlinks
]
