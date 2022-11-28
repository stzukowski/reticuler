# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from typing import MutableMapping

def get_meta() -> MutableMapping:
    """Get project metadata from pyproject.toml file.
    Returns:
        MutableMapping
    """
    import toml

    toml_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")

    with open(toml_path) as fopen:
        pyproject = toml.load(fopen)

    return pyproject

sys.path.insert(0, os.path.abspath('../../src'))
meta = get_meta()

# -- Project information -----------------------------------------------------

project = meta["project"]["name"]
author = ",".join(meta["project"]["authors"])
copyright = f"2022, {author}"

# The full version, including alpha/beta/rc tags
release = meta["project"]["version"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon', # Support for NumPy and Google style docstrings # instead of 'numpydoc'
    'sphinx.ext.imgmath', # render math as images
    'sphinx_copybutton', # adds copybutton to blocks of code
    'sphinx-prompt',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = "pydata_sphinx_theme"
html_sidebars = {
   'index': [],  # Hide sidebar
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']