# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LangCheck'
copyright = '2023, Citadel AI'
author = 'Citadel AI'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically generate "API Reference" pages from docstrings  # NOQA E501
    'sphinx.ext.napoleon',  # Read Google-style docstrings
    'sphinx.ext.viewcode',  # Add links from docs to source code
    'myst_parser'  # Enable Markdown support
]
myst_enable_extensions = [
    "colon_fence"  # Enable note/tip/warning "admonition" blocks in Markdown
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'LangCheck ✅'
