#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# PyTorch documentation build configuration file, created by
# sphinx-quickstart on Fri Dec 23 13:31:47 2016.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys

import pytorch_sphinx_theme
import torchcodec

sys.path.append(os.path.abspath("."))

# -- General configuration ------------------------------------------------

# Required version of sphinx is set from docs/requirements.txt

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.duration",
    "sphinx_gallery.gen_gallery",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "sphinx_copybutton",
]

sphinx_gallery_conf = {
    "examples_dirs": "../../examples/",  # path to your example scripts
    "gallery_dirs": "generated_examples",  # path to where to save gallery generated output
    "filename_pattern": ".py",
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("torchcodec",),
    "remove_config_comments": True,
}

# We override sphinx-gallery's example header to prevent sphinx-gallery from
# creating a note at the top of the renderred notebook.
# https://github.com/sphinx-gallery/sphinx-gallery/blob/451ccba1007cc523f39cbcc960ebc21ca39f7b75/sphinx_gallery/gen_rst.py#L1267-L1271
# This is because we also want to add a link to google collab, so we write our own note in each example.
from sphinx_gallery import gen_rst

gen_rst.EXAMPLE_HEADER = """
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "{0}"
.. LINE NUMBERS ARE GIVEN BELOW.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_{1}:

"""

napoleon_use_ivar = True
napoleon_numpy_docstring = False
napoleon_google_docstring = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst"]

html_title = f"TorchCodec {torchcodec.__version__} Documentation"

# The master toctree document.
master_doc = "index"

# General information about the project.
copyright = "2023-present, TorchCodec Contributors"
author = "Torch Contributors"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "pytorch_project": "docs",
    "navigation_with_keys": True,
    "analytics_id": "GTM-T8XT4PS",
}

html_logo = "_static/img/pytorch-logo-dark.svg"

html_css_files = ["css/custom_torchcodec.css"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "PyTorchdoc"

autosummary_generate = True

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}


def inject_minigalleries(app, what, name, obj, options, lines):
    """Inject a minigallery into a docstring.

    This avoids having to manually write the .. minigallery directive for every item we want a minigallery for,
    as it would be easy to miss some.

    This callback is called after the .. auto directives (like ..autoclass) have been processed,
    and modifies the lines parameter inplace to add the .. minigallery that will show which examples
    are using which object.

    It's a bit hacky, but not *that* hacky when you consider that the recommended way is to do pretty much the same,
    but instead with templates using autosummary (which we don't want to use):
    (https://sphinx-gallery.github.io/stable/configuration.html#auto-documenting-your-api-with-links-to-examples)

    For docs on autodoc-process-docstring, see the autodoc docs:
    https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
    """

    if what in ("class", "function"):
        lines.append(f".. minigallery:: {name}")
        lines.append(f"    :add-heading: Examples using ``{name.split('.')[-1]}``:")
        # avoid heading entirely to avoid warning. As a bonud it actually renders better
        lines.append("    :heading-level: 9")
        lines.append("\n")


def setup(app):
    app.connect("autodoc-process-docstring", inject_minigalleries)
