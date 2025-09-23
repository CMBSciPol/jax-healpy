# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

project = 'jax-healpy'
copyright = '2024, Pierre Chanial, Simon Biquard, Wassim Kabalan'
author = 'Pierre Chanial, Simon Biquard, Wassim Kabalan'

# Get version from setuptools_scm
try:
    from setuptools_scm import get_version

    release = get_version(root='../..')
except Exception:
    # Fallback if setuptools_scm fails
    try:
        from importlib.metadata import version

        release = version('jax-healpy')
    except Exception:
        release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'path_to_docs': 'docs/source',
    'repository_url': 'https://github.com/CMBSciPol/jax-healpy',
    'use_download_button': True,
    'use_edit_page_button': True,
    'use_issues_button': True,
    'use_repository_button': True,
    'use_sidenotes': True,
}

# Napoleon settings
napoleon_attr_annotations = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'healpy': ('https://healpy.readthedocs.io/en/latest/', None),
}

# MyST parser configuration
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'replacements',
    'smartquotes',
    'strikethrough',
    'tasklist',
]

# Nbsphinx configuration
nbsphinx_execute = 'always'
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Autodoc configuration
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Copy button configuration
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True
