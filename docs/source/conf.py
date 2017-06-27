# -*- coding: utf-8 -*-
#
# grond documentation build configuration file, created by
# sphinx-quickstart on Fri Jun 23 11:40:19 2017.
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


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import grond
import sphinx_sleekcat_theme


extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.imgmath',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'grond'
copyright = u'2017, The Pyrocko Developers'
author = u'The Pyrocko Developers'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = grond.__version__
# The full version, including alpha/beta/rc tags.
release = grond.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_sleekcat_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # 'githuburl': 'https://github.com/pyrocko/pyrocko/',
    'bodyfont': '"Lucida Grande",Arial,sans-serif',
    'headfont': '"Lucida Grande",Arial,sans-serif',
    'codefont': 'monospace,sans-serif',
    'linkcolor': '#204a87',
    'visitedlinkcolor': '#204a87',
    'nosidebar': True,
    # 'appendcss': open('style.css').read(),
    # 'googlewebfonturl':
    # 'https://fonts.googleapis.com/css?family=Roboto+Slab',
    # 'bodyfont': '"Roboto Slab",Arial,sans-serif',
}

pygments_style = 'friendly'

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_sleekcat_theme.get_html_theme_path()]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = u"%s v%s Manual" % (project, release)

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'gronddoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'grond.tex', u'grond Documentation',
     u'The Pyrocko Developers', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'grond', u'grond Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'grond', u'grond Documentation',
     author, 'grond', 'One line description of project.',
     'Miscellaneous'),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'numpy': ('https://docs.scipy.org/doc/numpy/',
                                 None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference/',
                                 None),
                       # 'matplotlib': ('http://matplotlib.org/',
                       #           None),
                       'python': ('https://docs.python.org/2.7',
                                  None)}


def process_signature(app, what, name, obj, options, signature,
                      return_annotation):

    from pyrocko import guts

    if what == 'attribute' and isinstance(obj, guts.TBase):
        return str(obj)

    if what == 'class' and issubclass(obj, guts.Object):
        if obj.dummy_for is not None:
            return ('(dummy)', '%s' % obj.dummy_for.__name__)

    return


def skip_member(app, what, name, obj, skip, options):
    if what == 'class' and name == 'dummy_for':
        return True


def setup(app):
    app.connect('autodoc-process-signature', process_signature)
    app.connect('autodoc-skip-member', skip_member)