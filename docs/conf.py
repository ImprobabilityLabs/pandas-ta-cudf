project = 'pandas_ta'


copyright = '2019, Kevin Johnson'


author = 'Kevin Johnson'


version = '0.0.1'


release = 'alpha'


extensions = ['sphinx.ext.todo', 'sphinx.ext.mathjax', 'sphinx.ext.viewcode']


templates_path = ['_templates']


source_suffix = '.rst'


master_doc = 'index'


language = None


exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


pygments_style = None


html_theme = 'alabaster'


html_static_path = ['_static']


htmlhelp_basename = 'pandas_tadoc'


latex_elements = {}


latex_documents = [(master_doc, 'pandas_ta.tex',
    'pandas\\_ta Documentation', 'Kevin Johnson', 'manual')]


man_pages = [(master_doc, 'pandas_ta', 'pandas_ta Documentation', [author], 1)]


texinfo_documents = [(master_doc, 'pandas_ta', 'pandas_ta Documentation',
    author, 'pandas_ta', 'One line description of project.', 'Miscellaneous')]


epub_title = project


epub_exclude_files = ['search.html']


todo_include_todos = True
