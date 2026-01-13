project = 'GLAND'
copyright = '2024, kai'
author = 'kai'

# 扩展：启用 Markdown 支持和 RTD 主题
extensions = [
    'myst_parser',
    'sphinx_rtd_theme',
    'nbsphinx',
]
root_doc = 'index'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'sphinx_rtd_theme'
