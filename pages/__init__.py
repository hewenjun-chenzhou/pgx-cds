"""
pages/ — Streamlit 多页面模块包

把这个目录变成 Python 包，使 `pages/` 下脚本可以互相 import
（例如 2_分析报告.py 导入 _agent_section.py）。

注意：Streamlit 多页面机制会自动跳过 __init__.py 与 _ 开头的文件，
不会把它们当作独立页面注册到侧边栏。
"""