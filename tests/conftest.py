"""pytest 配置 — 确保 backend/ 在 Python 路径中，使旧 shim 导入正常工作。

旧测试文件使用 ``from circuit import ...`` 这类导入，依赖 backend/ 下的
桥接层（shim）文件 re-export 到 app/ 模块。此 conftest 注入 backend/ 到
sys.path，无需逐个修改测试文件。
"""
import os
import sys

# 项目根目录（tests/ 的上级目录）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将 backend/ 注入 sys.path，供旧导入路径使用
_backend_dir = os.path.join(_project_root, "backend")
if os.path.isdir(_backend_dir) and _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# 同时将项目根目录加入路径
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
