"""pytest 配置 — 注入项目根目录到 sys.path，确保 app.* 导入正常工作。"""
import os
import sys

# 项目根目录（tests/ 的上级目录）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将项目根目录加入路径
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
