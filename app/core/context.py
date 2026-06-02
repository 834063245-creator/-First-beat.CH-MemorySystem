"""应用上下文 — AppContext 和 ctx_manager。

迁移桥接层：从 backend/main.py 导入 AppContext（逐步拆解至独立）。
New API code (deps.py) 通过此模块引用，避免直接操作 sys.path。
"""
import os
import sys

# 确保 backend/ 可导入（仅桥接层做此操作，api/deps.py 不再需要）
_backend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_backend_dir))

from main import AppContext  # noqa: E402, F401
from user_context import ctx_manager  # noqa: E402, F401
