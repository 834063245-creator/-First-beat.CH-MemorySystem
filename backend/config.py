"""Configuration — backend 兼容层，继承自 app/config/settings.py。

只覆盖 backend 特有的 BASE_DIR 和 USER_DATA_DIRS 等路径差异，
其余全部从 app.config.settings 继承。
"""
import os
import sys

# 确保能导入 app
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.config.settings import *  # noqa: E402, F403

# ── 覆盖：backend 专用路径 ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/ 目录

USER_DATA_DIRS: dict[str, str] = {
    "admin": os.path.join(BASE_DIR, "data"),
}

# ── 覆盖：embedding 回填标记（与 settings.py 路径一致但需明确） ──
EMBED_BACKFILL_MARKER = os.path.join(DATA_DIR, ".embed_model_backfill_done")
