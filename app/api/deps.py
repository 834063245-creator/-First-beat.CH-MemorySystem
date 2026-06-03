"""FastAPI 依赖注入 — 单用户上下文。

直接从 app/ 模块导入，不操作 sys.path。
"""
from app.core.context import AppContext, ctx_manager  # noqa: F401
from app.core.auth import (
    get_current_user,
    get_user_context,
)
from app.config.settings import USER_DATA_DIRS, DATA_DIR
