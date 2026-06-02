"""单用户上下文。

通过 X-Chuhen-User 请求头在配置的可用库之间切换。
"""
import json
import logging
import os
from typing import Optional

from fastapi import Depends, Header

from app.config.settings import DATA_DIR, USER_DATA_DIRS
from app.core.context import ctx_manager

logger = logging.getLogger(__name__)

_DEFAULT_USER = "admin"


def _load_knowledge_mode(data_dir: str = DATA_DIR) -> bool:
    """读取知识库模式开关。"""
    path = os.path.join(data_dir, "knowledge_mode.json")
    try:
        with open(path) as f:
            return json.load(f).get("enabled", False)
    except Exception:
        return False


def _save_knowledge_mode(enabled: bool, data_dir: str = DATA_DIR):
    """持久化知识库模式开关。"""
    from app.tools.atomic import atomic_write
    path = os.path.join(data_dir, "knowledge_mode.json")
    atomic_write(path, {"enabled": enabled})


def get_current_user(
    x_chuhen_user: Optional[str] = Header(None, alias="X-Chuhen-User"),
) -> str:
    """返回当前用户。可通过 X-Chuhen-User 请求头切库。"""
    if x_chuhen_user and x_chuhen_user in USER_DATA_DIRS:
        return x_chuhen_user
    return _DEFAULT_USER


def get_user_context(user: str = Depends(get_current_user)):
    """依赖注入：从当前用户获取其专属 AppContext。"""
    data_dir = USER_DATA_DIRS.get(user)
    if not data_dir:
        data_dir = DATA_DIR
    return ctx_manager.get_context(user, data_dir)
