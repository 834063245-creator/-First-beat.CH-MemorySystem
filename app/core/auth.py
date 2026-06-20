# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: c68e8479

"""单用户上下文。

通过 X-Chuhen-User 请求头或 chuhen_user cookie 切换数据目录。
"""
import json
import logging
import os
from typing import Optional

from fastapi import Depends, Header, Cookie, Request

from app.config.settings import DATA_DIR, USER_DATA_DIRS
from app.core.context import ctx_manager

logger = logging.getLogger(__name__)

_DEFAULT_USER = "admin"


def get_current_user(
    request: Request,
    x_chuhen_user: str | None = Header(None, alias="X-Chuhen-User"),
) -> str:
    """返回当前用户。优先级：Header > Cookie > 默认 admin。"""
    # 1) X-Chuhen-User 请求头
    if x_chuhen_user and x_chuhen_user in USER_DATA_DIRS:
        return x_chuhen_user
    # 2) chuhen_user cookie
    cookie_user = request.cookies.get("chuhen_user")
    if cookie_user and cookie_user in USER_DATA_DIRS:
        return cookie_user
    return _DEFAULT_USER


def get_user_context(user: str = Depends(get_current_user)):
    """依赖注入：从当前用户获取其专属 AppContext。"""
    data_dir = USER_DATA_DIRS.get(user)
    if not data_dir:
        data_dir = DATA_DIR
    return ctx_manager.get_context(user, data_dir)
