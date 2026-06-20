# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 9bdaa9f3

"""测试 app/core/auth.py — 用户认证提取。

覆盖：get_current_user 的 Header / Cookie / 默认回退逻辑。
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request


class TestGetCurrentUser:
    def test_header_priority(self):
        from app.core.auth import get_current_user
        mock_request = MagicMock(spec=Request)
        mock_request.cookies = {}
        with patch("app.core.auth.USER_DATA_DIRS", {"user_a": "/d/a", "admin": "/d/adm"}):
            result = get_current_user(mock_request, x_chuhen_user="user_a")
            assert result == "user_a"

    def test_cookie_fallback(self):
        from app.core.auth import get_current_user
        mock_request = MagicMock(spec=Request)
        mock_request.cookies = {"chuhen_user": "user_b"}
        with patch("app.core.auth.USER_DATA_DIRS", {"user_b": "/d/b", "admin": "/d/adm"}):
            result = get_current_user(mock_request, x_chuhen_user=None)
            assert result == "user_b"

    def test_header_overrides_cookie(self):
        from app.core.auth import get_current_user
        mock_request = MagicMock(spec=Request)
        mock_request.cookies = {"chuhen_user": "user_b"}
        with patch("app.core.auth.USER_DATA_DIRS", {"user_a": "/d/a", "user_b": "/d/b", "admin": "/d/adm"}):
            result = get_current_user(mock_request, x_chuhen_user="user_a")
            assert result == "user_a"

    def test_default_admin(self):
        from app.core.auth import get_current_user
        mock_request = MagicMock(spec=Request)
        mock_request.cookies = {}
        with patch("app.core.auth.USER_DATA_DIRS", {"admin": "/d/adm"}):
            result = get_current_user(mock_request, x_chuhen_user=None)
            assert result == "admin"

    def test_unknown_header_user_falls_back(self):
        """Header 中的用户不在 USER_DATA_DIRS 中时回退。"""
        from app.core.auth import get_current_user
        mock_request = MagicMock(spec=Request)
        mock_request.cookies = {"chuhen_user": "known_user"}
        with patch("app.core.auth.USER_DATA_DIRS", {"known_user": "/d/k", "admin": "/d/adm"}):
            result = get_current_user(mock_request, x_chuhen_user="unknown_user")
            # unknown 不在 USER_DATA_DIRS → 回退到 cookie
            assert result == "known_user"

    def test_no_cookie_or_header_falls_back_admin(self):
        from app.core.auth import get_current_user
        mock_request = MagicMock(spec=Request)
        mock_request.cookies = {}
        with patch("app.core.auth.USER_DATA_DIRS", {"admin": "/d/adm"}):
            result = get_current_user(mock_request, x_chuhen_user=None)
            assert result == "admin"
