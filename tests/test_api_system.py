# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: 049bb7aa

"""测试 app/api/system.py — 系统端点（不需 AppContext 依赖的路径）。

覆盖：/api/ping /api/user/list /prompt GET/POST /login /api/user/switch
"""
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.api.app import app
    return TestClient(app)


class TestPing:
    def test_returns_ok(self, client):
        resp = client.get("/api/ping")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestUserList:
    @patch("app.config.settings.USER_DATA_DIRS", {"admin": "/d/adm", "alice": "/d/alice"})
    def test_returns_user_list(self, client):
        resp = client.get("/api/user/list")
        assert resp.status_code == 200
        users = resp.json()["users"]
        assert "admin" in users
        assert "alice" in users


class TestUserSwitch:
    def test_sets_cookie(self, client):
        resp = client.post("/api/user/switch?user=alice")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["user"] == "alice"
        assert "chuhen_user" in resp.cookies


class TestPromptGet:
    def test_returns_prompt_content(self, client):
        resp = client.get("/prompt")
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data

    @patch("app.api.system.open")
    def test_file_not_found_returns_empty(self, mock_open, client):
        mock_open.side_effect = FileNotFoundError()
        resp = client.get("/prompt")
        assert resp.status_code == 200
        assert resp.json()["content"] == ""


class TestPromptPost:
    @patch("builtins.open")
    def test_update_prompt_writes_content(self, mock_open, client):
        """验证 POST /prompt 路径校验通过后写入文件。"""
        resp = client.post("/prompt", json={"content": "test content"})
        assert resp.status_code == 200
        mock_open.assert_called()

    def test_content_not_string_rejected(self, client):
        resp = client.post("/prompt", json={"content": 12345})
        assert resp.status_code == 400
        assert "content" in resp.json()["detail"].lower()

    def test_content_too_long_rejected(self, client):
        resp = client.post("/prompt", json={"content": "x" * 50001})
        assert resp.status_code == 400
        assert "过长" in resp.json()["detail"]


class TestLogin:
    @patch("app.api.system._USERS", {"admin": "changeme", "alice": "pw123"})
    @patch("app.api.system._AUTH_TOKENS", {})
    @patch("app.api.system._save_auth_tokens")
    def test_login_success(self, mock_save, client):
        resp = client.post("/login", json={"username": "alice", "password": "pw123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["username"] == "alice"
        assert len(data["token"]) > 10

    @patch("app.api.system._USERS", {"admin": "changeme"})
    def test_login_wrong_password(self, client):
        resp = client.post("/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    @patch("app.api.system._USERS", {"admin": "changeme"})
    def test_login_unknown_user(self, client):
        resp = client.post("/login", json={"username": "ghost", "password": "x"})
        assert resp.status_code == 401
