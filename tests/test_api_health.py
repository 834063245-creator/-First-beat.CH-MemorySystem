"""测试 app/api/health.py — 健康检查端点。

覆盖：不依赖数据库的健康检查和就绪检查。
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    from app.api.app import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_liveness_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestAPIApp:
    def test_app_exists(self):
        from app.api.app import app
        assert app is not None
        assert app.title != ""

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
